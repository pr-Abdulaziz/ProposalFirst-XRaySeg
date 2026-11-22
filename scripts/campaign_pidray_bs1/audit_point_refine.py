#!/usr/bin/env python
"""Read-only audit of the PIDray point-refinement implementations (transformer and MLP).

Builds each model from its resolved PIDray config, loads its final checkpoint strictly, and runs one
small CPU forward/backward on a real training batch (one non-square image with objects, one image
without objects) plus one empty-only batch and one inference pass. Every tensor the refiner uses is
captured and re-derived independently (manual bilinear sampling), so alignment errors between P2
features, coarse logits and targets would show up as mismatches. Nothing is trained or written back.

    python scripts/campaign_pidray_bs1/audit_point_refine.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

CAMP = ROOT / "output/pidray_bs2_e10_2026-09-22"
WORK = ROOT / "work_dirs/pidray_bs2_e10_2026-09-22"
OUT = CAMP / "audit_point_refine_2026-09-26"
ARMS = {"transformer": "m09_point_refine_transformer", "mlp": "m09m_point_refine_mlp"}
STAGE_W = [1.0, 0.5, 0.25]
torch.set_num_threads(4)


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def bilinear(img, px, py):
    """Manual bilinear sampling with zero padding; img [C,H,W]; px, py pixel coords (centres at integers)."""
    C, H, W = img.shape
    x0, y0 = torch.floor(px).long(), torch.floor(py).long()
    out = torch.zeros(C, px.numel(), dtype=img.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - (px - xi.float()).abs()) * (1 - (py - yi.float()).abs())
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            v = torch.zeros(C, px.numel(), dtype=img.dtype)
            v[:, ok] = img[:, yi[ok], xi[ok]]
            out += v * w
    return out


def box_iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def build(slug):
    cfg = Config.fromfile(str(CAMP / f"protocol/resolved_configs/{slug}/seed_42.py"))
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    ck = torch.load(WORK / slug / "seed_42/epoch_10.pth", map_location="cpu")
    sd, msd = ck["state_dict"], model.state_dict()
    report = dict(
        missing=[k for k in msd if k not in sd], unexpected=[k for k in sd if k not in msd],
        shape_mismatch=[k for k in msd if k in sd and tuple(msd[k].shape) != tuple(sd[k].shape)],
        num_params_model=len(msd), num_params_ckpt=len(sd),
        ckpt_dataset_classes=list((ck.get("meta", {}).get("dataset_meta") or {}).get("classes", [])))
    model.load_state_dict(sd, strict=True)
    return cfg, model, report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import projects.proposal_first_pidray_occ.models.point_refine.head as H
    import projects.proposal_first_pidray_occ.models.point_refine.cascade_adapter as CA
    from mmdet.registry import DATASETS
    results = {}
    batch_cache = {}
    for arm, slug in ARMS.items():
        print(f"=== {arm}", flush=True)
        R = results[arm] = {}
        cfg, model, R["checkpoint"] = build(slug)
        rh = model.roi_head; ph = rh.point_refine_head
        R["dims"] = dict(
            cfg_num_classes=ph.num_classes,
            bbox_fc_cls_out=[int(b.fc_cls.out_features) for b in rh.bbox_head],
            mask_conv_logits_out=[int(m.conv_logits.out_channels) for m in rh.mask_head],
            mask_embedding_in=int(ph.embedding.mask_proj[0].in_features),
            decoder_out=int(ph.transformer.decoder.out_features),
            mixer=type(ph.transformer).__name__, stage_loss_weights=list(rh.stage_loss_weights),
            output_sizes=list(ph.output_sizes), train_points=ph.train_points, infer_points=ph.infer_points,
            point_params=int(sum(p.numel() for p in ph.parameters())))
        print(R["checkpoint"]["missing"][:3], R["checkpoint"]["unexpected"][:3], R["dims"], flush=True)

        # ---------------- dataset and the audit batch (same images for both arms)
        if not batch_cache:
            ds = DATASETS.build(cfg.train_dataloader.dataset); ds.full_init()
            info = [ds.get_data_info(i) for i in range(len(ds))]
            rng = random.Random(2026)
            order = list(range(len(ds))); rng.shuffle(order)
            nonsq = next(i for i in order if len({x["bbox_label"] for x in info[i]["instances"]}) >= 2
                         and abs(info[i]["width"] - info[i]["height"]) > 100)
            empty = next(i for i in order if len(info[i]["instances"]) == 0)
            empty2 = next(i for i in order if len(info[i]["instances"]) == 0 and i != empty)
            seed_all(7)
            for _ in range(20):
                a = ds[nonsq]
                if len(a["data_samples"].gt_instances) > 0:
                    break
            b, c = ds[empty], ds[empty2]
            batch_cache.update(ds_len=len(ds), cat2label={int(k): int(v) for k, v in ds.cat2label.items()},
                               classes=list(ds.metainfo["classes"]),
                               nonsq=dict(index=nonsq, file=info[nonsq]["img_path"].split("/")[-1],
                                          wh=[info[nonsq]["width"], info[nonsq]["height"]],
                                          instances=len(info[nonsq]["instances"])),
                               empty=[info[empty]["img_path"].split("/")[-1], info[empty2]["img_path"].split("/")[-1]],
                               empty_after_pipeline=[b is not None, c is not None],
                               samples=(a, b, c))
        a, b, c = batch_cache["samples"]
        R["batch"] = dict(nonsq=batch_cache["nonsq"], img_shapes=[s["data_samples"].img_shape for s in (a, b)],
                          gt_after_pipeline=[len(s["data_samples"].gt_instances) for s in (a, b)])

        # ---------------- capture the refiner's internals during one training step
        calls = []
        names = ["select_training_points", "select_inference_points", "sample_fine_features",
                 "sample_coarse_logits", "sample_mask_targets", "scatter_points"]
        originals = {n: getattr(H, n) for n in names}
        for n in names:
            def make(n, f):
                def g(*args, **kw):
                    r = f(*args, **kw); calls.append((n, args, kw, r)); return r
                return g
            setattr(H, n, make(n, originals[n]))
        stage_calls = []
        orig_refine = ph.refine
        def refine(*args, **kw):  # noqa: E306
            named = dict(zip(("coarse_logits", "rois", "labels", "fine_features"), args))
            stage_calls.append(dict(kw={**named, **kw}, start=len(calls)))
            return orig_refine(*args, **kw)
        ph.refine = refine
        tok_calls = []
        orig_tokens = ph._tokens
        def tokens(*args, **kw):  # noqa: E306
            t = orig_tokens(*args, **kw); tok_calls.append(t); return t
        ph._tokens = tokens
        lr_out = []
        orig_lr = ph.loss_and_refine
        def loss_and_refine(*args, **kw):  # noqa: E306
            o = orig_lr(*args, **kw); lr_out.append({k: v for k, v in o.items() if k == "loss_point"}); return o
        ph.loss_and_refine = loss_and_refine

        model.train()
        data = model.data_preprocessor(pseudo_collate([copy_sample(a), copy_sample(b)]), True)
        seed_all(11)
        losses = model.loss(data["inputs"], data["data_samples"])
        R["batch"]["input_shape"] = list(data["inputs"].shape)
        R["batch"]["gt_mask_shapes"] = [tuple(ds_.gt_instances.masks.masks.shape) if len(ds_.gt_instances) else None
                                        for ds_ in data["data_samples"]]
        finite = {k: bool(torch.isfinite(v).all()) for k, v in losses.items() if torch.is_tensor(v)}
        R["losses_finite"] = all(finite.values())
        R["loss_keys_with_loss"] = sorted(k for k in losses if "loss" in k)

        # stage weighting: s{i}.loss_point == unweighted head loss * w_i
        R["stage_weighting"] = [dict(stage=i, head=float(lr_out[i]["loss_point"]),
                                     reported=float(losses[f"s{i}.loss_point"]),
                                     ratio=float(losses[f"s{i}.loss_point"] / lr_out[i]["loss_point"])
                                     if float(lr_out[i]["loss_point"]) != 0 else None, expected=STAGE_W[i])
                                for i in range(3)]

        # independent re-derivation of features, coarse tokens and targets
        align = []
        for si, sc in enumerate(stage_calls):
            kw = sc["kw"]
            coarse, rois, labels, fmap, gtm = kw["coarse_logits"], kw["rois"], kw["labels"], kw["fine_features"], kw["gt_masks"]
            seg = calls[sc["start"]:stage_calls[si + 1]["start"] if si + 1 < len(stage_calls) else len(calls)]
            feats = [x for x in seg if x[0] == "sample_fine_features"]
            ctoks = [x for x in seg if x[0] == "sample_coarse_logits"]
            tgts = [x for x in seg if x[0] == "sample_mask_targets"]
            sels = [x for x in seg if x[0] == "select_training_points"]
            for lvl in range(len(feats)):
                coords = feats[lvl][1][2]; fs = feats[lvl][3]; ct = ctoks[lvl][3]; tg = tgts[lvl][3]
                idx = sels[lvl][3]
                N, K = coords.shape[:2]
                g = torch.Generator().manual_seed(si * 10 + lvl)
                pick = torch.randint(0, K, (min(40, K),), generator=g)
                fd, cd, td, assign_ok = 0.0, 0.0, 0.0, []
                for n in range(N):
                    b_ = int(rois[n, 0]); x1, y1, x2, y2 = rois[n, 1:].tolist()
                    u = coords[n, pick]
                    ax = x1 + u[:, 0] * (x2 - x1); ay = y1 + u[:, 1] * (y2 - y1)
                    man_f = bilinear(fmap[b_], ax / 4 - 0.5, ay / 4 - 0.5).T
                    fd = max(fd, float((man_f - fs[n, pick]).abs().max()))
                    man_c = bilinear(coarse[n], u[:, 0] * coarse.shape[-1] - 0.5, u[:, 1] * coarse.shape[-2] - 0.5).T
                    cd = max(cd, float((man_c - ct[n, pick]).abs().max()))
                    man_t = bilinear(gtm[n:n + 1].float(), ax - 0.5, ay - 0.5)[0]
                    td = max(td, float((man_t - tg[n, pick]).abs().max()))
                    # which ground-truth instance is it? highest-IoU box of the same class in that image
                    gi = data["data_samples"][b_].gt_instances
                    ious = box_iou(rois[n:n + 1, 1:], gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes)[0]
                    ious[gi.labels != labels[n]] = -1
                    best = int(ious.argmax())
                    mask_best = torch.as_tensor(gi.masks.masks[best]).float()
                    assign_ok.append(bool(torch.equal(mask_best, gtm[n].float()[:mask_best.shape[0], :mask_best.shape[1]])))
                align.append(dict(
                    stage=si, level=lvl, grid=int(ph.output_sizes[lvl]), rois=N, points=K,
                    roi_image_index=sorted({int(v) for v in rois[:, 0]}), labels=[int(labels.min()), int(labels.max())] if N else None,
                    coords_range=[float(coords.min()), float(coords.max())],
                    unique_indices=bool(all(len(set(r.tolist())) == len(r) for r in idx)),
                    feature_max_abs_diff=fd, coarse_token_max_abs_diff=cd, target_max_abs_diff=td,
                    target_range=[float(tg.min()), float(tg.max())],
                    gt_mask_is_assigned_instance=f"{sum(assign_ok)}/{len(assign_ok)}",
                    roi_within_padded_image=bool((rois[:, 1] >= -1).all() and (rois[:, 3] <= data['inputs'].shape[-1] + 1).all()
                                                 and (rois[:, 2] >= -1).all() and (rois[:, 4] <= data['inputs'].shape[-2] + 1).all())))
        R["alignment"] = align

        # attention confinement on real tokens (stage 0, level 0)
        tk = tok_calls[0].detach()
        mixer = ph.transformer.eval()
        with torch.no_grad():
            base = mixer(tk)
            j = tk.shape[0] - 1
            pert = tk.clone(); pert[j] += torch.randn_like(pert[j])
            o = mixer(pert)
            other_rois = float((o[:j] - base[:j]).abs().max()) if j > 0 else None
            pert2 = tk.clone(); pert2[0, 0] += torch.randn_like(pert2[0, 0])
            o2 = mixer(pert2)
            same_roi_other_points = float((o2[0, 1:] - base[0, 1:]).abs().max())
        R["attention_confinement"] = dict(rois=int(tk.shape[0]), points=int(tk.shape[1]),
                                          change_in_other_rois=other_rois,
                                          change_in_other_points_same_roi=same_roi_other_points)
        mixer.train()

        # gradient policy of the point loss alone
        model.zero_grad(set_to_none=True)
        sum(losses[f"s{i}.loss_point"] for i in range(3)).backward()
        groups = {}
        for name, p in model.named_parameters():
            g_ = 0.0 if p.grad is None else float(p.grad.norm())
            key = ".".join(name.split(".")[:3]) if name.startswith("roi_head") else name.split(".")[0]
            groups[key] = groups.get(key, 0.0) + g_ ** 2
        R["point_loss_grad_norm_by_module"] = {k: round(v ** 0.5, 6) for k, v in sorted(groups.items())}
        model.zero_grad(set_to_none=True)

        # empty-only batch
        calls.clear(); stage_calls.clear(); lr_out.clear()
        data_e = model.data_preprocessor(pseudo_collate([copy_sample(c), copy_sample(b)]), True)
        seed_all(12)
        le = model.loss(data_e["inputs"], data_e["data_samples"])
        R["empty_only_batch"] = dict(
            finite=all(bool(torch.isfinite(v).all()) for v in le.values() if torch.is_tensor(v)),
            loss_point=[float(le[f"s{i}.loss_point"]) for i in range(3)],
            loss_point_requires_grad=[bool(le[f"s{i}.loss_point"].requires_grad) for i in range(3)],
            loss_mask=[float(le[f"s{i}.loss_mask"]) for i in range(3)],
            loss_cls=[float(le[f"s{i}.loss_cls"]) for i in range(3)])
        tot = sum(v for k, v in le.items() if "loss" in k and torch.is_tensor(v))
        model.zero_grad(set_to_none=True); tot.backward()
        R["empty_only_batch"]["backward_ok"] = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                   for p in model.parameters())
        model.zero_grad(set_to_none=True)

        # inference refinement loop on the training batch's stage-0 inputs
        kw0 = stage_calls_first = None
        calls.clear(); stage_calls.clear()
        model.eval()
        with torch.no_grad():
            feats = model.extract_feat(data["inputs"])
            gi = data["data_samples"][0].gt_instances
            rois = torch.cat([torch.zeros(len(gi), 1), gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes], 1)
            coarse = rh._mask_forward(0, feats, rois)["mask_preds"]
            refined = ph.predict_refine(coarse_logits=coarse, rois=rois, labels=gi.labels, fine_features=feats[0], stage=0)
        sel = [x for x in calls if x[0] == "select_inference_points"]
        sc_ = [x for x in calls if x[0] == "sample_coarse_logits"]
        scat = [x for x in calls if x[0] == "scatter_points"]
        chk = {}
        cur0 = F.interpolate(coarse, size=(56, 56), mode="bilinear", align_corners=False)
        chk["level0_selects_on_interpolated_coarse"] = bool(torch.equal(sel[0][1][0], cur0))
        chk["level1_selects_on_refined_level0"] = bool(torch.equal(
            sel[1][1][0], F.interpolate(scat[0][3], size=(112, 112), mode="bilinear", align_corners=False)))
        chk["tokens_use_original_coarse"] = all(x[1][0] is coarse for x in sc_)
        for lv, sx in enumerate(scat):
            grid, idx, vals, new = sx[1][0], sx[1][1], sx[1][2], sx[3]
            n, c_, h, w = grid.shape
            fg, fn = grid.reshape(n, c_, -1), new.reshape(n, c_, -1)
            m = torch.zeros(n, h * w, dtype=torch.bool); m.scatter_(1, idx, True)
            chk[f"level{lv}_unselected_kept"] = bool(torch.equal(fg.permute(0, 2, 1)[~m], fn.permute(0, 2, 1)[~m]))
            chk[f"level{lv}_selected_written"] = bool(torch.allclose(
                fn.gather(2, idx.unsqueeze(1).expand(-1, c_, -1)), vals.permute(0, 2, 1)))
            chk[f"level{lv}_points"] = int(idx.shape[1])
        chk["refined_shape"] = list(refined.shape)
        R["inference_refinement"] = chk

        # end-to-end predict on the non-square training image (test-time path, rescale to original)
        seed_all(13)
        if "test_sample" not in batch_cache:
            tds = DATASETS.build(cfg.test_dataloader.dataset); tds.full_init()
            tinfo = [tds.get_data_info(i) for i in range(len(tds))]
            ti = next(i for i in range(len(tds)) if len(tinfo[i]["instances"]) >= 1
                      and abs(tinfo[i]["width"] - tinfo[i]["height"]) > 50)
            batch_cache["test_sample"] = tds[ti]
            batch_cache["test_image"] = dict(file=tinfo[ti]["img_path"].split("/")[-1],
                                             wh=[tinfo[ti]["width"], tinfo[ti]["height"]],
                                             instances=len(tinfo[ti]["instances"]))
        data_p = model.data_preprocessor(pseudo_collate([copy_sample(batch_cache["test_sample"])]), False)
        merges = []
        orig_merge = CA.merge_aug_masks
        def merge(aug, meta):  # noqa: E306
            r = orig_merge(aug, meta); merges.append(([m.clone() for m in aug], r)); return r
        CA.merge_aug_masks = merge
        with torch.no_grad():
            out = model.predict(data_p["inputs"], data_p["data_samples"], rescale=True)[0]
        CA.merge_aug_masks = orig_merge
        pi = out.pred_instances
        aug, merged = merges[0]
        R["predict"] = dict(
            ori_shape=list(out.ori_shape), img_shape=list(out.img_shape), detections=len(pi),
            labels_range=[int(pi.labels.min()), int(pi.labels.max())] if len(pi) else None,
            mask_shape=list(pi.masks.shape), masks_match_ori_shape=list(pi.masks.shape[1:]) == list(out.ori_shape),
            boxes_within_image=bool((pi.bboxes[:, 0] >= -1).all() and (pi.bboxes[:, 2] <= out.ori_shape[1] + 1).all()
                                    and (pi.bboxes[:, 1] >= -1).all() and (pi.bboxes[:, 3] <= out.ori_shape[0] + 1).all()),
            stage_masks_in_0_1=all(bool((m >= 0).all() and (m <= 1).all()) for m in aug),
            merged_is_mean_of_stages=bool(torch.allclose(merged, torch.stack(aug).mean(0), atol=1e-6)),
            stage_mask_shape=list(aug[0].shape))

        # restore patched names
        for n in names:
            setattr(H, n, originals[n])
        del ph.refine, ph._tokens, ph.loss_and_refine
        print(json.dumps({k: v for k, v in R.items() if k not in ("alignment",)}, default=str)[:1500], flush=True)
        for row in R["alignment"]:
            print(row, flush=True)

    meta = {k: v for k, v in batch_cache.items() if k not in ("samples", "test_sample")}
    (OUT / "audit_point_refine.json").write_text(json.dumps(dict(dataset=meta, arms=results), indent=1, default=str))
    print("dataset:", json.dumps(meta, default=str))
    return 0


def copy_sample(s):
    import copy
    return copy.deepcopy(s)


if __name__ == "__main__":
    sys.exit(main())
