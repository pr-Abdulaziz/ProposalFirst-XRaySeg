#!/usr/bin/env python
"""Inference-geometry preflight: masks are returned in original-image coordinates.

A 160-microbatch preflight model often has no detection above the frozen score
threshold, so the main preflight cannot always exercise mask pasting. This check
loads the method's preflight checkpoint (preflight/<method>/work/epoch_1.pth) into
the frozen seed-42 config with one preflight-only change, score_thr = 0.0, and runs
the frozen test pipeline on difficult test images.

Per image: the mask tensor has the original image size; at most 100 detections;
every pixel of every non-empty mask lies inside its predicted box (clipped to the
image, 2 px tolerance), which holds only if masks and boxes share original-image
coordinates. Box overflow past the image edge is recorded.

    python scripts/campaign_multiseed/preflight_inference_geometry.py <method id>
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import Runner, load_checkpoint  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import BY_ID, OUT, resolved_config, sha256  # noqa: E402
from scripts.campaign_multiseed.preflight import HARD_TEST, dataset_positions  # noqa: E402

EXTRA_TEST = [0, 500, 1000, 1400]


def sam_synthetic_geometry(cfg) -> dict:
    """Map a known rectangle through the frozen SAM mask post-processing for every test image size.

    A logit canvas of the padded input size is filled with a rectangle placed at the resized-image
    position of a known original-image rectangle; the head's own ``_predict_by_feat_single`` must
    return that rectangle in original-image coordinates, and must divide the box by scale_factor once.
    The metadata (ori_shape, scale_factor, batch_input_shape) come from the frozen test pipeline and
    data preprocessor, one image per distinct test size.
    """
    from mmengine.config import ConfigDict
    from mmdet.registry import DATASETS, MODELS
    from mmdet.occlu import models as occ_models
    from projects.proposal_first_pidray_occ.models.weighted_bilayer_mask_head import WeightedOcPrompterAnchorMaskHead
    head_cls = MODELS.get(cfg.model["roi_head"]["mask_head"]["type"])
    fn = occ_models.RSPrompterAnchorMaskHead._predict_by_feat_single
    shared = head_cls._predict_by_feat_single is fn and WeightedOcPrompterAnchorMaskHead._predict_by_feat_single is fn
    legacy = bool(occ_models._legacy_sam_geometry())
    ds = DATASETS.build(copy.deepcopy(cfg.test_dataloader.dataset))
    seen, picks = set(), []
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        if (info["width"], info["height"]) not in seen:
            seen.add((info["width"], info["height"]))
            picks.append(i)
    pre = MODELS.build(copy.deepcopy(cfg.model["data_preprocessor"]))
    rows, passed = [], shared and not legacy
    rcnn_cfg = ConfigDict(cfg.model["test_cfg"]["rcnn"])
    for i in picks:
        item = ds[i]
        batch = pre(dict(inputs=[item["inputs"]], data_samples=[item["data_samples"]]), False)
        meta = batch["data_samples"][0].metainfo
        h, w = meta["ori_shape"]
        sw, sh = meta["scale_factor"]
        bh, bw = meta["batch_input_shape"]
        ox0, oy0, ox1, oy1 = round(0.21 * w), round(0.27 * h), round(0.63 * w), round(0.71 * h)
        rx0, ry0, rx1, ry1 = round(ox0 * sw), round(oy0 * sh), round(ox1 * sw), round(oy1 * sh)
        logits = torch.full((1, 1, bh, bw), -20.0)
        logits[..., ry0:ry1, rx0:rx1] = 20.0
        boxes = torch.tensor([[rx0, ry0, rx1, ry1]], dtype=torch.float32)
        mask = fn(None, logits, boxes, torch.zeros(1, dtype=torch.long), meta, rcnn_cfg, rescale=True)
        expected = torch.zeros(h, w, dtype=torch.bool)
        expected[oy0:oy1, ox0:ox1] = True
        got = mask[0].bool()
        iou = float((got & expected).sum() / (got | expected).sum())
        box_err = float((boxes[0] - torch.tensor([ox0, oy0, ox1, oy1], dtype=torch.float32)).abs().max())
        ok = list(mask.shape[1:]) == [h, w] and iou >= 0.97 and box_err <= 1.5
        passed &= ok
        rows.append(dict(ori_size=[w, h], scale_factor=[sw, sh], batch_input_shape=[bh, bw],
                         mask_shape=list(mask.shape), rectangle_iou=iou, box_error_px=box_err, ok=ok))
    return dict(passed=bool(passed), head_function_shared_by_05_06=shared, legacy_geometry_enabled=legacy,
                distinct_test_sizes=len(rows), min_iou=min(r["rectangle_iou"] for r in rows),
                max_box_error_px=max(r["box_error_px"] for r in rows), sizes=rows)


def main() -> int:
    mid = sys.argv[1]
    m = BY_ID[mid]
    cfg_path = resolved_config(mid, 42)
    ckpt = OUT / "preflight" / m.slug / "work" / "epoch_1.pth"
    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    torch.cuda.empty_cache()
    frozen_thr = cfg.model["test_cfg"]["rcnn"]["score_thr"]
    cfg.model["test_cfg"]["rcnn"]["score_thr"] = 0.0
    from mmdet.registry import MODELS
    model = MODELS.build(copy.deepcopy(cfg.model))
    load_checkpoint(model, str(ckpt), map_location="cpu", strict=True, logger="silent")
    model = model.cuda().eval()

    from mmdet.registry import DATASETS
    ds_full = DATASETS.build(copy.deepcopy(cfg.test_dataloader.dataset))
    ids = HARD_TEST + [ds_full.get_data_info(i)["img_id"] for i in EXTRA_TEST]
    del ds_full
    idx, _ = dataset_positions(cfg.test_dataloader.dataset, ids)
    tds = copy.deepcopy(cfg.test_dataloader.dataset)
    tds["indices"] = idx
    loader = Runner.build_dataloader(dict(batch_size=1, num_workers=2, persistent_workers=False,
                                          sampler=dict(type="DefaultSampler", shuffle=False), dataset=tds), seed=0)
    rows, ok_all, nonempty_total = [], True, 0
    for batch in loader:
        with torch.no_grad():
            out = model.test_step(batch)[0]
        pred = out.pred_instances
        h, w = out.metainfo["ori_shape"]
        masks, boxes = pred.masks, pred.bboxes
        shape_ok = list(masks.shape[1:]) == [h, w]
        fractions, empty = [], 0
        for i in range(len(pred)):
            ys, xs = torch.nonzero(masks[i], as_tuple=True)
            if len(xs) == 0:
                empty += 1
                continue
            x1, y1 = max(float(boxes[i, 0]), 0.0), max(float(boxes[i, 1]), 0.0)
            x2, y2 = min(float(boxes[i, 2]), float(w)), min(float(boxes[i, 3]), float(h))
            inside = ((xs >= x1 - 2) & (xs <= x2 + 2) & (ys >= y1 - 2) & (ys <= y2 + 2)).float().mean()
            fractions.append(float(inside))
        nonempty_total += len(fractions)
        overflow = 0.0 if len(boxes) == 0 else float(max(-boxes[:, 0].min(), -boxes[:, 1].min(),
                                                          boxes[:, 2].max() - w, boxes[:, 3].max() - h, 0))
        # FCN heads paste each mask inside its box, so containment proves a shared original frame.
        # SAM heads predict image-level masks from prompts; their pixels need not lie in the box, and
        # their frame is verified separately with a synthetic mask through the real post-processing.
        contained = m.family == "sam" or not fractions or min(fractions) >= 0.99
        ok = shape_ok and len(pred) <= 100 and contained
        ok_all &= ok
        rows.append(dict(img_id=out.metainfo["img_id"], ori_shape=[h, w], img_shape=list(out.metainfo["img_shape"]),
                         scale_factor=list(out.metainfo["scale_factor"]), detections=len(pred),
                         mask_shape=list(masks.shape), nonempty_masks=len(fractions), empty_masks=empty,
                         min_mask_fraction_inside_box=min(fractions) if fractions else None,
                         box_overflow_px=overflow, ok=ok))
    synthetic = sam_synthetic_geometry(cfg) if m.family == "sam" else None
    if synthetic is not None:
        ok_all &= synthetic["passed"]
    status = "PASS" if ok_all and nonempty_total > 0 else "FAIL"
    res = dict(method=mid, config=str(cfg_path), config_sha256=sha256(cfg_path), checkpoint=str(ckpt),
               checkpoint_sha256=sha256(ckpt), job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
               preflight_only_change=f"test_cfg.rcnn.score_thr {frozen_thr} -> 0.0", images=rows,
               nonempty_masks_checked=nonempty_total, sam_synthetic_geometry=synthetic, status=status,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUT / "preflight" / m.slug / "inference_geometry.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(method=mid, status=status, nonempty_masks=nonempty_total,
                          max_overflow=max(r["box_overflow_px"] for r in rows),
                          min_inside=min((r["min_mask_fraction_inside_box"] for r in rows
                                          if r["min_mask_fraction_inside_box"] is not None), default=None))))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
