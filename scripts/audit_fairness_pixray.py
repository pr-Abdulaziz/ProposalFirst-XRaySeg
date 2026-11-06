#!/usr/bin/env python
"""Hyperparameter and experimental-fairness audit: PIXray 03 / 09M / 09.

Evidence rules, applied throughout:

  configured  -- read from a *training-time resolved* config that the run itself
                 dumped into its work_dir, not from a current source file.
  observed    -- read from a run's own log, checkpoint or output artefact.
  unverified  -- not established by either; said so rather than assumed.

Method 09M has not started at audit time, so its rows are `configured` only.
That is a statement about evidence, not a claim that it will differ.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config  # noqa: E402

OUT = ROOT / "output/audit/2026-09-12-fairness"

# Training-time resolved configs, dumped by the runs themselves.
SOURCES = {
    "03":  ("work_dirs/pixray_p2/pixray_p2_cascade_attfpn.py", "training-time resolved (2026-09-03)"),
    "09":  ("work_dirs/pixray_09_point_refine/pixray_09_point_refine.py", "training-time resolved (2026-09-12, job 11048)"),
    "09M": ("output/full_split_campaign/2026-09-12/resolved_09M.py", "frozen resolved config; run queued, not yet started"),
}
EVAL_03 = "work_dirs/pixray_p2/test/test/pixray_p2_cascade_attfpn.py"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def name_of(value):
    """Class references and strings both reduce to a bare type name, so a
    pure-Python config and a dumped config compare on equal terms."""
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, str) and "." in value and value[0].islower():
        return value.rsplit(".", 1)[-1]
    return value


def pipeline_of(steps, kind):
    return [dict(t) for t in steps]


def extract(cfg: Config) -> dict:
    m = cfg.model
    rh = m["roi_head"]
    rpn = m["rpn_head"]
    train_cfg, test_cfg = m["train_cfg"], m["test_cfg"]
    td = cfg.train_dataloader
    ted = cfg.test_dataloader
    opt = cfg.optim_wrapper
    bbox_heads = rh["bbox_head"] if isinstance(rh["bbox_head"], list) else [rh["bbox_head"]]
    prh = rh.get("point_refine_head")

    out = {}
    A = out.__setitem__

    # -- data ---------------------------------------------------------------
    A("data.train_ann", Path(td["dataset"]["ann_file"]).name)
    A("data.train_prefix", td["dataset"]["data_prefix"]["img"])
    A("data.test_ann", Path(ted["dataset"]["ann_file"]).name)
    A("data.test_prefix", ted["dataset"]["data_prefix"]["img"])
    A("data.dataset_type", name_of(td["dataset"]["type"]))
    A("data.classes", tuple(td["dataset"]["metainfo"]["classes"]))
    A("data.num_classes_metainfo", len(td["dataset"]["metainfo"]["classes"]))

    # -- preprocessing ------------------------------------------------------
    dp = m["data_preprocessor"]
    A("pre.mean", tuple(dp["mean"]))
    A("pre.std", tuple(dp["std"]))
    A("pre.bgr_to_rgb", dp.get("bgr_to_rgb"))
    A("pre.pad_size_divisor", dp.get("pad_size_divisor"))
    A("pre.pad_mask", dp.get("pad_mask"))
    A("pre.mask_pad_value", dp.get("mask_pad_value"))
    for step in cfg.train_pipeline:
        t = name_of(step["type"])
        for k, v in step.items():
            if k == "type":
                continue
            A(f"aug.train.{t}.{k}", tuple(v) if isinstance(v, list) else v)
    A("aug.train.order", tuple(name_of(s["type"]) for s in cfg.train_pipeline))
    for step in cfg.test_pipeline:
        t = name_of(step["type"])
        for k, v in step.items():
            if k == "type":
                continue
            A(f"aug.test.{t}.{k}", tuple(v) if isinstance(v, list) else v)
    A("aug.test.order", tuple(name_of(s["type"]) for s in cfg.test_pipeline))

    # -- backbone / neck ----------------------------------------------------
    bb = m["backbone"]
    for k in ("type", "depth", "num_stages", "out_indices", "frozen_stages",
              "norm_eval", "style", "dcn", "stage_with_dcn"):
        if k in bb:
            A(f"backbone.{k}", name_of(bb[k]) if k == "type"
              else (tuple(bb[k]) if isinstance(bb[k], list) else bb[k]))
    A("backbone.norm_cfg", dict(bb["norm_cfg"]) if "norm_cfg" in bb else None)
    A("backbone.init_cfg", dict(bb["init_cfg"]) if bb.get("init_cfg") else None)
    nk = m["neck"]
    for k in ("type", "in_channels", "out_channels", "num_outs", "start_level",
              "add_extra_convs", "relu_before_extra_convs"):
        if k in nk:
            A(f"neck.{k}", name_of(nk[k]) if k == "type"
              else (tuple(nk[k]) if isinstance(nk[k], list) else nk[k]))
    for k in sorted(set(nk) - {"type", "in_channels", "out_channels", "num_outs",
                               "start_level", "add_extra_convs",
                               "relu_before_extra_convs"}):
        A(f"neck.{k}", name_of(nk[k]) if not isinstance(nk[k], (dict, list)) else str(nk[k]))

    # -- RPN ----------------------------------------------------------------
    ag = rpn["anchor_generator"]
    A("rpn.type", name_of(rpn["type"]))
    A("rpn.in_channels", rpn["in_channels"])
    A("rpn.feat_channels", rpn["feat_channels"])
    A("rpn.anchor.scales", tuple(ag["scales"]))
    A("rpn.anchor.ratios", tuple(ag["ratios"]))
    A("rpn.anchor.strides", tuple(ag["strides"]))
    A("rpn.loss_cls", f"{name_of(rpn['loss_cls']['type'])} w={rpn['loss_cls'].get('loss_weight', 1.0)}")
    A("rpn.loss_bbox", f"{name_of(rpn['loss_bbox']['type'])} w={rpn['loss_bbox'].get('loss_weight', 1.0)}")
    r = train_cfg["rpn"]
    A("rpn.train.assigner", f"pos>={r['assigner']['pos_iou_thr']} neg<{r['assigner']['neg_iou_thr']} min={r['assigner']['min_pos_iou']}")
    A("rpn.train.sampler", f"{name_of(r['sampler']['type'])} num={r['sampler']['num']} pos_frac={r['sampler']['pos_fraction']}")
    A("rpn.train.allowed_border", r.get("allowed_border"))
    for phase, key in (("train", "rpn_proposal"), ("test", "rpn")):
        p = (train_cfg if phase == "train" else test_cfg)[key]
        A(f"rpn.{phase}.nms_pre", p["nms_pre"])
        A(f"rpn.{phase}.max_per_img", p["max_per_img"])
        A(f"rpn.{phase}.nms_iou", p["nms"]["iou_threshold"])
        A(f"rpn.{phase}.min_bbox_size", p.get("min_bbox_size"))

    # -- cascade ------------------------------------------------------------
    A("roi.type", name_of(rh["type"]))
    A("roi.num_stages", rh["num_stages"])
    A("roi.stage_loss_weights", tuple(rh["stage_loss_weights"]))
    be = rh["bbox_roi_extractor"]
    A("roi.bbox_extractor", f"{name_of(be['roi_layer']['type'])} out={be['roi_layer']['output_size']} "
                            f"sr={be['roi_layer']['sampling_ratio']} ch={be['out_channels']} strides={tuple(be['featmap_strides'])}")
    me = rh["mask_roi_extractor"]
    A("roi.mask_extractor", f"{name_of(me['roi_layer']['type'])} out={me['roi_layer']['output_size']} "
                            f"sr={me['roi_layer']['sampling_ratio']} ch={me['out_channels']} strides={tuple(me['featmap_strides'])}")
    for i, bh in enumerate(bbox_heads):
        A(f"roi.bbox_head{i}.type", name_of(bh["type"]))
        A(f"roi.bbox_head{i}.num_classes", bh["num_classes"])
        A(f"roi.bbox_head{i}.fc_out_channels", bh.get("fc_out_channels"))
        A(f"roi.bbox_head{i}.reg_class_agnostic", bh.get("reg_class_agnostic"))
        A(f"roi.bbox_head{i}.target_stds", tuple(bh["bbox_coder"]["target_stds"]))
        A(f"roi.bbox_head{i}.loss_cls", f"{name_of(bh['loss_cls']['type'])} w={bh['loss_cls'].get('loss_weight',1.0)}")
        A(f"roi.bbox_head{i}.loss_bbox", f"{name_of(bh['loss_bbox']['type'])} w={bh['loss_bbox'].get('loss_weight',1.0)}")
        st = train_cfg["rcnn"][i] if isinstance(train_cfg["rcnn"], list) else train_cfg["rcnn"]
        A(f"roi.stage{i}.assigner", f"pos>={st['assigner']['pos_iou_thr']} neg<{st['assigner']['neg_iou_thr']} min={st['assigner']['min_pos_iou']}")
        A(f"roi.stage{i}.sampler", f"{name_of(st['sampler']['type'])} num={st['sampler']['num']} pos_frac={st['sampler']['pos_fraction']}")
        A(f"roi.stage{i}.mask_size", st.get("mask_size"))
        A(f"roi.stage{i}.pos_weight", st.get("pos_weight"))
    mh = rh["mask_head"]
    mh0 = mh[0] if isinstance(mh, list) else mh
    A("roi.mask_head.type", name_of(mh0["type"]))
    A("roi.mask_head.num_convs", mh0.get("num_convs"))
    A("roi.mask_head.in_channels", mh0.get("in_channels"))
    A("roi.mask_head.conv_out_channels", mh0.get("conv_out_channels"))
    A("roi.mask_head.num_classes", mh0.get("num_classes"))
    A("roi.mask_head.class_agnostic", mh0.get("class_agnostic"))
    A("roi.mask_head.loss_mask", f"{name_of(mh0['loss_mask']['type'])} "
                                 f"use_mask={mh0['loss_mask'].get('use_mask')} "
                                 f"w={mh0['loss_mask'].get('loss_weight',1.0)}")
    A("roi.mask_head.shared", "one head" if not isinstance(mh, list) else f"{len(mh)} heads")

    # -- point refinement ---------------------------------------------------
    A("refine.enabled", rh.get("refine_enabled", False))
    A("refine.head_type", name_of(prh["type"]) if prh else None)
    A("refine.fine_level", rh.get("fine_level"))
    A("refine.generator_seed", rh.get("seed"))
    for k in ("num_classes", "output_sizes", "train_points", "infer_points",
              "oversample_ratio", "uncertain_fraction", "feature_stride",
              "embed_dim", "num_heads", "num_blocks", "ffn_dim",
              "use_stage_embedding"):
        v = prh.get(k) if prh else None
        A(f"refine.{k}", tuple(v) if isinstance(v, list) else v)

    # -- inference ----------------------------------------------------------
    rcnn = test_cfg["rcnn"]
    A("infer.score_thr", rcnn["score_thr"])
    A("infer.nms_iou", rcnn["nms"]["iou_threshold"])
    A("infer.nms_type", name_of(rcnn["nms"].get("type", "nms")))
    A("infer.max_per_img", rcnn["max_per_img"])
    A("infer.mask_thr_binary", rcnn["mask_thr_binary"])

    # -- optimisation -------------------------------------------------------
    o = opt["optimizer"]
    A("optim.wrapper", name_of(opt["type"]))
    A("optim.amp_dtype", opt.get("dtype"))
    A("optim.type", name_of(o["type"]))
    A("optim.lr", o["lr"])
    A("optim.weight_decay", o.get("weight_decay"))
    A("optim.betas", tuple(o["betas"]) if "betas" in o else "torch default (0.9, 0.999)")
    A("optim.eps", o.get("eps", "torch default 1e-8"))
    A("optim.clip_grad", f"max_norm={opt['clip_grad']['max_norm']} norm_type={opt['clip_grad']['norm_type']}")
    A("optim.accumulative_counts", opt["accumulative_counts"])
    A("optim.paramwise_cfg", opt.get("paramwise_cfg", "none -- single parameter group"))
    A("batch.physical", td["batch_size"])
    A("batch.effective", td["batch_size"] * opt["accumulative_counts"])
    A("sched.max_epochs", cfg.train_cfg["max_epochs"])
    A("sched.train_loop", name_of(cfg.train_cfg["type"]))
    for i, s in enumerate(cfg.param_scheduler):
        A(f"sched.{i}", f"{name_of(s['type'])} by_epoch={s.get('by_epoch')} "
                        f"begin={s.get('begin')} end={s.get('end')} "
                        f"{'start_factor='+str(s['start_factor']) if 'start_factor' in s else 'eta_min='+format(s.get('eta_min'), '.3g')}")
    A("sched.auto_scale_lr", dict(cfg.auto_scale_lr) if cfg.get("auto_scale_lr") else None)

    # -- runtime ------------------------------------------------------------
    A("run.seed", cfg.randomness["seed"])
    A("run.deterministic", cfg.randomness["deterministic"])
    A("run.diff_rank_seed", cfg.randomness.get("diff_rank_seed"))
    A("run.cudnn_benchmark", cfg.env_cfg.get("cudnn_benchmark"))
    A("run.mp_start_method", cfg.env_cfg["mp_cfg"].get("mp_start_method"))
    A("run.train_num_workers", td["num_workers"])
    A("run.train_persistent_workers", td["persistent_workers"])
    A("run.train_sampler", f"{name_of(td['sampler']['type'])} shuffle={td['sampler']['shuffle']}")
    A("run.train_batch_sampler", td.get("batch_sampler"))
    A("run.test_num_workers", ted["num_workers"])
    A("run.test_batch_size", ted["batch_size"])
    A("run.test_drop_last", ted.get("drop_last"))
    A("run.load_from", cfg.load_from)
    A("run.resume", cfg.resume)
    A("ckpt.hook", {k: v for k, v in cfg.default_hooks["checkpoint"].items() if k != "type"})
    A("ckpt.save_best", cfg.default_hooks["checkpoint"].get("save_best", "absent"))
    A("val.cfg", cfg.get("val_cfg"))
    A("val.dataloader", "None" if cfg.get("val_dataloader") is None else "PRESENT")
    ev = cfg.test_evaluator
    ev = ev if isinstance(ev, list) else [ev]
    for i, e in enumerate(ev):
        A(f"eval.{i}.type", name_of(e["type"]))
        A(f"eval.{i}.metric", tuple(e["metric"]) if "metric" in e else None)
        A(f"eval.{i}.classwise", e.get("classwise"))
        A(f"eval.{i}.zero_id_compat", e.get("zero_id_compat"))
        A(f"eval.{i}.ann_file", Path(e["ann_file"]).name if "ann_file" in e else None)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tables, meta = {}, {}
    for arm, (path, note) in SOURCES.items():
        full = ROOT / path
        tables[arm] = extract(Config.fromfile(str(full)))
        meta[arm] = dict(source=path, source_sha256=sha256(full), evidence=note)

    keys = []
    for arm in tables:
        for k in tables[arm]:
            if k not in keys:
                keys.append(k)

    rows = []
    for k in keys:
        vals = {a: tables[a].get(k, "<absent>") for a in tables}
        distinct = {json.dumps(v, default=str, sort_keys=True) for v in vals.values()}
        rows.append(dict(key=k, values={a: str(v) for a, v in vals.items()},
                         identical=len(distinct) == 1))

    payload = dict(generated="2026-09-12", sources=meta, rows=rows)
    (OUT / "hyperparameter_table.json").write_text(json.dumps(payload, indent=2, default=str))

    differing = [r for r in rows if not r["identical"]]
    print(f"{len(rows)} settings compared; {len(differing)} differ\n")
    for r in differing:
        print(f"  {r['key']}")
        for a, v in r["values"].items():
            print(f"      {a:<4} {v[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
