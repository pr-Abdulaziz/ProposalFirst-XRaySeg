#!/usr/bin/env python
"""CPU verification of the gradient-isolated refiners (no training, nothing written back).

For each refiner (transformer, MLP), the original and the detached model load the same PIDray
checkpoint and see the same batch (one non-square image with objects, one empty image) and RNG state:
  1. forward equivalence: every loss value identical;
  2. point loss alone (detached): gradients only in the point refiner; exactly zero or absent in the
     backbone, neck, RPN, box heads and coarse mask heads;
  3. detector losses alone (detached): gradients in backbone, neck, RPN, box heads and mask heads;
  4. contrast: point loss alone in the original model reaches the backbone, neck and mask heads;
  5. inference equivalence on a test image: identical boxes, scores and masks.

    python scripts/campaign_pidray_bs1/check_gradiso.py
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

SRC = ROOT / "output/pidray_bs2_e10_2026-09-22"
DIAG = ROOT / "output/pidray_diag_gradiso_2026-09-26"
WORK = ROOT / "work_dirs/pidray_bs2_e10_2026-09-22"
PAIRS = {"transformer": ("m09_point_refine_transformer", "m09d_point_refine_transformer_detached"),
         "mlp": ("m09m_point_refine_mlp", "m09md_point_refine_mlp_detached")}
GROUPS = ("backbone", "neck", "rpn_head", "roi_head.bbox_head", "roi_head.mask_head", "roi_head.point_refine_head")
torch.set_num_threads(4)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def build(cfg_path, ckpt):
    cfg = Config.fromfile(str(cfg_path))
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope("mmdet")
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=True)
    return cfg, model


def val(v):
    """MMEngine sums list-valued losses (e.g. per-level RPN losses); do the same."""
    return sum(v) if isinstance(v, (list, tuple)) else v


def grad_summary(model):
    out = {}
    for g in GROUPS:
        vals = [p.grad for n, p in model.named_parameters() if n.startswith(g)]
        nz = [v for v in vals if v is not None and bool((v != 0).any())]
        out[g] = dict(params=len(vals), with_nonzero_grad=len(nz),
                      max_abs=float(max((v.abs().max() for v in nz), default=torch.tensor(0.0))))
    return out


def main() -> int:
    from mmdet.registry import DATASETS
    report = {}
    batch = test_sample = None
    for arm, (orig_slug, det_slug) in PAIRS.items():
        ckpt = WORK / orig_slug / "seed_42/epoch_10.pth"
        cfg_o, m_o = build(SRC / f"protocol/resolved_configs/{orig_slug}/seed_42.py", ckpt)
        cfg_d, m_d = build(DIAG / f"protocol/resolved_configs/{det_slug}/seed_42.py", ckpt)
        R = report[arm] = dict(head_types=[type(m_o.roi_head.point_refine_head).__name__,
                                           type(m_d.roi_head.point_refine_head).__name__])
        if batch is None:
            ds = DATASETS.build(cfg_o.train_dataloader.dataset); ds.full_init()
            info = [ds.get_data_info(i) for i in range(len(ds))]
            order = list(range(len(ds))); random.Random(2026).shuffle(order)
            nonsq = next(i for i in order if len({x["bbox_label"] for x in info[i]["instances"]}) >= 2
                         and abs(info[i]["width"] - info[i]["height"]) > 100)
            empty = next(i for i in order if len(info[i]["instances"]) == 0)
            seed_all(7)
            a = ds[nonsq]
            batch = (a, ds[empty])
            tds = DATASETS.build(cfg_o.test_dataloader.dataset); tds.full_init()
            tinfo = [tds.get_data_info(i) for i in range(len(tds))]
            ti = next(i for i in range(len(tds)) if len(tinfo[i]["instances"]) >= 1
                      and abs(tinfo[i]["width"] - tinfo[i]["height"]) > 50)
            test_sample = tds[ti]

        def run_loss(model):
            model.train(); model.zero_grad(set_to_none=True)
            data = model.data_preprocessor(pseudo_collate([copy.deepcopy(s) for s in batch]), True)
            seed_all(11)
            return model.loss(data["inputs"], data["data_samples"])

        lo, ld = run_loss(m_o), run_loss(m_d)
        R["forward_equivalence_max_abs_diff"] = max(float((val(lo[k]) - val(ld[k])).abs().max()) for k in lo)
        R["loss_values"] = {k: round(float(val(v)), 6) for k, v in ld.items() if "loss" in k}

        point = [k for k in ld if k.endswith("loss_point")]
        detector = [k for k in ld if "loss" in k and k not in point]
        sum(val(ld[k]) for k in point).backward()
        R["detached_point_loss_only"] = grad_summary(m_d)
        ld = run_loss(m_d)
        sum(val(ld[k]) for k in detector).backward()
        R["detached_detector_losses_only"] = grad_summary(m_d)
        lo = run_loss(m_o)
        sum(val(lo[k]) for k in point).backward()
        R["original_point_loss_only"] = grad_summary(m_o)
        m_o.zero_grad(set_to_none=True); m_d.zero_grad(set_to_none=True)

        preds = []
        for model in (m_o, m_d):
            model.eval()
            data = model.data_preprocessor(pseudo_collate([copy.deepcopy(test_sample)]), False)
            with torch.no_grad():
                preds.append(model.predict(data["inputs"], data["data_samples"], rescale=True)[0].pred_instances)
        po, pd = preds
        R["inference_equivalence"] = dict(
            detections=[len(po), len(pd)],
            boxes_equal=bool(torch.equal(po.bboxes, pd.bboxes)), scores_equal=bool(torch.equal(po.scores, pd.scores)),
            labels_equal=bool(torch.equal(po.labels, pd.labels)), masks_equal=bool(torch.equal(po.masks, pd.masks)))
        print(arm, json.dumps(R, indent=1), flush=True)
    out = DIAG / "verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_gradiso.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
