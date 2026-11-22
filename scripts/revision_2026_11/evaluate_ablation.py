#!/usr/bin/env python
"""Inference-time ablations of the trained refiners.

The frozen config and final-epoch checkpoint are left unchanged. Each variant only alters how the
already-trained refiner is used at test time:
  off        refinement disabled: the plain cascade mask path (28x28 masks, stage-averaged)
  level56    one refinement level only (28 -> 56); the second 56 -> 112 level is skipped
  random784  784 points per level chosen uniformly at random (fixed generator) instead of the most uncertain
  pts392     392 most-uncertain points per level
  pts1568    1568 most-uncertain points per level
  laststage  only the final cascade stage is refined; stages 1-2 are bilinearly upsampled 28->56->112
Scored with the campaign scorer. Writes under output/<campaign>/revision_2026-11/ablation/<variant>/.

    python scripts/revision_2026_11/evaluate_ablation.py <09|09M> <seed> <variant> [<variant> ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import MAX_EPOCHS, OUT, run_key, sha256  # noqa: E402
from scripts.campaign_multiseed.run_guard_amended import verified_manifest  # noqa: E402
from scripts.campaign_multiseed.score_predictions import EVALUATOR, score  # noqa: E402

_ORIGINAL_SELECT = None
VARIANTS = ("off", "level56", "random784", "pts392", "pts1568", "laststage")


def _random_selector(seed: int = 1234):
    state = {}

    def select(logits, labels, num_points=784):
        n, _, h, w = logits.shape
        total = h * w
        g = state.get(logits.device)
        if g is None:
            g = state[logits.device] = torch.Generator(device=logits.device).manual_seed(seed)
        scores = torch.rand(n, total, device=logits.device, generator=g)
        return scores.topk(min(num_points, total), dim=1).indices
    return select


def apply_variant(model, variant: str) -> dict:
    """Patch the built model in place; returns a description of what changed."""
    import projects.proposal_first_pidray_occ.models.point_refine.head as head_mod
    rh = model.roi_head
    head = rh.point_refine_head
    if variant == "off":
        rh.refine_enabled = False
    elif variant == "level56":
        head.output_sizes = (56,)
    elif variant == "random784":
        head.infer_points = 784
        head_mod.select_inference_points = _random_selector()
    elif variant == "pts392":
        head.infer_points = 392
    elif variant == "pts1568":
        head.infer_points = 1568
    elif variant == "laststage":
        original = head.predict_refine
        last = rh.num_stages - 1
        sizes = head.output_sizes

        def predict_refine(coarse_logits, rois, labels, fine_features, stage=0):
            if stage == last:
                return original(coarse_logits, rois, labels, fine_features, stage=stage)
            out = coarse_logits
            for s in sizes:
                out = F.interpolate(out, size=(s, s), mode="bilinear", align_corners=False)
            return out
        head.predict_refine = predict_refine
    else:
        raise SystemExit(f"unknown variant {variant}")
    return dict(refine_enabled=rh.refine_enabled, output_sizes=list(head.output_sizes),
                infer_points=head.infer_points,
                selector=head_mod.select_inference_points.__name__,
                last_stage_only=variant == "laststage")


def run_one(mid: str, seed: int, variant: str) -> float:
    import projects.proposal_first_pidray_occ.models.point_refine.head as head_mod
    global _ORIGINAL_SELECT
    if _ORIGINAL_SELECT is None:
        _ORIGINAL_SELECT = head_mod.select_inference_points
    head_mod.select_inference_points = _ORIGINAL_SELECT   # undo any patch from a previous variant
    key = run_key(mid, seed)
    run = json.loads(verified_manifest().read_text())["runs"][key]
    cfg_path = ROOT / run["resolved_config"]
    if sha256(cfg_path) != run["resolved_config_sha256"]:
        raise SystemExit(f"ABORT: {cfg_path} changed since the manifest was frozen")
    ckpt = Path(run["work_dir"]) / f"epoch_{MAX_EPOCHS}.pth"
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    out = OUT / "revision_2026-11" / "ablation" / variant / key
    if (out / "record.json").exists():
        print(f"skip {out}: record exists")
        return float("nan")
    out.mkdir(parents=True, exist_ok=True)
    pred = out / f"predictions_job{job}.pkl"
    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.load_from, cfg.resume, cfg.launcher = str(ckpt), False, "none"
    cfg.work_dir = str(out / f"job{job}")
    runner = Runner.from_cfg(cfg)
    applied = apply_variant(runner.model, variant)
    from mmdet.evaluation import DumpDetResults
    runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(pred)))
    t0 = time.perf_counter()
    runner.test()
    seconds = time.perf_counter() - t0
    full = score(pred)
    (out / "metrics_fullprec.json").write_text(json.dumps(full, indent=1, default=float))
    rec = dict(purpose="inference-time ablation", variant=variant, applied=applied,
               run_key=key, method=mid, seed=seed, job=job, node=os.uname().nodename,
               checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
               resolved_config_sha256=run["resolved_config_sha256"], predictions=str(pred),
               predictions_sha256=full["predictions_sha256"], evaluator=EVALUATOR, coverage=full["coverage"],
               seconds=seconds, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    ap = full["segm"]["stats"]["AP"]
    print(json.dumps(dict(run=key, variant=variant, applied=applied, segm_AP=ap,
                          bbox_AP=full["bbox"]["stats"]["AP"])), flush=True)
    del runner
    torch.cuda.empty_cache()
    return ap


def main() -> int:
    mid, seed, variants = sys.argv[1], int(sys.argv[2]), sys.argv[3:] or list(VARIANTS)
    if mid not in ("09", "09M"):
        raise SystemExit("only the refiners")
    bad = 0
    for v in variants:
        try:
            run_one(mid, seed, v)
        except Exception as e:  # keep going; one failed variant must not hide the others
            bad += 1
            print(f"FAILED {mid} s{seed} {v}: {type(e).__name__}: {e}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
