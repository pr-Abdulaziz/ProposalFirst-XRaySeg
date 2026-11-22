#!/usr/bin/env python
"""Re-evaluate a refiner at a different inference point budget.

Same frozen config, same final-epoch checkpoint and same campaign scorer as evaluate_run.py.
The only change is model.roi_head.point_refine_head.infer_points. Writes only under
output/<campaign>/revision_2026-11/; the campaign's runs/, evaluations/ and work_dirs are untouched.

    python scripts/revision_2026_11/evaluate_infer_points.py <method id 09|09M> <seed> <infer_points>
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
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import MAX_EPOCHS, OUT, run_key, sha256  # noqa: E402
from scripts.campaign_multiseed.run_guard_amended import verified_manifest  # noqa: E402
from scripts.campaign_multiseed.score_predictions import EVALUATOR, score  # noqa: E402

REV = OUT / "revision_2026-11"


def main() -> int:
    mid, seed, points = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if mid not in ("09", "09M"):
        raise SystemExit("only the two refiners have an inference point budget")
    key = run_key(mid, seed)
    run = json.loads(verified_manifest().read_text())["runs"][key]
    cfg_path = ROOT / run["resolved_config"]
    if sha256(cfg_path) != run["resolved_config_sha256"]:
        raise SystemExit(f"ABORT: {cfg_path} changed since the manifest was frozen")
    ckpt = Path(run["work_dir"]) / f"epoch_{MAX_EPOCHS}.pth"
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    out = REV / f"a1_infer{points}" / key
    if (out / "record.json").exists():
        raise SystemExit(f"ABORT: {out}/record.json exists (write-once)")
    out.mkdir(parents=True, exist_ok=True)
    pred = out / f"predictions_job{job}.pkl"

    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    head = cfg.model.roi_head.point_refine_head
    original = head.infer_points
    head.infer_points = points
    cfg.load_from = str(ckpt)
    cfg.resume = False
    cfg.work_dir = str(out / f"job{job}")
    cfg.launcher = "none"
    runner = Runner.from_cfg(cfg)
    built = runner.model.roi_head.point_refine_head.infer_points
    if built != points:
        raise SystemExit(f"ABORT: built model uses infer_points={built}, expected {points}")
    from mmdet.evaluation import DumpDetResults
    runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(pred)))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    metrics = runner.test()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    full = score(pred)
    (out / "metrics_fullprec.json").write_text(json.dumps(full, indent=1, default=float))
    record = dict(
        purpose="inference point budget",
        run_key=key, method=mid, seed=seed, job=job, node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
        infer_points=points, infer_points_frozen_config=original, train_points=head.train_points,
        checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
        resolved_config=run["resolved_config"], resolved_config_sha256=run["resolved_config_sha256"],
        predictions=str(pred), predictions_sha256=full["predictions_sha256"], evaluator=EVALUATOR,
        coverage=full["coverage"], mmengine_metrics_rounded=metrics,
        seconds=seconds, images_per_second=len(runner.test_dataloader.dataset) / seconds)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(dict(run=key, infer_points=points, segm_AP=full["segm"]["stats"]["AP"],
                          bbox_AP=full["bbox"]["stats"]["AP"]), indent=1))
    return 0 if full["coverage"]["images_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
