#!/usr/bin/env python
"""Evaluate one campaign run's final checkpoint on the official PIXray test split.

Reads the frozen campaign manifest only. Refuses to start before every approved training run has completed, and refuses anything but the final-epoch checkpoint of
the run's own work directory, verifies the resolved config hash, runs inference
with the frozen config, dumps predictions, scores them with the single campaign
scorer at full precision and writes a write-once evaluation record.

    python scripts/campaign_multiseed/evaluate_run.py <method id> <seed>
"""
from __future__ import annotations

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
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_multiseed.campaign_common import (  # noqa: E402
    MANIFEST, MAX_EPOCHS, append_jsonl, eval_dir, run_dir, run_key, sha256, write_once)
from scripts.campaign_multiseed.score_predictions import EVALUATOR, score  # noqa: E402

EVALUATOR_SOURCES = [
    "scripts/campaign_multiseed/evaluate_run.py",
    "scripts/campaign_multiseed/score_predictions.py",
    "projects/proposal_first_pidray_occ/evaluation/xray_coco_metric.py",
    "projects/proposal_first_pidray_occ/evaluation/zero_id_compat.py",
    "projects/proposal_first_pidray_occ/evaluation/proposal_first_metric.py",
]


def main() -> int:
    mid, seed = sys.argv[1], int(sys.argv[2])
    key = run_key(mid, seed)
    manifest = json.loads(MANIFEST.read_text())
    run = manifest["runs"][key]
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    frozen = json.loads((ROOT / manifest["source_snapshot"]["per_file"]).read_text())["files"]
    for src in EVALUATOR_SOURCES:
        if src in frozen and sha256(ROOT / src) != frozen[src]:
            raise SystemExit(f"ABORT: evaluator source {src} changed after the freeze")
    cfg_path = ROOT / run["resolved_config"]
    if sha256(cfg_path) != run["resolved_config_sha256"]:
        raise SystemExit(f"ABORT: {cfg_path} changed since the manifest was frozen")
    # No evaluation until every approved training run has completed (user instruction, 2026-09-14).
    incomplete = []
    for k, r in manifest["runs"].items():
        if not r["launchable"]:
            continue
        st = run_dir(r["method"], r["seed"]) / "status.jsonl"
        events = [json.loads(x)["event"] for x in st.read_text().splitlines()] if st.is_file() else []
        if "train_complete" not in events:
            incomplete.append(k)
    if incomplete:
        raise SystemExit(f"ABORT: {len(incomplete)} training runs are not complete (e.g. {incomplete[:3]}); "
                         "evaluation waits until all of them have finished")
    ckpt = Path(run["work_dir"]) / f"epoch_{MAX_EPOCHS}.pth"
    if not ckpt.is_file():
        raise SystemExit(f"ABORT: {ckpt} missing; the checkpoint rule is the final epoch")
    ev = eval_dir(mid, seed)
    ev.mkdir(parents=True, exist_ok=True)
    pred = Path(run["work_dir"]) / "test" / f"predictions_job{job}.pkl"
    pred.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(run_dir(mid, seed) / "status.jsonl", dict(event="eval_started", job=job, time=time.time()))

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gpu_free_before, gpu_total = torch.cuda.mem_get_info()
    cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.load_from = str(ckpt)
    cfg.resume = False
    cfg.work_dir = str(Path(run["work_dir"]) / "test" / f"job{job}")
    cfg.launcher = "none"
    runner = Runner.from_cfg(cfg)
    from mmdet.evaluation import DumpDetResults
    runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(pred)))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    metrics = runner.test()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    images = len(runner.test_dataloader.dataset)

    full = score(pred)
    full_path = ev / f"metrics_fullprec_job{job}.json"
    full_path.write_text(json.dumps(full, indent=1, default=float))
    record = dict(
        run_key=key, method=mid, seed=seed, job=job, node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
        checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
        resolved_config=run["resolved_config"], resolved_config_sha256=run["resolved_config_sha256"],
        predictions=str(pred), predictions_sha256=full["predictions_sha256"],
        metrics_fullprec=str(full_path.relative_to(ROOT)), metrics_fullprec_sha256=sha256(full_path),
        evaluator=EVALUATOR, evaluator_sources={s: sha256(ROOT / s) for s in EVALUATOR_SOURCES},
        inference_settings=dict(model_test_cfg=cfg.model.get("test_cfg"), test_batch_size=cfg.test_dataloader.batch_size,
                                test_annotations_loader=cfg.test_dataloader.dataset.ann_file,
                                test_evaluator=cfg.test_evaluator, precision="fp32 (no autocast at test)"),
        coverage=full["coverage"],
        gpu_memory_before=dict(free_mib=gpu_free_before / 2**20, total_mib=gpu_total / 2**20,
                               pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        mmengine_metrics_rounded=metrics,
        efficiency=dict(evaluation_seconds=seconds, images=images, images_per_second=images / seconds,
                        peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                        note="end-to-end including data loading, COCO evaluation and dumping"))
    write_once(ev / f"eval_record_job{job}.json", record)
    ok = full["coverage"]["images_match"]
    append_jsonl(run_dir(mid, seed) / "status.jsonl",
                 dict(event="eval_complete" if ok else "eval_coverage_failed", job=job, time=time.time(),
                      segm_AP=full["segm"]["stats"]["AP"], bbox_AP=full["bbox"]["stats"]["AP"]))
    print(json.dumps(dict(run=key, segm=full["segm"]["stats"], bbox=full["bbox"]["stats"]), indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
