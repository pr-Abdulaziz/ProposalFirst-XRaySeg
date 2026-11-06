#!/usr/bin/env python
"""Evaluate one PIDray-30 run's final checkpoint on all three released test splits.

Reads the frozen manifest only. Refuses to start before every approved training
run has completed, refuses anything but the final-epoch checkpoint of the run's
own work directory, verifies the resolved-config and evaluator-source hashes,
then for each of easy / hard / hidden runs inference with the frozen config,
dumps predictions and scores them at full precision. One write-once record.

PIDray's ``area`` field is the mask area, so the COCO small/medium/large bins
here are mask-area bins (on PIXray the same field is the box area).

    python scripts/campaign_pidray_bs1/evaluate_run.py <method id> [<seed>]
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
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_multiseed.score_predictions import EVALUATOR, score  # noqa: E402

EVALUATOR_SOURCES = [
    "scripts/campaign_pidray_bs1/evaluate_run.py",
    "scripts/campaign_multiseed/score_predictions.py",
    "projects/proposal_first_pidray_occ/evaluation/xray_coco_metric.py",
    "projects/proposal_first_pidray_occ/evaluation/zero_id_compat.py",
    "projects/proposal_first_pidray_occ/evaluation/proposal_first_metric.py",
]
PIDRAY_EVALUATOR = dict(EVALUATOR, area_field_semantics="PIDray 'area' = mask area")


def main() -> int:
    C.require_leakage_policy()
    mid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else C.SEEDS[0]
    key = C.run_key(mid, seed)
    manifest = json.loads(C.MANIFEST.read_text())
    run = manifest["runs"][key]
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    frozen = json.loads((ROOT / manifest["source_snapshot"]["per_file"]).read_text())["files"]
    for src in EVALUATOR_SOURCES:
        if src in frozen and C.sha256(ROOT / src) != frozen[src]:
            raise SystemExit(f"ABORT: evaluator source {src} changed after the freeze")
    cfg_path = ROOT / run["resolved_config"]
    if C.sha256(cfg_path) != run["resolved_config_sha256"]:
        raise SystemExit(f"ABORT: {cfg_path} changed since the manifest was frozen")
    incomplete = []
    for k, r in manifest["runs"].items():
        if not r["launchable"]:
            continue
        st = C.run_dir(r["method"], r["seed"]) / "status.jsonl"
        events = [json.loads(x)["event"] for x in st.read_text().splitlines()] if st.is_file() else []
        if "train_complete" not in events:
            incomplete.append(k)
    if incomplete:
        raise SystemExit(f"ABORT: {len(incomplete)} training runs are not complete "
                         f"(e.g. {incomplete[:3]}); evaluation waits until all of them have finished")
    ckpt = Path(run["work_dir"]) / f"epoch_{C.MAX_EPOCHS}.pth"
    if not ckpt.is_file():
        raise SystemExit(f"ABORT: {ckpt} missing; the checkpoint rule is the final epoch")

    ev = C.eval_dir(mid, seed)
    ev.mkdir(parents=True, exist_ok=True)
    C.append_jsonl(C.run_dir(mid, seed) / "status.jsonl",
                   dict(event="eval_started", job=job, time=time.time()))
    splits, ok = {}, True
    for split in C.TEST_SPLITS:
        cfg = Config.fromfile(str(cfg_path), import_custom_modules=True)
        import_modules_from_strings(**cfg.custom_imports)
        paths = cfg.campaign_test_splits[split]
        # The loader may read the occluder track (06, 07 need bg_object_segmentation);
        # scoring is always against the released standard annotations.
        ann = Path(paths["evaluator"])
        ds = copy.deepcopy(dict(cfg.test_dataloader.dataset))
        ds["ann_file"] = paths["loader"]
        cfg.test_dataloader.dataset = ds
        for e in (cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]):
            if "ann_file" in e:
                e["ann_file"] = str(ann)
        cfg.load_from = str(ckpt)
        cfg.resume = False
        cfg.work_dir = str(Path(run["work_dir"]) / "test" / f"job{job}_{split}")
        cfg.launcher = "none"
        pred = Path(run["work_dir"]) / "test" / f"predictions_job{job}_{split}.pkl"
        pred.parent.mkdir(parents=True, exist_ok=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
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
        full = score(pred, ann)
        out_path = ev / f"metrics_fullprec_job{job}_{split}.json"
        out_path.write_text(json.dumps(full, indent=1, default=float))
        ok &= full["coverage"]["images_match"]
        splits[split] = dict(
            loader_annotations=paths["loader"], loader_annotations_sha256=C.sha256(Path(paths["loader"])),
            annotations=str(ann.relative_to(ROOT)), annotations_sha256=C.sha256(ann),
            images=images, predictions=str(pred), predictions_sha256=full["predictions_sha256"],
            metrics_fullprec=str(out_path.relative_to(ROOT)),
            coverage=full["coverage"], mmengine_metrics_rounded=metrics,
            segm=full["segm"]["stats"], bbox=full["bbox"]["stats"],
            efficiency=dict(seconds=seconds, images=images, images_per_second=images / seconds,
                            peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20))
        print(json.dumps(dict(run=key, split=split, segm_AP=full["segm"]["stats"]["AP"],
                              bbox_AP=full["bbox"]["stats"]["AP"])), flush=True)
        del runner

    total = sum(v["images"] for v in splits.values())
    record = dict(
        run_key=key, method=mid, seed=seed, job=job, node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
        checkpoint=str(ckpt), checkpoint_sha256=C.sha256(ckpt),
        resolved_config=run["resolved_config"], resolved_config_sha256=run["resolved_config_sha256"],
        evaluator=PIDRAY_EVALUATOR,
        evaluator_sources={s: C.sha256(ROOT / s) for s in EVALUATOR_SOURCES},
        splits=splits,
        test_images_total=total,
        image_weighted_segm_AP=sum(v["segm"]["AP"] * v["images"] for v in splits.values()) / total,
        image_weighted_bbox_AP=sum(v["bbox"]["AP"] * v["images"] for v in splits.values()) / total,
        precision="fp32 (no autocast at test)")
    C.write_once(ev / f"eval_record_job{job}.json", record)
    C.append_jsonl(C.run_dir(mid, seed) / "status.jsonl",
                   dict(event="eval_complete" if ok else "eval_coverage_failed", job=job,
                        time=time.time(),
                        segm_AP={s: v["segm"]["AP"] for s, v in splits.items()}))
    print(json.dumps({s: dict(segm=v["segm"], bbox=v["bbox"]) for s, v in splits.items()}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
