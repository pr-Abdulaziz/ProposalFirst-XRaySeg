#!/usr/bin/env python
"""Evaluate one PIDray diagnostic run on easy / hard / hidden with the corrected category mapping.

Uses the run's resolved config and final checkpoint (epoch 10), the same test pipeline and inference
settings as the frozen campaign, and the documented easy-split amendment (one undecodable image
excluded; derived files re-hashed before use). Predictions are dumped per split, scored at full
precision with the frozen scorer plus the MMDetection label -> category mapping, checked against
MMDetection's own metric, and summarised with pooled AP over all three splits. Training accounting
(skipped updates, non-finite losses) and logged gradient-norm statistics are recorded.

    python scripts/campaign_pidray_bs1/evaluate_diag.py <slug> <seed>
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

import scripts.campaign_multiseed.score_predictions as SP  # noqa: E402
from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402
from scripts.campaign_pidray_bs1.build_eval_exclusion import RECORD as EXCL  # noqa: E402
from scripts.campaign_pidray_bs1.coco_resample_mapped import label_to_category, to_coco_mapped  # noqa: E402
from scripts.campaign_pidray_bs1.evaluation_checks import diagnostic_errors  # noqa: E402

DIAG = ROOT / "output/pidray_diag_gradiso_2026-09-26"
OFF = {"easy": 0, "hard": 1_000_000, "hidden": 2_000_000}


def amended_easy() -> dict:
    rec = json.loads(EXCL.read_text())
    mapping = {}
    for r in rec["derived_files"].values():
        src, dst = ROOT / r["source"], ROOT / r["path"]
        assert C.sha256(src) == r["source_sha256"] and C.sha256(dst) == r["sha256"], "amendment files changed"
        mapping[str(src)] = str(dst)
    return mapping


def grad_norm_stats(log_dir: Path) -> dict:
    """Logged grad_norm is the mean of the last 50 iterations (MMEngine window), before clipping."""
    logs = sorted(log_dir.glob("*/*.log"))
    vals = [float(v) for f in logs for v in re.findall(r"Epoch\(train\).*?grad_norm: ([0-9.]+)", f.read_text())]
    a = np.array(vals)
    return dict(logged_windows=len(a), window_iterations=50, max_norm=1.0,
                mean=float(a.mean()) if len(a) else None, median=float(np.median(a)) if len(a) else None,
                share_windows_mean_above_max_norm=float((a > 1.0).mean()) if len(a) else None,
                note="window means: a window mean above 1.0 implies clipping in that window; below 1.0 does not exclude it")


def main() -> int:
    slug, seed = sys.argv[1], int(sys.argv[2])
    cfg_path = DIAG / "protocol/resolved_configs" / slug / f"seed_{seed}.py"
    base = Config.fromfile(str(cfg_path), import_custom_modules=True)
    import_modules_from_strings(**base.custom_imports)
    ckpt = Path(base.work_dir) / f"epoch_{C.MAX_EPOCHS}.pth"
    if not ckpt.is_file():
        raise SystemExit(f"ABORT: {ckpt} missing")
    job = os.environ.get("SLURM_JOB_ID", "interactive")
    out = DIAG / "evaluations" / slug / f"seed_{seed}"
    prior_outputs = [out / f"eval_record_job{job}.json"]
    for split in C.TEST_SPLITS:
        prior_outputs.extend((out / f"metrics_fullprec_job{job}_{split}.json",
                              Path(base.work_dir) / "test" / f"predictions_job{job}_{split}.pkl"))
    existing = [str(path) for path in prior_outputs if path.exists()]
    if existing:
        raise SystemExit(f"ABORT: refusing to overwrite diagnostic outputs: {existing}; use a fresh evaluation job")
    out.mkdir(parents=True, exist_ok=True)
    easy = amended_easy()
    splits, pooled_gt, pooled_dt = {}, dict(images=[], annotations=[], categories=None), {"bbox": [], "segm": []}
    for split in C.TEST_SPLITS:
        cfg = copy.deepcopy(base)
        paths = dict(cfg.campaign_test_splits[split])
        for role in ("loader", "evaluator"):
            paths[role] = easy.get(paths[role], paths[role])
        ds = dict(cfg.test_dataloader.dataset); ds["ann_file"] = paths["loader"]; cfg.test_dataloader.dataset = ds
        for e in (cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]):
            if "ann_file" in e:
                e["ann_file"] = paths["evaluator"]
        cfg.load_from, cfg.resume, cfg.launcher = str(ckpt), False, "none"
        cfg.work_dir = str(Path(base.work_dir) / "test" / f"job{job}_{split}")
        pred = Path(base.work_dir) / "test" / f"predictions_job{job}_{split}.pkl"
        pred.parent.mkdir(parents=True, exist_ok=True)
        runner = Runner.from_cfg(cfg)
        from mmdet.evaluation import DumpDetResults
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=str(pred)))
        t0 = time.perf_counter()
        mm = runner.test()
        seconds = time.perf_counter() - t0
        del runner
        torch.cuda.empty_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(paths["evaluator"])
        mapping = label_to_category(gt, C.CLASSES)
        SP.to_coco = lambda samples, t, _m=mapping: to_coco_mapped(samples, t, _m)
        full = SP.score(pred, Path(paths["evaluator"]))
        check = {t: dict(mmdet=mm[k], official=full[t]["stats"]["AP"], zero_id_safe=full[t]["stats_zero_id_safe"]["AP"],
                         matches=bool(min(abs(round(full[t]["stats"]["AP"], 3) - mm[k]),
                                          abs(round(full[t]["stats_zero_id_safe"]["AP"], 3) - mm[k])) < 1.5e-3))
                 for t, k in (("segm", "coco/segm_mAP"), ("bbox", "coco/bbox_mAP"))}
        (out / f"metrics_fullprec_job{job}_{split}.json").write_text(json.dumps({**full, "check": check, "mapping": mapping},
                                                                               indent=1, default=float))
        splits[split] = dict(loader=paths["loader"], evaluator=paths["evaluator"], predictions=str(pred),
                             predictions_sha256=C.sha256(pred), images=full["coverage"]["test_images"],
                             coverage=full["coverage"],
                             images_match=full["coverage"]["images_match"], seconds=seconds, check=check,
                             mask_AP=100 * full["segm"]["stats"]["AP"], box_AP=100 * full["bbox"]["stats"]["AP"],
                             mask_AP75=100 * full["segm"]["stats"]["AP75"])
        print(json.dumps({split: {k: splits[split][k] for k in ("mask_AP", "box_AP", "images", "images_match")}}), flush=True)
        # pooled test set: offset image and annotation ids per split (annotation ids +1 avoid id 0)
        d = json.loads(Path(paths["evaluator"]).read_text()); o = OFF[split]
        pooled_gt["categories"] = d["categories"]
        pooled_gt["images"] += [dict(im, id=im["id"] + o) for im in d["images"]]
        pooled_gt["annotations"] += [dict(a, id=a["id"] + 1 + o, image_id=a["image_id"] + o) for a in d["annotations"]]
        samples = pickle.load(open(pred, "rb"))
        for t in ("bbox", "segm"):
            for x in to_coco_mapped(samples, t, mapping):
                x["image_id"] += o; pooled_dt[t].append(x)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = pooled_gt; coco.createIndex()
        pooled = {}
        for t in ("bbox", "segm"):
            ev = COCOeval(coco, coco.loadRes(copy.deepcopy(pooled_dt[t])), t); ev.evaluate(); ev.accumulate(); ev.summarize()
            pooled[t] = dict(AP=100 * float(ev.stats[0]), AP50=100 * float(ev.stats[1]), AP75=100 * float(ev.stats[2]))
    acc_file = Path(base.work_dir) / "update_accounting.jsonl"
    accounting = json.loads(acc_file.read_text().splitlines()[-1]) if acc_file.is_file() else None
    validation_errors = diagnostic_errors(splits, C.TEST_SPLITS)
    record = dict(slug=slug, seed=seed, job=job, config=str(cfg_path.relative_to(ROOT)), config_sha256=C.sha256(cfg_path),
                  checkpoint=str(ckpt), checkpoint_sha256=C.sha256(ckpt), splits=splits, pooled=pooled,
                  all_checks_match=not validation_errors, validation_errors=validation_errors,
                  training_accounting=accounting, grad_norm=grad_norm_stats(Path(base.work_dir)),
                  scoring="frozen score_predictions.score with the MMDetection label->category mapping (label_to_category)")
    C.write_once(out / f"eval_record_job{job}.json", record)
    print(json.dumps(dict(pooled=pooled, all_checks_match=record["all_checks_match"],
                          validation_errors=validation_errors), indent=1))
    return 0 if record["all_checks_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
