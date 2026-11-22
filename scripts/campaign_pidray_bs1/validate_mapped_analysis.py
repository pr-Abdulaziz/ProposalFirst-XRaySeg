#!/usr/bin/env python
"""Validate the category-safe analysis scripts before any PIDray statistics are generated.

1. PIXray regression (identity mapping): the mapped analysis must reproduce the stored PIXray analysis
   record and per-object table exactly, and the mapped AP cache must reproduce the stored COCO AP.
2. PIDray agreement: the mapped AP cache must reproduce the corrected scorer's AP (rescored files)
   and the mapped error analysis's TP count at score >= 0.5 must equal the corrected scorer's box TP
   count at IoU 0.5. The original logic is run on the same data for contrast.

    python scripts/campaign_pidray_bs1/validate_mapped_analysis.py --output <new-report.json>

The default output path is retained for compatibility, but an existing report is never overwritten.
Validation failures are recorded in the new report and produce a nonzero exit status.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.campaign_multiseed.analyze_errors_quality as ORIG  # noqa: E402
from scripts.campaign_pidray_bs1 import analyze_errors_quality_mapped as MAP  # noqa: E402
from scripts.campaign_pidray_bs1.coco_resample_mapped import APCacheMapped, self_check  # noqa: E402
from scripts.campaign_pidray_bs1.campaign_common import CLASSES as PIDRAY_CLASSES  # noqa: E402
from scripts.campaign_pidray_bs1.evaluation_checks import (  # noqa: E402
    AP_TOLERANCE, mapped_analysis_errors, write_new_report)
from projects.proposal_first_pidray_occ.datasets.pidray_dataset import PIXRAY_CLASSES  # noqa: E402

PIX = ROOT / "output/pixray_multiseed_bs4_e50_2026-09-14"
PID = ROOT / "output/pidray_bs2_e10_2026-09-22"
OUT = ROOT / "output/pidray_diag_gradiso_2026-09-26/verification/mapped_analysis_validation.json"


def load_coco(p):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(p))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New JSON report path; must not already exist")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"report already exists: {args.output}; choose a new --output path")
    report = {}
    # ---------------- 1. PIXray regression
    rec = json.loads(next((PIX / "evaluations/m09m_point_refine_mlp/seed_42").glob("eval_record_*.json")).read_text())
    ann = ROOT / "data/PIXray/annotations/test.json"
    new, per_gt = MAP.analyze(Path(rec["predictions"]), ann, PIXRAY_CLASSES)
    old = json.loads((PIX / "analysis/runs/m09m_point_refine_mlp/seed_42.json").read_text())
    old_gt = np.load(PIX / "analysis/runs/m09m_point_refine_mlp/seed_42_per_gt.npy")
    new_gt = np.array([[k, *v] for k, v in sorted(per_gt.items())], dtype=float)
    full = json.loads(next((PIX / "evaluations/m09m_point_refine_mlp/seed_42").glob("metrics_fullprec_*.json")).read_text())
    c = load_coco(ann)
    ap = {t: APCacheMapped(c, rec["predictions"], t, PIXRAY_CLASSES) for t in ("segm", "bbox")}
    report["pixray_regression"] = dict(
        mapping_is_identity=all(k == v for k, v in new["label_to_category"].items()),
        errors_identical=new["errors"] == old["errors"],
        mask_quality_identical=new["mask_quality"] == old["mask_quality"],
        pr_curves_identical=new["pr_curves"] == old["pr_curves"],
        per_gt_identical=bool(new_gt.shape == old_gt.shape and np.array_equal(new_gt, old_gt)),
        ap_cache_matches_scorer={t: abs(ap[t].official["AP"] - full[t]["stats"]["AP"]) for t in ap},
        ap_cache_self_check={t: self_check(ap[t]) for t in ap})
    print("PIXray:", report["pixray_regression"], flush=True)

    # ---------------- 2. PIDray agreement with the corrected scorer
    rows = json.loads((PID / "amendments/rescoring_category_mapping_2026-09-25.json").read_text())["rows"]
    report["pidray"] = []
    for method, split in (("03", "hard"), ("09M", "hard"), ("09M", "hidden")):
        r = next(x for x in rows if x["method"] == method and x["split"] == split)
        annp = ROOT / r["annotations"]
        rescored = json.loads(next(PID.glob(f"evaluations/*/seed_42/metrics_fullprec_rescored_job{r['job']}_{split}.json")).read_text())
        c = load_coco(annp)
        ap = {t: APCacheMapped(c, r["predictions"], t, PIDRAY_CLASSES) for t in ("segm", "bbox")}
        new, _ = MAP.analyze(Path(r["predictions"]), annp, PIDRAY_CLASSES)
        # original logic on the same data: raw category ids against class indices
        cats, gts = MAP.gt_tables(c, ap["segm"].mapping)
        raw = {k: dict(v, cls=np.array([cats[i] for i in v["cls"]], int)) for k, v in gts.items()}
        preds = {s["img_id"]: s for s in pickle.load(open(r["predictions"], "rb"))}
        try:
            orig_tp = ORIG.errors_and_confusion(preds, cats, raw)["counts"]["TP"]
        except Exception as e:  # noqa: BLE001
            orig_tp = f"fails: {type(e).__name__}: {e}"
        tp_scorer = rescored["bbox"]["prf_iou_0.50"]["at_0.50"]["TP"]
        row = dict(method=method, split=split,
                   ap_cache_vs_rescored={t: abs(ap[t].official["AP"] - rescored[t]["stats"]["AP"]) for t in ap},
                   ap_cache_self_check={t: self_check(ap[t]) for t in ap},
                   mask_AP=100 * ap["segm"].official["AP"], box_AP=100 * ap["bbox"].official["AP"],
                   tp_mapped_analysis=new["errors"]["counts"]["TP"], tp_corrected_scorer=tp_scorer,
                   tp_original_logic=orig_tp, matched_gt_mask_quality=new["mask_quality"]["matched_gt"],
                   num_gt=new["errors"]["num_gt"], coverage=new["mask_quality"]["coverage"],
                   mean_mask_iou=new["mask_quality"]["mean_iou"])
        report["pidray"].append(row)
        print("PIDray:", row, flush=True)
    errors = mapped_analysis_errors(report)
    report["validation"] = dict(passed=not errors, ap_absolute_tolerance=AP_TOLERANCE, errors=errors)
    write_new_report(args.output, report)
    print(json.dumps(report["validation"], indent=1), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
