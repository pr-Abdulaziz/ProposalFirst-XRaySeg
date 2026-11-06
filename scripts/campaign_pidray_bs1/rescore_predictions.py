#!/usr/bin/env python
"""Re-score the saved PIDray-30 predictions with the correct label -> category-id mapping.

Bug (found 2026-09-25, after the first four evaluations): the frozen full-precision scorer
(scripts/campaign_multiseed/score_predictions.py, written for PIXray) writes the model's class index
directly as the COCO ``category_id``. PIXray's category ids are 0..14, so this is correct there. PIDray's
ids are 1..12, so every PIDray detection was scored as the neighbouring class and the full-precision
AP is near zero. Predictions, checkpoints and MMDetection's own metric (``mmengine_metrics_rounded``
in each eval record) are not affected.

Fix: map class index i to the i-th category id of the scored annotation file (ids sorted ascending,
names checked against the campaign class order), exactly as MMDetection's CocoMetric does, and
otherwise call the frozen scorer unchanged. Each result is checked against the eval record's
MMDetection metric (3-decimal rounding). Outputs are new write-once files; nothing frozen is edited.

    python scripts/campaign_pidray_bs1/rescore_predictions.py [--methods 03 ...] [--splits hard ...]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402

import scripts.campaign_multiseed.score_predictions as SP  # noqa: E402
from scripts.campaign_pidray_bs1 import campaign_common as C  # noqa: E402

FROZEN_TO_COCO = SP.to_coco
AMEND = C.OUT / "amendments"
RECORD = AMEND / "rescoring_category_mapping_2026-09-25.json"


def mapped_to_coco(mapping):
    def to_coco(samples, iou_type):
        dets = FROZEN_TO_COCO(samples, iou_type)
        for d in dets:
            d["category_id"] = mapping[d["category_id"]]
        return dets
    return to_coco


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--splits", nargs="*", default=list(C.TEST_SPLITS))
    ap.add_argument("--dry", action="store_true", help="score and compare, write nothing")
    a = ap.parse_args()
    rows, bad = [], []
    for rec_path in sorted(C.OUT.glob("evaluations/*/seed_*/eval_record_job*.json")):
        rec = json.loads(rec_path.read_text())
        if a.methods and rec["method"] not in a.methods:
            continue
        for split in a.splits:
            sp = rec["splits"][split]
            ann = ROOT / sp["annotations"]
            with contextlib.redirect_stdout(io.StringIO()):
                coco = COCO(str(ann))
            cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
            if tuple(c["name"] for c in cats) != tuple(C.CLASSES):
                raise SystemExit(f"{ann}: category order differs from the campaign classes")
            mapping = {i: c["id"] for i, c in enumerate(cats)}
            SP.to_coco = mapped_to_coco(mapping)
            t0 = time.time()
            res = SP.score(Path(sp["predictions"]), ann)
            SP.to_coco = FROZEN_TO_COCO
            mm = sp["mmengine_metrics_rounded"]
            check = {}
            for t, key in (("segm", "coco/segm_mAP"), ("bbox", "coco/bbox_mAP")):
                off, zid = res[t]["stats"]["AP"], res[t]["stats_zero_id_safe"]["AP"]
                ok = min(abs(round(off, 3) - mm[key]), abs(round(zid, 3) - mm[key])) < 1.5e-3
                check[t] = dict(mmdet=mm[key], official=off, zero_id_safe=zid, matches=bool(ok))
                if not ok:
                    bad.append(f"{rec['method']}/{split}/{t}")
            row = dict(method=rec["method"], seed=rec["seed"], job=rec["job"], split=split,
                       predictions=sp["predictions"], predictions_sha256=res["predictions_sha256"],
                       annotations=sp["annotations"], annotations_sha256=res["annotations_sha256"],
                       images=res["coverage"]["test_images"], images_match=res["coverage"]["images_match"],
                       segm=res["segm"]["stats"], bbox=res["bbox"]["stats"], check=check,
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(f"{rec['method']:>4} {split:<6} mask AP {100 * row['segm']['AP']:.2f} (mmdet {100 * mm['coco/segm_mAP']:.1f}) "
                  f"box AP {100 * row['bbox']['AP']:.2f} (mmdet {100 * mm['coco/bbox_mAP']:.1f}) "
                  f"match={check['segm']['matches'] and check['bbox']['matches']} {row['seconds']} s", flush=True)
            if not a.dry:
                out = rec_path.parent / f"metrics_fullprec_rescored_job{rec['job']}_{split}.json"
                if not out.exists():
                    C.write_once(out, {**res, "rescoring": dict(reason="label index -> category id mapping",
                                                                mapping=mapping, check=check,
                                                                record=str(RECORD.relative_to(ROOT)))})
    if bad:
        print("MISMATCH with MMDetection metric:", bad)
        return 1
    if not a.dry and rows:
        AMEND.mkdir(parents=True, exist_ok=True)
        prev = json.loads(RECORD.read_text()) if RECORD.exists() else None
        rec = dict(
            amendment="full-precision rescoring with the correct label -> category-id mapping",
            found="2026-09-25, after the first four PIDray evaluations",
            cause="score_predictions.to_coco writes the class index as category_id; correct for PIXray (ids 0-14), "
                  "wrong for PIDray (ids 1-12)",
            fix="class index i -> i-th sorted category id of the scored annotation file; frozen scorer otherwise unchanged",
            unaffected="predictions, checkpoints, MMDetection metrics in the eval records, all PIXray results",
            verification="every rescored AP equals the MMDetection metric of the same run and split (3-decimal rounding)",
            script="scripts/campaign_pidray_bs1/rescore_predictions.py",
            rows=(prev["rows"] if prev else []) + rows)
        RECORD.unlink(missing_ok=True) if prev else None
        C.write_once(RECORD, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
