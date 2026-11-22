"""Measure R0 gate behaviour on the PIXray test set, and what it costs.

Two things the port document listed as unmeasured:
  1. gate acceptance and rejection counts, from the r0_score each prediction
     carries;
  2. what score-zeroing actually costs on the complete test predictions, rather
     than on a two-detection fixture.

Both are recoverable from the final checkpoint's predictions. Read-only.
"""
from __future__ import annotations
import contextlib, copy, io, json, pickle, sys
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_zero_id_effect import to_coco  # noqa: E402

PRED = ROOT / "work_dirs/pixray_08_pidray_r0r1/test/test/predictions.pkl"
ANN = ROOT / "data/PIXray/annotations/test.json"
THRESHOLD = 0.5


def score(coco, dets, iou_type):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(copy.deepcopy(dets)), iou_type)
        ev.params.maxDets = [100, 300, 1000]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main() -> int:
    preds = pickle.load(open(PRED, "rb"))
    r0 = np.array([s.get("r0_score", float("nan")) for s in preds], dtype=float)
    have = ~np.isnan(r0)
    print("=== R0 gate on the PIXray test set ===")
    print(f"  images                      : {len(preds)}")
    print(f"  carrying an r0_score        : {int(have.sum())}")
    if not have.any():
        print("  no r0_score recorded; gate behaviour cannot be measured here")
        return 1
    r0 = r0[have]
    accepted = int((r0 > THRESHOLD).sum()); rejected = int((r0 <= THRESHOLD).sum())
    print(f"  accepted (score > {THRESHOLD})       : {accepted}  ({accepted/len(r0):.2%})")
    print(f"  rejected (score <= {THRESHOLD})      : {rejected}  ({rejected/len(r0):.2%})")
    print(f"  r0_score  min {r0.min():.6f}  median {np.median(r0):.6f}  max {r0.max():.6f}")
    print(f"  every released test image contains target items, so each of the")
    print(f"  {rejected} rejections is a target-image rejection (a false negative")
    print(f"  for the gate's stated purpose).")

    # how many detections actually carry a zeroed score?
    zeroed = sum(int((np.asarray(s["pred_instances"]["scores"]) == 0).sum()) for s in preds)
    total = sum(len(s["pred_instances"]["scores"]) for s in preds)
    print(f"\n  detections with score exactly 0 : {zeroed} of {total}")

    print("\n=== what score-zeroing costs on the complete test predictions ===")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ANN))
    out = {"threshold": THRESHOLD, "images": len(preds), "accepted": accepted,
           "rejected": rejected, "zeroed_detections": zeroed, "total_detections": total,
           "r0_min": float(r0.min()), "r0_median": float(np.median(r0)),
           "r0_max": float(r0.max()), "results": {}}
    for iou_type in ("bbox", "segm"):
        gated = to_coco(preds, iou_type)
        # counterfactual: restore each zeroed detection to its pre-gate rank by
        # giving it the highest score among that image's surviving detections.
        restored = copy.deepcopy(gated)
        by_img = {}
        for d in restored:
            by_img.setdefault(d["image_id"], []).append(d)
        n_restored = 0
        for dets in by_img.values():
            nonzero = [d["score"] for d in dets if d["score"] > 0]
            fill = max(nonzero) if nonzero else 1.0
            for d in dets:
                if d["score"] == 0.0:
                    d["score"] = fill; n_restored += 1
        a = score(coco, gated, iou_type)
        b = score(coco, restored, iou_type)
        out["results"][iou_type] = {"as_evaluated": a, "counterfactual_ungated": b,
                                    "cost_of_gating": a - b, "restored": n_restored}
        print(f"  {iou_type}: as evaluated {a*100:.5f}   counterfactual {b*100:.5f}   "
              f"cost {(a-b)*100:+.5f} pp   ({n_restored} detections restored)")
    p = ROOT / "output/audit/2026-09-11/method08_gate_measurement.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
