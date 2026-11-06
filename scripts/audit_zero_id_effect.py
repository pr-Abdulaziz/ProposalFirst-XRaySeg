"""Official vs zero-id-compatibility AP at full evaluator precision.

``CocoMetric`` rounds every value to three decimals, which puts a 0.1pp floor
on any delta computed from its output. This reads ``COCOeval.stats`` directly,
so deltas are exact; rounding happens only for presentation.

Replays saved predictions. Reads the released annotations read-only and never
modifies an id.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.proposal_first_pidray_occ.evaluation.zero_id_compat import ZeroIdSafeCOCOeval

ARMS = [("01", "B2", "pixray_b2"), ("02", "P1", "pixray_p1"), ("03", "P2", "pixray_p2"),
        ("04", "P3", "pixray_p3"), ("05", "B5", "pixray_b5"), ("06", "S2", "pixray_s2"),
        ("07", "B8", "pixray_b8")]
STATS = {"AP": 0, "AP50": 1, "AP75": 2, "AP_s": 3, "AP_m": 4, "AP_l": 5, "AR@100": 8}


def to_coco(preds, iou_type):
    out = []
    for s in preds:
        p = s["pred_instances"]
        boxes = np.asarray(p["bboxes"]); scores = np.asarray(p["scores"])
        labels = np.asarray(p["labels"])
        masks = p.get("masks") if iou_type == "segm" else None
        for i in range(len(scores)):
            d = {"image_id": int(s["img_id"]), "category_id": int(labels[i]),
                 "score": float(scores[i])}
            if iou_type == "bbox":
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                d["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                if masks is None:
                    continue
                rle = masks[i]
                if isinstance(rle, dict) and isinstance(rle.get("counts"), bytes):
                    rle = {**rle, "counts": rle["counts"].decode()}
                d["segmentation"] = rle
            out.append(d)
    return out


def evaluate(coco, dets, iou_type, eval_cls):
    """Score one detection set.

    ``COCO.loadRes`` mutates the list it is given: the segmentation branch adds
    a ``bbox`` field, and its bbox branch is tested *first*, so a second call on
    the same list silently takes the bbox path and recomputes ``area`` as
    width*height instead of mask area. That changes every detection's COCO area
    range. Each call therefore gets its own deep copy.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        if not dets:
            return None, 0
        coco_dt = coco.loadRes(copy.deepcopy(dets))
        ev = eval_cls(coco, coco_dt, iou_type)
        ev.params.maxDets = [100, 300, 1000]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats, getattr(ev, "zero_id_repairs", 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="data/PIXray/annotations/test.json")
    ap.add_argument("--out", default="output/audit/2026-09-11/zero_id_full_precision.json")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(args.annotations)
    payload = {"annotations": args.annotations, "arms": []}

    print(f"{'ID':<4}{'hist':<5}{'iou':<6}"
          + "".join(f"{k:>26}" for k in ("AP", "AP50", "AP75")))
    print(f"{'':<15}" + "".join(f"{'official':>13}{'compat':>13}" for _ in range(3)))
    for nid, hist, wd in ARMS:
        pkl = Path(f"work_dirs/{wd}/test/test/predictions.pkl")
        if not pkl.is_file():
            print(f"{nid:<4}{hist:<5}  predictions missing"); continue
        preds = pickle.load(open(pkl, "rb"))
        entry = {"id": nid, "historical_id": hist, "iou_types": {}}
        for iou_type in ("bbox", "segm"):
            dets = to_coco(preds, iou_type)
            if not dets:
                entry["iou_types"][iou_type] = None; continue
            off, _ = evaluate(coco, dets, iou_type, COCOeval)
            com, rep = evaluate(coco, dets, iou_type, ZeroIdSafeCOCOeval)
            rec = {"repairs": int(rep), "official": {}, "compat": {}, "delta": {}}
            for name, idx in STATS.items():
                o, c = float(off[idx]), float(com[idx])
                rec["official"][name] = o
                rec["compat"][name] = c
                rec["delta"][name] = c - o
            entry["iou_types"][iou_type] = rec
            print(f"{nid:<4}{hist:<5}{iou_type:<6}"
                  + "".join(f"{rec['official'][k]*100:>13.6f}{rec['compat'][k]*100:>13.6f}"
                            for k in ("AP", "AP50", "AP75")))
        payload["arms"].append(entry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
