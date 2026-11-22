#!/usr/bin/env python
"""Why does small-object AR fall for the refiners while small-object AP rises?

For the Parent and both refiners, all three seeds, on PIXray's 'small' bin (box area < 32^2: PIXray's
'area' field is the box area), from the campaign's official prediction files:
  1. COCOeval recall and precision per IoU threshold (maxDets=100), mask and box;
  2. per small ground-truth object: is there a same-class detection with box IoU >= 0.5 (score >= 0.05),
     and the best mask IoU among those detections;
  3. counts of small objects whose best mask IoU falls below each threshold, split into
     "no box-level detection" and "detected, mask too poor".
Writes output/<campaign>/revision_2026-11/a4_small_objects.json.
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from scripts.campaign_multiseed.analyze_errors_quality import predictions_path  # noqa: E402
from scripts.campaign_multiseed.campaign_common import ANN, OUT, SEEDS  # noqa: E402
from scripts.campaign_multiseed.score_predictions import to_coco  # noqa: E402

ARMS = ("03", "09", "09M")
THR = np.round(np.arange(0.5, 0.951, 0.05), 2)


def coco_curves(gt, dets, iou_type):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    rec = ev.eval["recall"][:, :, 1, 2]                 # [T, K] small, maxDet 100
    prec = ev.eval["precision"][:, :, :, 1, 2]          # [T, R, K]
    rec = np.where(rec < 0, np.nan, rec)
    prec = np.where(prec < 0, np.nan, prec)
    return dict(AR_s=float(np.nanmean(rec)), AP_s=float(np.nanmean(prec)),
                recall_by_iou=np.nanmean(rec, axis=1).tolist(), ap_by_iou=np.nanmean(prec, axis=(1, 2)).tolist())


def per_object(gt, samples):
    small = [a for a in gt.loadAnns(gt.getAnnIds(iscrowd=None)) if a["area"] < 32 ** 2]
    by_img = {}
    for s in samples:
        by_img[int(s["img_id"])] = s["pred_instances"]
    out = []
    for a in small:
        p = by_img.get(a["image_id"])
        g_rle = gt.annToRLE(a)
        x, y, w, h = a["bbox"]
        best_box, best_mask = 0.0, 0.0
        if p is not None and len(p["scores"]):
            labels = np.asarray(p["labels"]); keep = np.where(labels == a["category_id"])[0]
            if len(keep):
                b = np.asarray(p["bboxes"])[keep]
                bx = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
                box_iou = mask_util.iou(bx.tolist(), [[x, y, w, h]], [0]).ravel()
                ok = np.where(box_iou >= 0.5)[0]
                best_box = float(box_iou.max())
                if len(ok):
                    masks = [p["masks"][keep[i]] for i in ok]
                    best_mask = float(mask_util.iou(masks, [g_rle], [0]).max())
        out.append(dict(id=a["id"], cat=a["category_id"], box_area=a["area"],
                        mask_area=float(mask_util.area(g_rle)), best_box_iou=best_box, best_mask_iou=best_mask))
    return out


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN["test"]))
    res = dict(note="small = PIXray 'area' (box area) < 32^2; maxDets 100; per-object uses same-class "
                    "detections with box IoU >= 0.5 at score >= 0.05", runs={})
    for mid in ARMS:
        for s in SEEDS:
            samples = pickle.load(open(predictions_path(mid, s), "rb"))
            r = dict(segm=coco_curves(gt, to_coco(samples, "segm"), "segm"),
                     bbox=coco_curves(gt, to_coco(samples, "bbox"), "bbox"))
            objs = per_object(gt, samples)
            m = np.array([o["best_mask_iou"] for o in objs]); det = np.array([o["best_box_iou"] >= 0.5 for o in objs])
            r["objects"] = dict(n=len(objs), box_detected=int(det.sum()),
                                mean_best_mask_iou_detected=float(m[det].mean()),
                                below={f"{t:.2f}": dict(undetected=int((~det).sum()),
                                                         detected_mask_below=int((det & (m < t)).sum()))
                                       for t in THR})
            r["per_object"] = objs
            res["runs"][f"{mid}/seed_{s}"] = r
            print(f"{mid} s{s}: segm AR_s {100 * r['segm']['AR_s']:.1f} AP_s {100 * r['segm']['AP_s']:.1f} | "
                  f"box AR_s {100 * r['bbox']['AR_s']:.1f} | detected {det.sum()}/{len(objs)} "
                  f"mean mask IoU {m[det].mean():.3f}", flush=True)
    (OUT / "revision_2026-11" / "a4_small_objects.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
