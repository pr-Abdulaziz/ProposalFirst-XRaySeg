#!/usr/bin/env python
"""Full-precision scoring of one prediction file against the official PIXray test split.

One scorer for every run. Official pycocotools ``COCOeval`` with its standard
parameters (IoU 0.50:0.05:0.95, maxDets 1/10/100, COCO area ranges) gives the
primary numbers. PIXray's ``area`` field is the bounding-box area, so the
small/medium/large split bins instances by box area, not mask area.

A second pass uses ``ZeroIdSafeCOCOeval``, which repairs only the evaluator's
internal match array for the ground-truth annotation whose id is 0. It is reported
beside the official values and never substituted for them. Precision, recall and
F1 use the repaired matching (COCO's greedy, score-ordered, class-wise one-to-one
assignment, area 'all', at most 100 detections per image).

    python scripts/campaign_multiseed/score_predictions.py <predictions.pkl> <out.json>
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from projects.proposal_first_pidray_occ.evaluation.zero_id_compat import ZeroIdSafeCOCOeval  # noqa: E402
from scripts.campaign_multiseed.campaign_common import ANN, sha256  # noqa: E402

S12 = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
       "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
THR = np.round(np.arange(0.05, 0.9501, 0.01), 2)
EVALUATOR = dict(implementation="pycocotools COCOeval", iouThrs="0.50:0.05:0.95",
                 maxDets=[1, 10, 100], areaRng="COCO default (all, <32^2, 32^2-96^2, >96^2)",
                 area_field_semantics="PIXray 'area' = bounding-box area",
                 zero_id_policy="official values primary; ZeroIdSafeCOCOeval reported separately",
                 prf_matching="ZeroIdSafeCOCOeval dtMatches, area all, maxDet 100")


def to_coco(samples, iou_type):
    out = []
    for s in samples:
        p = s["pred_instances"]
        boxes = np.asarray(p["bboxes"]); scores = np.asarray(p["scores"])
        labels = np.asarray(p["labels"])
        masks = p.get("masks") if iou_type == "segm" else None
        for i in range(len(scores)):
            d = {"image_id": int(s["img_id"]), "category_id": int(labels[i]), "score": float(scores[i])}
            if iou_type == "bbox":
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                d["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                rle = masks[i]
                if isinstance(rle, dict) and isinstance(rle.get("counts"), bytes):
                    rle = {**rle, "counts": rle["counts"].decode()}
                d["segmentation"] = rle
            out.append(d)
    return out


def run(coco, dets, iou_type, cls):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = cls(coco, coco.loadRes(copy.deepcopy(dets)), iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


def prf(ev, ti):
    p = ev.params
    per = {c: {"s": [], "tp": [], "ngt": 0} for c in p.catIds}
    for e in ev.evalImgs:
        if e is None or e["aRng"] != p.areaRng[0] or e["maxDet"] != p.maxDets[-1]:
            continue
        c = e["category_id"]
        per[c]["ngt"] += int((~np.asarray(e["gtIgnore"], bool)).sum())
        if len(e["dtScores"]):
            dm = np.asarray(e["dtMatches"])[ti]; di = np.asarray(e["dtIgnore"])[ti].astype(bool)
            per[c]["s"].append(np.asarray(e["dtScores"])[~di]); per[c]["tp"].append(dm[~di] != 0)
    tot_tp = np.zeros(len(THR)); tot_fp = np.zeros(len(THR)); tot_gt = 0; per_class = {}; f1s = []
    for c, d in per.items():
        s = np.concatenate(d["s"]) if d["s"] else np.zeros(0)
        tp = np.concatenate(d["tp"]) if d["tp"] else np.zeros(0, bool)
        TP = np.array([(tp & (s >= t)).sum() for t in THR], float)
        FP = np.array([(~tp & (s >= t)).sum() for t in THR], float)
        tot_tp += TP; tot_fp += FP; tot_gt += d["ngt"]
        P = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
        R = TP / d["ngt"] if d["ngt"] else np.zeros_like(TP)
        F = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) > 0)
        f1s.append(F)
        i5 = int(np.where(THR == 0.5)[0][0])
        per_class[str(c)] = {"P": P[i5], "R": R[i5], "F1": F[i5], "TP": TP[i5], "FP": FP[i5], "num_gt": d["ngt"]}
    P = np.divide(tot_tp, tot_tp + tot_fp, out=np.zeros_like(tot_tp), where=(tot_tp + tot_fp) > 0)
    R = tot_tp / tot_gt
    F = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) > 0)
    macro = np.mean(f1s, axis=0)
    i5 = int(np.where(THR == 0.5)[0][0]); ib = int(np.argmax(F))
    return {"thresholds": THR.tolist(), "micro_P": P.tolist(), "micro_R": R.tolist(), "micro_F1": F.tolist(),
            "macro_F1": macro.tolist(), "num_gt": tot_gt,
            "at_0.50": {"P": P[i5], "R": R[i5], "F1": F[i5], "macro_F1": macro[i5],
                        "TP": tot_tp[i5], "FP": tot_fp[i5], "FN": tot_gt - tot_tp[i5]},
            "best": {"threshold": float(THR[ib]), "P": P[ib], "R": R[ib], "F1": F[ib]},
            "per_class_at_0.50": per_class}


def score(pred_path: Path, ann_file: Path = ANN["test"]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann_file))
    samples = pickle.load(open(pred_path, "rb"))
    img_ids = sorted(int(s["img_id"]) for s in samples)
    dets_per_img = [len(s["pred_instances"]["scores"]) for s in samples]
    coverage = dict(prediction_images=len(img_ids), unique_images=len(set(img_ids)),
                    test_images=len(coco.getImgIds()),
                    images_match=sorted(set(img_ids)) == sorted(coco.getImgIds()),
                    max_detections_per_image=max(dets_per_img), total_detections=sum(dets_per_img),
                    min_score=float(min((float(np.asarray(s["pred_instances"]["scores"]).min())
                                         for s in samples if len(s["pred_instances"]["scores"])), default=0.0)),
                    categories=len(coco.getCatIds()))
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    out = {"predictions": str(pred_path), "predictions_sha256": sha256(pred_path),
           "annotations": str(ann_file), "annotations_sha256": sha256(ann_file),
           "evaluator": EVALUATOR, "coverage": coverage}
    for t in ("bbox", "segm"):
        dets = to_coco(samples, t)
        ev = run(coco, dets, t, COCOeval)
        evz = run(coco, dets, t, ZeroIdSafeCOCOeval)
        pr = ev.eval["precision"]; rc = ev.eval["recall"]
        per_class = {}
        for i, c in enumerate(cats):
            q = pr[:, :, i, 0, 2]; r = rc[:, i, 0, 2]
            q50 = pr[0, :, i, 0, 2]; q75 = pr[5, :, i, 0, 2]
            per_class[c["name"]] = {"AP": float(q[q > -1].mean()), "AP50": float(q50[q50 > -1].mean()),
                                    "AP75": float(q75[q75 > -1].mean()), "AR100": float(r[r > -1].mean())}
        out[t] = {"stats": dict(zip(S12, map(float, ev.stats))),
                  "stats_zero_id_safe": dict(zip(S12, map(float, evz.stats))),
                  "per_class": per_class,
                  "ap_by_iou": [float(pr[i, :, :, 0, 2][pr[i, :, :, 0, 2] > -1].mean()) for i in range(10)],
                  "prf_iou_0.50": prf(evz, 0), "prf_iou_0.75": prf(evz, 5)}
    return out


def main() -> int:
    pred, dest = Path(sys.argv[1]), Path(sys.argv[2])
    result = score(pred)
    dest.write_text(json.dumps(result, indent=1, default=float))
    for t in ("bbox", "segm"):
        s = result[t]["stats"]
        print(f"{t}: AP {100*s['AP']:.3f} AP50 {100*s['AP50']:.3f} AP75 {100*s['AP75']:.3f} AR100 {100*s['AR100']:.3f}")
    print("coverage:", result["coverage"])
    return 0 if result["coverage"]["images_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
