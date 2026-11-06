#!/usr/bin/env python
"""Category-safe version of ``scripts/campaign_multiseed/analyze_errors_quality.py`` (original unchanged).

The original compares ground-truth ``category_id`` values with predicted class indices and indexes the
confusion matrix by category id. Both hold only when category ids are 0..C-1 (PIXray). Here every
ground-truth class is converted to its contiguous index (position of its id in the sorted id list),
all comparisons and the confusion matrix use that index, per-class counts stay keyed by category id,
and precision-recall curves receive mapped category ids. For PIXray the mapping is the identity, so
the output is unchanged. Matching rules, thresholds and the output schema are exactly the original's.

    python scripts/campaign_pidray_bs1/analyze_errors_quality_mapped.py <predictions.pkl> <annotations.json> [out.json]
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.analyze_errors_quality import (SCORE, boundary_band, box_iou_matrix,  # noqa: E402
                                                               rle_to_mask)
from scripts.campaign_pidray_bs1.coco_resample_mapped import label_to_category, to_coco_mapped  # noqa: E402


def gt_tables(c: COCO, mapping=None):
    """``mapping`` (class index -> category id) fixes the contiguous order; default: ascending ids."""
    cats = [mapping[i] for i in range(len(mapping))] if mapping is not None else sorted(c.getCatIds())
    index = {cid: k for k, cid in enumerate(cats)}                  # category id -> contiguous index
    out = {}
    for iid in sorted(c.getImgIds()):
        info = c.loadImgs(iid)[0]
        anns = c.loadAnns(c.getAnnIds(imgIds=iid, iscrowd=None))
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns]).reshape(-1, 4)
        out[iid] = dict(ids=[a["id"] for a in anns], cls=np.array([index[a["category_id"]] for a in anns], int),
                        boxes=boxes, anns=anns, d=int(round(0.02 * np.hypot(info["height"], info["width"]))))
    return cats, out


def errors_and_confusion(preds, cats, gts):
    C = len(cats)
    cnt = dict(TP=0, duplicate=0, wrong_class=0, poor_localization=0, other=0, background=0, missed=0)
    conf = np.zeros((C + 1, C + 1), int)
    per_class = {c: dict(TP=0, FP=0, missed=0, gt=0) for c in cats}
    for iid, g in gts.items():
        gb, gc = g["boxes"], g["cls"]
        s = preds.get(iid)
        if s is None:
            pb, pc, ps = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
        else:
            p = s["pred_instances"]; ps = np.asarray(p["scores"]); keep = ps >= SCORE
            pb = np.asarray(p["bboxes"])[keep]; pc = np.asarray(p["labels"])[keep].astype(int); ps = ps[keep]
            o = np.argsort(-ps, kind="stable"); pb, pc, ps = pb[o], pc[o], ps[o]
        M = box_iou_matrix(pb, gb)
        matched = np.zeros(len(gb), bool); wrong_for_gt = {}
        for c in gc:
            per_class[cats[c]]["gt"] += 1
        for i in range(len(pb)):
            row = M[i] if len(gb) else np.zeros(0)
            same = gc == pc[i] if len(gb) else np.zeros(0, bool)
            su = np.where(same & ~matched & (row >= 0.5))[0]
            if len(su):
                j = su[np.argmax(row[su])]; matched[j] = True; cnt["TP"] += 1
                per_class[cats[pc[i]]]["TP"] += 1; conf[gc[j], pc[i]] += 1
                continue
            per_class[cats[pc[i]]]["FP"] += 1
            if len(gb) and (same & (row >= 0.5)).any():
                cnt["duplicate"] += 1
            elif len(gb) and (~same & (row >= 0.5)).any():
                cnt["wrong_class"] += 1
                j = np.where(~same & (row >= 0.5))[0]; j = j[np.argmax(row[j])]
                wrong_for_gt.setdefault(int(j), int(pc[i]))
            elif len(gb) and (same & (row >= 0.1)).any():
                cnt["poor_localization"] += 1
            elif len(gb) and (row >= 0.1).any():
                cnt["other"] += 1
            else:
                cnt["background"] += 1; conf[C, pc[i]] += 1
        for j in np.where(~matched)[0]:
            if int(j) in wrong_for_gt:
                conf[gc[j], wrong_for_gt[int(j)]] += 1
            else:
                conf[gc[j], C] += 1
            cnt["missed"] += 1; per_class[cats[gc[j]]]["missed"] += 1
    fp = sum(cnt[x] for x in ("duplicate", "wrong_class", "poor_localization", "other", "background"))
    return dict(counts=cnt, false_positives=fp, num_gt=int(sum(len(g["cls"]) for g in gts.values())),
                confusion=conf.tolist(), per_class={str(k): v for k, v in per_class.items()})


def mask_quality(preds, gts, c: COCO):
    per_gt, unmatched, npred = {}, 0, 0
    for iid, g in gts.items():
        s = preds.get(iid)
        if s is None:
            continue
        p = s["pred_instances"]; sc = np.asarray(p["scores"]); lb = np.asarray(p["labels"]); keep = np.where(sc >= SCORE)[0]
        npred += len(keep)
        if not len(g["cls"]) or not len(keep):
            unmatched += len(keep); continue
        pb = np.asarray(p["bboxes"])[keep]
        M = box_iou_matrix(g["boxes"], pb)
        same = g["cls"][:, None] == lb[keep][None, :].astype(int)
        prof = np.where(same & (M >= 0.5), 1000.0 + M, 0.0)
        used = set()
        for gi, pi in zip(*linear_sum_assignment(-prof)):
            if prof[gi, pi] <= 0:
                continue
            used.add(pi)
            gm = c.annToMask(g["anns"][gi]).astype(np.uint8); pm = rle_to_mask(p["masks"][keep[pi]])
            inter = int(np.logical_and(gm, pm).sum()); sg, sp = int(gm.sum()), int(pm.sum())
            b1, b2 = boundary_band(gm, g["d"]), boundary_band(pm, g["d"]); bu = int(np.logical_or(b1, b2).sum())
            per_gt[int(g["ids"][gi])] = (inter / max(sg + sp - inter, 1), 2 * inter / max(sg + sp, 1),
                                          int(np.logical_and(b1, b2).sum()) / bu if bu else 0.0)
        unmatched += len(keep) - len(used)
    total = int(sum(len(g["cls"]) for g in gts.values()))
    v = np.array(list(per_gt.values())) if per_gt else np.zeros((0, 3))
    summary = dict(total_gt=total, matched_gt=len(per_gt), coverage=len(per_gt) / total, predictions_at_0_50=npred,
                   unmatched_predictions=unmatched, mean_iou=float(v[:, 0].mean()), median_iou=float(np.median(v[:, 0])),
                   mean_dice=float(v[:, 1].mean()), mean_boundary_iou=float(v[:, 2].mean()),
                   iou_ge_0_75=float((v[:, 0] >= 0.75).mean()), iou_ge_0_90=float((v[:, 0] >= 0.90).mean()))
    return summary, per_gt


def pr_curves(samples, c: COCO, mapping):
    out = {}
    for t in ("bbox", "segm"):
        dets = to_coco_mapped(samples, t, mapping)
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(c, c.loadRes(dets), t); ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"]
        out[t] = {f"iou_{iou}": [float(np.mean(x[x > -1])) if (x > -1).any() else None for x in pr[ti, :, :, 0, 2]]
                  for ti, iou in ((0, "0.50"), (5, "0.75"))}
    out["recall_points"] = np.linspace(0, 1, 101).tolist()
    return out


def analyze(pred_path, ann_path, classes=None):
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(str(ann_path))
    mapping = label_to_category(c, classes)
    cats, gts = gt_tables(c, mapping)
    samples = pickle.load(open(pred_path, "rb"))
    preds = {s["img_id"]: s for s in samples}
    err = errors_and_confusion(preds, cats, gts)
    mq, per_gt = mask_quality(preds, gts, c)
    pr = pr_curves(samples, c, mapping)
    rec = dict(predictions=str(pred_path), annotations=str(ann_path), label_to_category=mapping,
               classes=[x["name"] for x in c.loadCats(cats)], errors=err, mask_quality=mq, pr_curves=pr)
    return rec, per_gt


def main() -> int:
    rec, per_gt = analyze(Path(sys.argv[1]), Path(sys.argv[2]))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(rec, indent=1))
        np.save(Path(sys.argv[3]).with_suffix(".per_gt.npy"),
                np.array([[k, *v] for k, v in sorted(per_gt.items())], dtype=float))
    print(rec["errors"]["counts"], {k: rec["mask_quality"][k] for k in ("coverage", "mean_iou", "mean_boundary_iou")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
