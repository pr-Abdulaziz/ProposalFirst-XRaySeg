#!/usr/bin/env python
"""Extended evaluation of every campaign run (all methods x seeds), from the saved predictions.

Complements the per-run COCO scoring done by evaluate_run.py (12 COCO stats, per-class AP, AP by IoU,
P/R/F1). Uses the same protocols as the earlier supplementary report so numbers are comparable:

1. Detection errors and confusion matrix (box IoU, score >= 0.50). Predictions are processed in
   descending score order per image and labelled TP / duplicate / wrong class / poor localisation /
   other / background; unmatched ground truth is 'missed'. Confusion rows: GT classes + background;
   columns: predicted classes + missed. A GT without a TP is assigned to its highest-IoU wrong-class
   prediction (IoU >= 0.5) if one exists.
2. Matched-instance mask quality: score >= 0.50, one-to-one same-class matching at box IoU >= 0.50
   (Hungarian), mask IoU, Dice, boundary IoU (band = round(0.02 x image diagonal)) at original
   resolution; per-GT values kept for paired comparisons between methods on the same seed.
3. Class-averaged precision-recall curves (101 recall points) at IoU 0.50 and 0.75, box and mask.
4. Paired mask-quality comparisons for fixed method pairs, per seed, on GT matched by both.

Reads predictions only; writes output/<campaign>/analysis/. Never modifies predictions or annotations.

    python scripts/campaign_multiseed/analyze_errors_quality.py [--workers 8]
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.campaign_common import ANN, BY_ID, METHODS, OUT, SEEDS, eval_dir, sha256  # noqa: E402

AN = OUT / "analysis"
SCORE = 0.50
PAIRS = [("09M", "03"), ("09", "03"), ("09M", "09"), ("02", "01"), ("03", "02"), ("04", "03"),
         ("08", "03"), ("07", "04"), ("06", "05")]
_COCO = None


def coco():
    global _COCO
    if _COCO is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _COCO = COCO(str(ANN["test"]))
    return _COCO


def predictions_path(mid: str, seed: int) -> Path:
    recs = sorted(glob.glob(str(eval_dir(mid, seed) / "eval_record_job*.json")))
    if not recs:
        raise FileNotFoundError(f"no evaluation record for {mid} seed {seed}")
    return Path(json.loads(Path(recs[-1]).read_text())["predictions"])


def rle_to_mask(m):
    r = dict(m)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode()
    return mask_util.decode(r).astype(np.uint8)


def boundary_band(mask, d):
    if d < 1:
        return mask.copy()
    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(pad, np.ones((3, 3), np.uint8), iterations=d)
    return mask - eroded[1:-1, 1:-1]


def box_iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def gt_tables():
    c = coco()
    cats = sorted(c.getCatIds())
    out = {}
    for iid in sorted(c.getImgIds()):
        info = c.loadImgs(iid)[0]
        anns = c.loadAnns(c.getAnnIds(imgIds=iid, iscrowd=None))
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns]).reshape(-1, 4)
        out[iid] = dict(ids=[a["id"] for a in anns], cls=np.array([a["category_id"] for a in anns], int), boxes=boxes,
                        anns=anns, d=int(round(0.02 * np.hypot(info["height"], info["width"]))))
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
            per_class[int(c)]["gt"] += 1
        for i in range(len(pb)):
            row = M[i] if len(gb) else np.zeros(0)
            same = gc == pc[i] if len(gb) else np.zeros(0, bool)
            su = np.where(same & ~matched & (row >= 0.5))[0]
            if len(su):
                j = su[np.argmax(row[su])]; matched[j] = True; cnt["TP"] += 1
                per_class[int(pc[i])]["TP"] += 1; conf[gc[j], pc[i]] += 1
                continue
            per_class[int(pc[i])]["FP"] += 1
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
            cnt["missed"] += 1; per_class[int(gc[j])]["missed"] += 1
    fp = sum(cnt[x] for x in ("duplicate", "wrong_class", "poor_localization", "other", "background"))
    return dict(counts=cnt, false_positives=fp, num_gt=int(sum(len(g["cls"]) for g in gts.values())),
                confusion=conf.tolist(), per_class={str(k): v for k, v in per_class.items()})


def mask_quality(preds, gts):
    c = coco()
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


def pr_curves(samples):
    c = coco()
    out = {}
    for t in ("bbox", "segm"):
        dets = []
        for s in samples:
            p = s["pred_instances"]; b = np.asarray(p["bboxes"]); sc = np.asarray(p["scores"]); lb = np.asarray(p["labels"])
            for i in range(len(sc)):
                d = dict(image_id=int(s["img_id"]), category_id=int(lb[i]), score=float(sc[i]))
                if t == "bbox":
                    d["bbox"] = [float(b[i, 0]), float(b[i, 1]), float(b[i, 2] - b[i, 0]), float(b[i, 3] - b[i, 1])]
                else:
                    m = dict(p["masks"][i]); m["counts"] = m["counts"].decode() if isinstance(m["counts"], bytes) else m["counts"]
                    d["segmentation"] = m
                dets.append(d)
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(c, c.loadRes(dets), t); ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"]
        out[t] = {f"iou_{iou}": [float(np.mean(x[x > -1])) if (x > -1).any() else None for x in pr[ti, :, :, 0, 2]]
                  for ti, iou in ((0, "0.50"), (5, "0.75"))}
    out["recall_points"] = np.linspace(0, 1, 101).tolist()
    return out


def run_one(args):
    mid, seed = args
    cats, gts = gt_tables()
    path = predictions_path(mid, seed)
    samples = pickle.load(open(path, "rb"))
    preds = {s["img_id"]: s for s in samples}
    err = errors_and_confusion(preds, cats, gts)
    mq, per_gt = mask_quality(preds, gts)
    pr = pr_curves(samples)
    d = AN / "runs" / BY_ID[mid].slug
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(method=mid, seed=seed, predictions=str(path), predictions_sha256=sha256(path),
               classes=[c["name"] for c in coco().loadCats(cats)], errors=err, mask_quality=mq, pr_curves=pr)
    (d / f"seed_{seed}.json").write_text(json.dumps(rec, indent=1))
    np.save(d / f"seed_{seed}_per_gt.npy", np.array([[k, *v] for k, v in sorted(per_gt.items())], dtype=float))
    return f"{mid} s{seed}: TP {err['counts']['TP']} FP {err['false_positives']} missed {err['counts']['missed']} | " \
           f"IoU {mq['mean_iou']:.4f} BIoU {mq['mean_boundary_iou']:.4f}"


def paired():
    rows = []
    for a, b in PAIRS:
        for seed in SEEDS:
            pa = AN / "runs" / BY_ID[a].slug / f"seed_{seed}_per_gt.npy"
            pb = AN / "runs" / BY_ID[b].slug / f"seed_{seed}_per_gt.npy"
            A = {int(r[0]): r[1:] for r in np.load(pa)}; B = {int(r[0]): r[1:] for r in np.load(pb)}
            common = sorted(set(A) & set(B))
            da = np.array([A[k] for k in common]); db = np.array([B[k] for k in common])
            diff = da - db
            rows.append(dict(a=a, b=b, seed=seed, common_gt=len(common), d_iou=float(diff[:, 0].mean()),
                             d_dice=float(diff[:, 1].mean()), d_boundary_iou=float(diff[:, 2].mean()),
                             share_improved=float((diff[:, 0] > 0).mean())))
    (AN / "paired_mask_quality.json").write_text(json.dumps(rows, indent=1))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(m.id, s) for m in METHODS for s in SEEDS]
    missing = [j for j in jobs if not glob.glob(str(eval_dir(*j) / "eval_record_job*.json"))]
    if missing:
        raise SystemExit(f"ABORT: evaluation records missing for {missing}")
    AN.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for line in pool.imap_unordered(run_one, jobs):
            print(line, flush=True)
    for r in paired():
        print(f"{r['a']} vs {r['b']} s{r['seed']}: dIoU {100*r['d_iou']:+.2f} dBIoU {100*r['d_boundary_iou']:+.2f} "
              f"improved {100*r['share_improved']:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
