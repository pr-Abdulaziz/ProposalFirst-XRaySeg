#!/usr/bin/env python
"""Fast re-computation of COCO AP under image weights, for bootstrap and permutation tests.

pycocotools' ``evaluate()`` is run once per prediction file; its per-image match results
(area 'all', maxDet 100) are cached per category. AP for any vector of image weights is then
recomputed without re-matching: detections keep their global score order, cumulative TP/FP
counts and the number of ground-truth objects are weighted by how many times each image is
drawn. With all weights equal to 1 this reproduces the official COCOeval AP exactly (checked
by ``self_check``). Weighted detections also give exact paired-permutation statistics: the
union of two methods' detections is scored with per-image selector weights.
"""
from __future__ import annotations

import contextlib
import io
import pickle
from dataclasses import dataclass

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REC_THRS = np.linspace(0.0, 1.0, 101)


def _to_coco(samples, iou_type):
    out = []
    for s in samples:
        p = s["pred_instances"]
        boxes = np.asarray(p["bboxes"]); scores = np.asarray(p["scores"]); labels = np.asarray(p["labels"])
        for i in range(len(scores)):
            d = {"image_id": int(s["img_id"]), "category_id": int(labels[i]), "score": float(scores[i])}
            if iou_type == "bbox":
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
                d["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                m = dict(p["masks"][i])
                if isinstance(m["counts"], bytes):
                    m["counts"] = m["counts"].decode()
                d["segmentation"] = m
            out.append(d)
    return out


@dataclass
class CatCache:
    img: np.ndarray        # [D] image index of each detection (in global score order)
    tp: np.ndarray         # [T, D] bool
    fp: np.ndarray         # [T, D] bool
    n_gt: np.ndarray       # [N] non-ignored GT per image
    score: np.ndarray      # [D]


class APCache:
    """Per-category cached matches for one prediction file and one IoU type."""

    def __init__(self, gt: COCO, pred_path, iou_type: str = "segm"):
        samples = pickle.load(open(pred_path, "rb"))
        dets = _to_coco(samples, iou_type)
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        self.official = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}
        p = ev.params
        self.img_ids = list(p.imgIds)
        self.cat_ids = list(p.catIds)
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        max_det = p.maxDets[-1]
        img_pos = {i: k for k, i in enumerate(self.img_ids)}
        self.cats: list[CatCache | None] = []
        for k, _ in enumerate(self.cat_ids):
            base = k * n_area * n_img          # area index 0 = 'all'
            E = [ev.evalImgs[base + i] for i in range(n_img)]
            E = [e for e in E if e is not None]
            n_gt = np.zeros(n_img, np.int64)
            for e in E:
                n_gt[img_pos[e["image_id"]]] = int(np.count_nonzero(e["gtIgnore"] == 0))
            if not E or n_gt.sum() == 0:
                self.cats.append(None)
                continue
            scores = np.concatenate([e["dtScores"][:max_det] for e in E])
            order = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in E], axis=1)[:, order]
            dtig = np.concatenate([e["dtIgnore"][:, :max_det] for e in E], axis=1)[:, order]
            img = np.concatenate([np.full(min(len(e["dtScores"]), max_det), img_pos[e["image_id"]]) for e in E])[order]
            self.cats.append(CatCache(img=img, tp=np.logical_and(dtm, ~dtig), fp=np.logical_and(dtm == 0, ~dtig),
                                      n_gt=n_gt, score=scores[order]))


def _precision_table(tp, fp, wdet, npig):
    """Interpolated precision at the 101 recall thresholds. tp/fp [D] bool, wdet [R, D], npig [R] -> [R, 101]."""
    tps = np.cumsum(wdet * tp, axis=1)
    fps = np.cumsum(wdet * fp, axis=1)
    rc = tps / np.maximum(npig[:, None], 1e-12)
    pr = tps / np.maximum(tps + fps, np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    q = np.zeros((len(npig), len(REC_THRS)))
    for r in range(len(npig)):
        inds = np.searchsorted(rc[r], REC_THRS, side="left")
        ok = inds < rc.shape[1]
        q[r, ok] = pr[r, inds[ok]]
    return q


def ap_weighted(cache: APCache, w_img: np.ndarray, det_sel=None) -> dict:
    """AP, AP50, AP75 and per-class AP for image weights w_img [R, N].

    det_sel: optional list (per category) of [R, D] detection weights multiplied in (permutation tests).
    Returns arrays of shape [R] (and [R, K] for per_class; NaN for classes without GT).
    """
    R = w_img.shape[0]
    K = len(cache.cats)
    T = 10
    q_all = np.full((R, T, K, len(REC_THRS)), np.nan)
    for k, c in enumerate(cache.cats):
        if c is None:
            continue
        npig = w_img @ c.n_gt
        wdet = w_img[:, c.img]
        if det_sel is not None:
            wdet = wdet * det_sel[k]
        for t in range(T):
            q_all[:, t, k] = _precision_table(c.tp[t], c.fp[t], wdet, npig)
            q_all[npig <= 0, t, k] = np.nan
    with np.errstate(invalid="ignore"):
        return {"AP": np.nanmean(q_all, axis=(1, 2, 3)), "AP50": np.nanmean(q_all[:, 0], axis=(1, 2)),
                "AP75": np.nanmean(q_all[:, 5], axis=(1, 2)), "per_class": np.nanmean(q_all, axis=(1, 3))}


class UnionCache:
    """Detections of two methods on the same images, merged in one score order, for paired permutation tests."""

    def __init__(self, a: APCache, b: APCache):
        assert a.img_ids == b.img_ids and a.cat_ids == b.cat_ids
        self.img_ids, self.cat_ids = a.img_ids, a.cat_ids
        self.cats, self.from_a = [], []
        for ca, cb in zip(a.cats, b.cats):
            if ca is None and cb is None:
                self.cats.append(None); self.from_a.append(None); continue
            parts = [c for c in (ca, cb) if c is not None]
            src = np.concatenate([np.full(len(c.img), c is ca) for c in parts])
            score = np.concatenate([c.score for c in parts])
            order = np.argsort(-score, kind="mergesort")
            n_gt = (ca or cb).n_gt
            self.cats.append(CatCache(img=np.concatenate([c.img for c in parts])[order],
                                      tp=np.concatenate([c.tp for c in parts], axis=1)[:, order],
                                      fp=np.concatenate([c.fp for c in parts], axis=1)[:, order],
                                      n_gt=n_gt, score=score[order]))
            self.from_a.append(src[order])

    def ap_pair(self, swap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """swap [R, N] in {0,1}: 1 = image i takes the other method's detections. Returns (AP_A', AP_B') [R]."""
        R = swap.shape[0]
        ones = np.ones_like(swap, dtype=float)
        sel_a, sel_b = [], []
        for c, fa in zip(self.cats, self.from_a):
            if c is None:
                sel_a.append(None); sel_b.append(None); continue
            s = swap[:, c.img]
            sel_a.append(np.where(fa[None, :], 1.0 - s, s))    # A' uses A's detections unless swapped
            sel_b.append(np.where(fa[None, :], s, 1.0 - s))
        return (ap_weighted(self, ones, sel_a)["AP"], ap_weighted(self, ones, sel_b)["AP"])


def self_check(cache: APCache) -> float:
    """Largest absolute difference between the cached recomputation and the official AP/AP50/AP75."""
    r = ap_weighted(cache, np.ones((1, len(cache.img_ids))))
    return max(abs(float(r[k][0]) - cache.official[k]) for k in ("AP", "AP50", "AP75"))
