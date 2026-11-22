#!/usr/bin/env python
"""Category-safe version of ``scripts/campaign_multiseed/coco_resample.py`` (the original is unchanged).

The original ``_to_coco`` writes a detection's class index as its COCO ``category_id``. That holds only
when category ids are 0..C-1 (PIXray). Here the class index ``i`` is mapped to the i-th category id of
the ground-truth file, sorted ascending, and the category names can be checked against the model's
class order -- the same rule MMDetection's CocoMetric and ``rescore_predictions.py`` use. For PIXray
the mapping is the identity, so results are unchanged. Everything that operates on the cached matches
(``ap_weighted``, ``UnionCache``, ``self_check``) is reused from the original module.
"""
from __future__ import annotations

import contextlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_multiseed.coco_resample import (CatCache, UnionCache, ap_weighted,  # noqa: E402,F401
                                                      self_check)


def label_to_category(gt: COCO, classes=None) -> dict:
    """Class index -> category id, by MMDetection's own rule.

    With ``classes`` (the model's class names), this reproduces ``CocoDataset``: ``cat_ids =
    coco.get_cat_ids(cat_names=classes)`` (ids of the named categories in annotation-file order) and
    label ``i`` -> ``cat_ids[i]`` (mmdet/datasets/coco.py:70-72). Every name must exist exactly once.
    Without ``classes`` it falls back to ascending ids, which is correct only when ids are 0..C-1.
    """
    if classes is not None:
        ids = gt.getCatIds(catNms=list(classes))
        names = [c["name"] for c in gt.loadCats(ids)]
        if len(ids) != len(classes) or sorted(names) != sorted(classes):
            raise ValueError(f"annotation categories {names} do not match the model classes {list(classes)}")
        return {i: cid for i, cid in enumerate(ids)}
    return {i: c["id"] for i, c in enumerate(sorted(gt.loadCats(gt.getCatIds()), key=lambda c: c["id"]))}


def to_coco_mapped(samples, iou_type, mapping):
    out = []
    for s in samples:
        p = s["pred_instances"]
        boxes = np.asarray(p["bboxes"]); scores = np.asarray(p["scores"]); labels = np.asarray(p["labels"])
        for i in range(len(scores)):
            d = {"image_id": int(s["img_id"]), "category_id": mapping[int(labels[i])], "score": float(scores[i])}
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


class APCacheMapped:
    """Same cache as ``coco_resample.APCache``, built from correctly mapped detections."""

    def __init__(self, gt: COCO, pred_path, iou_type: str = "segm", classes=None):
        self.mapping = label_to_category(gt, classes)
        samples = pickle.load(open(pred_path, "rb"))
        dets = to_coco_mapped(samples, iou_type, self.mapping)
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), iou_type)
            ev.evaluate(); ev.accumulate(); ev.summarize()
        self.official = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}
        p = ev.params
        self.img_ids = list(p.imgIds)
        self.cat_ids = list(p.catIds)
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        max_det = p.maxDets[-1]
        img_pos = {i: k for k, i in enumerate(self.img_ids)}
        self.cats = []
        for k, _ in enumerate(self.cat_ids):
            base = k * n_area * n_img
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
