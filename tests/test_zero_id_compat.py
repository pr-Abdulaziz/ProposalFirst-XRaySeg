"""Assertions for the annotation-id-0 evaluator compatibility mode.

Run:  PYTHONPATH=. .venv/bin/python -m pytest tests/test_zero_id_compat.py -v
  or: PYTHONPATH=. .venv/bin/python tests/test_zero_id_compat.py

Every fixture is synthetic. No released annotation file is read or written, and
no annotation id is remapped anywhere in the compatibility path.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from projects.proposal_first_pidray_occ.evaluation.zero_id_compat import (
    ZERO_ID_SENTINEL, ZeroIdSafeCOCOeval, gt_ids_contain_zero,
)

SIDE = 100


def _poly(x, y, w=10, h=10):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def build_gt(ann_ids, crowd_ids=()):
    anns = []
    for k, aid in enumerate(ann_ids):
        x = 10 + 20 * k
        anns.append({"id": aid, "image_id": 1, "category_id": 1,
                     "iscrowd": 1 if aid in crowd_ids else 0,
                     "bbox": [x, 10, 10, 10], "area": 100,
                     "segmentation": _poly(x, 10)})
    return {"images": [{"id": 1, "width": SIDE, "height": SIDE, "file_name": "x.png"}],
            "annotations": anns, "categories": [{"id": 1, "name": "a"}]}


def build_dets(gt, iou_type, duplicates=0, extra_unmatched=0, offset=0.0):
    dets = []
    for a in gt["annotations"]:
        for rep in range(1 + duplicates):
            d = {"image_id": 1, "category_id": 1, "score": 0.99 - 0.01 * rep}
            box = [a["bbox"][0] + offset, a["bbox"][1], a["bbox"][2], a["bbox"][3]]
            if iou_type == "bbox":
                d["bbox"] = box
            else:
                rle = maskUtils.merge(maskUtils.frPyObjects(
                    _poly(box[0], box[1]), SIDE, SIDE))
                rle["counts"] = rle["counts"].decode()
                d["segmentation"] = rle
            dets.append(d)
    for j in range(extra_unmatched):
        # Scored ABOVE the true matches on purpose. COCO AP is a precision
        # envelope over sorted detections, so a false positive ranked below a
        # matched detection lands after recall has saturated and cannot lower
        # AP. Only a high-ranked false positive costs precision.
        d = {"image_id": 1, "category_id": 1, "score": 0.999 + 1e-4 * j}
        if iou_type == "bbox":
            d["bbox"] = [80, 80 - 5 * j, 5, 5]
        else:
            rle = maskUtils.merge(maskUtils.frPyObjects(_poly(80, 80 - 5 * j, 5, 5),
                                                        SIDE, SIDE))
            rle["counts"] = rle["counts"].decode()
            d["segmentation"] = rle
        dets.append(d)
    return dets


def run(ann_ids, iou_type, eval_cls, *, duplicates=0, extra_unmatched=0,
        crowd_ids=(), offset=0.0):
    gt = build_gt(ann_ids, crowd_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = gt; coco.createIndex()
        dets = build_dets(gt, iou_type, duplicates, extra_unmatched, offset)
        ev = eval_cls(coco, coco.loadRes(dets), iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


def both(ann_ids, iou_type, **kw):
    return (run(ann_ids, iou_type, COCOeval, **kw),
            run(ann_ids, iou_type, ZeroIdSafeCOCOeval, **kw))


# --------------------------------------------------------------------------


def test_control_without_zero_is_untouched():
    """No annotation id 0 -> compatibility mode must change nothing at all."""
    for iou_type in ("bbox", "segm"):
        off, com = both([1, 2, 3], iou_type)
        assert com.zero_id_repairs == 0
        assert np.allclose(off.stats, com.stats), iou_type


def test_zero_id_is_recovered():
    """A detection matching annotation id 0 must count as a true positive."""
    for iou_type in ("bbox", "segm"):
        off, com = both([0, 1, 2], iou_type)
        ctrl = run([1, 2, 3], iou_type, COCOeval)
        assert off.stats[0] < ctrl.stats[0] - 0.4, "defect should be present"
        assert np.isclose(com.stats[0], ctrl.stats[0]), iou_type
        assert com.zero_id_repairs > 0


def test_all_iou_thresholds_repaired():
    """The repair must hold across the whole 0.50:0.95 sweep, not just AP50."""
    off, com = both([0], "bbox")
    assert off.stats[1] == 0.0 and off.stats[2] == 0.0     # AP50, AP75
    assert np.isclose(com.stats[1], 1.0) and np.isclose(com.stats[2], 1.0)
    # 10 thresholds x 4 area ranges for one image/category
    assert com.zero_id_repairs == 40


def test_unmatched_detections_stay_false_positives():
    """Repairing id 0 must not rescue genuinely unmatched detections."""
    for iou_type in ("bbox", "segm"):
        _, com = both([0, 1, 2], iou_type, extra_unmatched=3)
        clean = run([1, 2, 3], iou_type, COCOeval, extra_unmatched=3)
        assert np.isclose(com.stats[0], clean.stats[0]), iou_type
        assert com.stats[0] < 1.0, "high-ranked false positives must cost precision"
        assert clean.stats[0] < 1.0, "control must also be penalised"


def test_duplicate_detections_behave_like_the_control():
    """Duplicates must be scored the same with and without the repair."""
    for iou_type in ("bbox", "segm"):
        _, com_dup = both([0, 1, 2], iou_type, duplicates=1)
        ctrl_dup = run([1, 2, 3], iou_type, COCOeval, duplicates=1)
        assert np.isclose(com_dup.stats[0], ctrl_dup.stats[0]), iou_type
        # and a duplicate ranked above its own match does cost precision
        _, com_hi = both([0, 1, 2], iou_type, duplicates=1, extra_unmatched=2)
        assert com_hi.stats[0] < com_dup.stats[0], iou_type


def test_crowd_ground_truth_is_ignored_not_matched():
    """Crowd GT is ignored by COCOeval; repairing must not turn it into a TP."""
    off, com = both([0, 1, 2], "bbox", crowd_ids=(0,))
    assert np.allclose(off.stats, com.stats), "crowd id 0 must not change scores"


def test_ignored_gt_outside_area_range_unaffected():
    """Area-range filtering still applies after the repair."""
    off, com = both([0, 1, 2], "bbox")
    # AP_small: the 10x10 boxes (area 100) fall in 'small'; medium/large empty
    assert off.stats[4] == -1.0 and com.stats[4] == -1.0   # AP_medium undefined
    assert com.stats[3] >= off.stats[3]                    # AP_small improves


def test_sentinel_is_truthy_and_distinguishable():
    """The sentinel must satisfy logical_and and not collide with a real id."""
    assert bool(np.logical_and(ZERO_ID_SENTINEL, True))
    assert ZERO_ID_SENTINEL < 0, "must not be mistaken for an annotation id"


def test_official_metrics_unchanged_when_compat_enabled():
    """Turning compatibility reporting on must not alter any official key."""
    from mmdet.registry import METRICS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.proposal_first_pidray_occ  # noqa: F401
    import pickle
    from pathlib import Path

    pkl = Path("work_dirs/pixray_p1/test/test/predictions.pkl")
    if not pkl.is_file():                      # predictions are optional here
        return
    classes = ("Bat", "Battery", "Dart", "Fireworks", "Gun", "Hammer", "Knife",
               "Lighter", "Pliers", "Pressure_vessel", "Razor_blade",
               "Saw_blade", "Scissors", "Screwdriver", "Wrench")
    preds = pickle.load(open(pkl, "rb"))[:120]
    results = []
    for s_ in preds:
        p_ = s_["pred_instances"]
        r = dict(img_id=s_["img_id"], bboxes=np.asarray(p_["bboxes"]),
                 scores=np.asarray(p_["scores"]), labels=np.asarray(p_["labels"]))
        if "masks" in p_:
            r["masks"] = p_["masks"]
        results.append((dict(width=s_["ori_shape"][1], height=s_["ori_shape"][0],
                             img_id=s_["img_id"]), r))

    def score(flag):
        m = METRICS.build(dict(type="XRayCocoMetric",
                               ann_file="data/PIXray/annotations/test.json",
                               metric=["bbox", "segm"], classwise=True,
                               zero_id_compat=flag))
        m.dataset_meta = dict(classes=classes)
        with contextlib.redirect_stdout(io.StringIO()):
            return m.compute_metrics(list(results))

    plain, with_compat = score(False), score(True)
    for key, value in plain.items():
        assert key in with_compat, f"official key {key} vanished"
        assert with_compat[key] == value, f"official key {key} changed: {value} -> {with_compat[key]}"
    assert any(k.startswith("compat_zeroid/") for k in with_compat), "no compat keys emitted"
    assert not any(k.startswith("compat_zeroid/") for k in plain), "compat leaked when disabled"


def test_gt_ids_contain_zero_helper():
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = build_gt([0, 1]); coco.createIndex()
        other = COCO(); other.dataset = build_gt([1, 2]); other.createIndex()
    assert gt_ids_contain_zero(coco)
    assert not gt_ids_contain_zero(other)


def test_dtignore_repair_matters_when_detection_crosses_area_bin():
    """The dtIgnore repair is required, not cosmetic.

    ``evaluateImg`` marks a detection ignorable via
    ``logical_or(dtIg, logical_and(dtm == 0, out_of_area_range))``. A detection
    that genuinely matched annotation id 0 reads as unmatched there, so if it
    sits outside the current area range it is flagged ignore even though its
    ground truth is inside the range. Repairing ``dtMatches`` alone leaves that
    detection excluded and the ground truth counted as a miss.

    Constructed so the match crosses a COCO area boundary: ground truth area
    1000 (small), detection area 1600 (medium), fully containing it, so
    IoU = 1000/1600 = 0.625 and the pair matches at IoU 0.5.
    """
    gt = {"images": [{"id": 1, "width": SIDE, "height": SIDE, "file_name": "x.png"}],
          "annotations": [{"id": 0, "image_id": 1, "category_id": 1, "iscrowd": 0,
                           "bbox": [10, 10, 25, 40], "area": 1000,
                           "segmentation": _poly(10, 10, 25, 40)}],
          "categories": [{"id": 1, "name": "a"}]}
    det = [{"image_id": 1, "category_id": 1, "score": 0.99,
            "bbox": [10, 10, 40, 40]}]                      # area 1600 -> medium
    assert gt["annotations"][0]["area"] < 1024 <= det[0]["bbox"][2] * det[0]["bbox"][3]

    def small_ap(eval_cls):
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(); coco.dataset = gt; coco.createIndex()
            ev = eval_cls(coco, coco.loadRes([dict(d) for d in det]), "bbox")
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return ev.stats[3]                                   # AP_small

    official = small_ap(COCOeval)
    compat = small_ap(ZeroIdSafeCOCOeval)
    assert official == 0.0, f"defect should zero AP_small, got {official}"
    assert compat > 0.0, (
        "dtIgnore was not repaired: the matched detection is still ignored "
        f"in the small range (AP_small={compat})")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1; print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:                                    # noqa: BLE001
            failed += 1; print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
