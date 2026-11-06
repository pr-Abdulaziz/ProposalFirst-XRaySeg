"""Evaluator compatibility mode for ground-truth annotation id 0.

``COCOeval`` stores the matched ground-truth id in ``dtMatches`` and later tests
it for truthiness::

    tps = np.logical_and(               dtm,  np.logical_not(dtIg))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

Zero therefore serves two purposes at once: "no match", and "matched the
annotation whose id is 0". A detection that correctly matches annotation id 0 is
counted as a false positive, and that ground-truth instance as a miss.

PIXray releases annotation id 0 in both splits, so one instance per split is
affected. The released data is never modified: this repairs only the evaluator's
internal matching array, after matching has already happened, by replacing the
ambiguous 0 with a sentinel that is distinguishable and truthy. Every annotation
id, image id, category id, box, mask and ``area`` is left exactly as published.

This is a *separately labelled* mode. Official metrics are always computed and
reported; compatibility metrics appear alongside them under their own prefix so
the two are never conflated or silently substituted.
"""

from __future__ import annotations

import numpy as np
from pycocotools.cocoeval import COCOeval

# Truthy, and distinct from any real annotation id, so a downstream reader can
# tell a repaired match from an ordinary one.
ZERO_ID_SENTINEL = -1.0


class ZeroIdSafeCOCOeval(COCOeval):
    """``COCOeval`` that does not treat ground-truth id 0 as "unmatched".

    Matching itself is untouched -- this runs after ``evaluateImg`` and only
    rewrites entries that the parent already decided were matches.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.zero_id_repairs = 0

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        result = super().evaluateImg(imgId, catId, aRng, maxDet)
        if result is None:
            return result
        gt_ids = list(result["gtIds"])
        if 0 not in gt_ids:
            return result
        dt_ids = list(result["dtIds"])
        gt_matches = result["gtMatches"]
        dt_matches = result["dtMatches"]
        dt_ignore = result["dtIgnore"]
        gt_ignore = result["gtIgnore"]
        zero_row = gt_ids.index(0)
        for thr in range(gt_matches.shape[0]):
            matched_det_id = gt_matches[thr, zero_row]
            if matched_det_id <= 0:
                continue  # annotation id 0 was not matched at this threshold
            try:
                col = dt_ids.index(int(matched_det_id))
            except ValueError:
                continue
            if dt_matches[thr, col] != 0:
                continue
            dt_matches[thr, col] = ZERO_ID_SENTINEL
            # The parent also consumed the ambiguous zero when deciding what to
            # ignore:
            #     dtIg = logical_or(dtIg, logical_and(dtm == 0, out_of_area))
            # A detection that genuinely matched annotation id 0 read as
            # unmatched there, so an out-of-area-range detection was flagged
            # ignore when it should have inherited the ground truth's own
            # ignore state. Repairing dtMatches alone leaves that wrong, which
            # distorts the size-stratified metrics far more than the aggregate
            # ones. Restore the value the matching loop would have assigned.
            dt_ignore[thr, col] = gt_ignore[zero_row]
            self.zero_id_repairs += 1
        return result


def gt_ids_contain_zero(coco_api) -> bool:
    """Whether the loaded ground truth uses annotation id 0 at all."""
    try:
        return 0 in coco_api.anns
    except AttributeError:
        return False
