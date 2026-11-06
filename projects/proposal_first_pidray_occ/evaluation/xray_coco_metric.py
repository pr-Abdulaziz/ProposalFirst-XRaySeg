"""COCO metric that keeps per-metric detail the upstream implementation drops.

``CocoMetric`` loses two things the primary comparison table needs.

Per-category AP is written under ``f'{name}_precision'`` with no metric prefix
(``mmdet/evaluation/metrics/coco_metric.py``).  With ``metric=['bbox', 'segm']``
the segmentation pass therefore overwrites the detection pass and only one
survives, silently: the surviving fifteen values average to ``segm_mAP``, so a
reader has no way to tell the columns are not detection AP.

Average recall is reported only for the ``proposal`` metric, which scores
boxes.  Mask recall is never emitted even though ``COCOeval`` has already
computed it, which leaves a mask-quality comparison with no recall column.

Both are additive fixes.  Every key produced by ``CocoMetric`` keeps its
existing name and value so runs scored before this metric existed stay
comparable; the recovered numbers arrive under ``{iou_type}_`` prefixes.
"""

import itertools
import os.path as osp
import tempfile
from typing import Optional, Sequence

import numpy as np
from mmengine.fileio import load
from mmengine.logging import MMLogger
from pycocotools.cocoeval import COCOeval

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

from .zero_id_compat import ZeroIdSafeCOCOeval, gt_ids_contain_zero

# Index of each summary statistic in ``COCOeval.stats``.  Recall entries are
# named for ``proposal_nums`` because ``CocoMetric`` assigns
# ``params.maxDets = list(self.proposal_nums)`` for every iou type, not only
# for the proposal metric.
_RECALL_STATS = {
    "AR@100": 6,
    "AR@300": 7,
    "AR@1000": 8,
    "AR_s@1000": 9,
    "AR_m@1000": 10,
    "AR_l@1000": 11,
}


@METRICS.register_module()
class XRayCocoMetric(CocoMetric):
    """``CocoMetric`` with per-metric classwise AP and recall.

    Args:
        classwise_iou_types: Iou types to recover per-category AP for.  Only
            entries also present in ``metric`` are evaluated.  Defaults to
            ``('bbox', 'segm')``.
    """

    def __init__(
        self,
        *args,
        classwise_iou_types: Sequence[str] = ("bbox", "segm"),
        zero_id_compat: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.classwise_iou_types = tuple(classwise_iou_types)
        # Official metrics are always produced. When this is on, a second,
        # separately labelled evaluation is added under the "compat_zeroid/"
        # prefix; it never replaces an official value. See zero_id_compat.
        self.zero_id_compat = zero_id_compat

    def compute_metrics(self, results: list) -> dict:
        eval_results = super().compute_metrics(results)
        if self.format_only:
            return eval_results
        eval_results.update(self._detailed_metrics(results))
        if self.zero_id_compat:
            eval_results.update(self._zero_id_compat_metrics(results))
        return eval_results

    def _zero_id_compat_metrics(self, results: list) -> dict:
        """Rescore with the zero-id-safe evaluator, under its own prefix."""
        logger: MMLogger = MMLogger.get_current_instance()
        if not gt_ids_contain_zero(self._coco_api):
            logger.info("zero_id_compat: ground truth has no annotation id 0; "
                        "compatibility metrics are identical by construction")
        iou_types = [m for m in self.metrics if m in ("bbox", "segm")]
        if not iou_types:
            return {}
        _, preds = zip(*results)
        out = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = self.results2json(preds, osp.join(tmp_dir, "compat"))
            for iou_type in iou_types:
                coco_eval = self._build_eval(
                    files, iou_type, logger, eval_cls=ZeroIdSafeCOCOeval
                )
                if coco_eval is None:
                    continue
                for name, index in (("mAP", 0), ("mAP_50", 1), ("mAP_75", 2),
                                    ("mAP_s", 3), ("mAP_m", 4), ("mAP_l", 5)):
                    value = coco_eval.stats[index]
                    out[f"compat_zeroid/{iou_type}_{name}"] = float(f"{round(value, 3)}")
                for name, index in _RECALL_STATS.items():
                    value = coco_eval.stats[index]
                    out[f"compat_zeroid/{iou_type}_{name}"] = float(f"{round(value, 3)}")
                out[f"compat_zeroid/{iou_type}_repairs"] = coco_eval.zero_id_repairs
        return out

    def _detailed_metrics(self, results: list) -> dict:
        """Recover the per-iou-type numbers ``CocoMetric`` collapses or omits.

        This repeats the ``COCOeval`` pass rather than reimplementing the
        upstream loop.  The cost is a few seconds against an evaluation that
        already spends minutes on inference, and it buys an exact match with
        the values ``CocoMetric`` reports for the surviving iou type.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        iou_types = [
            metric
            for metric in self.metrics
            if metric in self.classwise_iou_types and metric in ("bbox", "segm")
        ]
        if not iou_types:
            return {}

        _, preds = zip(*results)
        detailed = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_files = self.results2json(preds, osp.join(tmp_dir, "detail"))
            for iou_type in iou_types:
                coco_eval = self._build_eval(result_files, iou_type, logger)
                if coco_eval is None:
                    continue
                for name, index in _RECALL_STATS.items():
                    value = coco_eval.stats[index]
                    detailed[f"{iou_type}_{name}"] = float(f"{round(value, 3)}")
                if self.classwise:
                    detailed.update(self._classwise_ap(coco_eval, iou_type))
        return detailed

    def _build_eval(
        self, result_files: dict, iou_type: str, logger: MMLogger, eval_cls=None
    ) -> Optional[COCOeval]:
        """Score ``iou_type`` with exactly the parameters ``CocoMetric`` uses."""
        predictions = load(result_files[iou_type])
        if iou_type == "segm":
            # cocoapi derives instance area from ``bbox`` when it is present,
            # which changes the small/medium/large split for mask scoring.
            # CocoMetric drops the key for the same reason.
            for prediction in predictions:
                prediction.pop("bbox", None)
        try:
            coco_dt = self._coco_api.loadRes(predictions)
        except IndexError:
            logger.error("The testing results of the whole dataset is empty.")
            return None

        coco_eval = (eval_cls or COCOeval)(self._coco_api, coco_dt, iou_type)
        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval

    def _classwise_ap(self, coco_eval: COCOeval, iou_type: str) -> dict:
        """Per-category AP under a key that names the iou type it came from."""
        # precision axes: (iou, recall, category, area range, max dets)
        precisions = coco_eval.eval["precision"]
        assert len(self.cat_ids) == precisions.shape[2]

        classwise = {}
        for index, cat_id in enumerate(self.cat_ids):
            name = self._coco_api.loadCats(cat_id)[0]["name"]
            # area range 0 is "all", max dets -1 is the largest budget
            precision = precisions[:, :, index, 0, -1]
            precision = precision[precision > -1]
            average = np.mean(precision) if precision.size else float("nan")
            classwise[f"{iou_type}_{name}_precision"] = round(float(average), 3)
        return classwise
