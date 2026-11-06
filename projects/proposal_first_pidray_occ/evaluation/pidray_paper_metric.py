"""PIDray/Occ paper-compatible COCO evaluation.

The PIDray paper uses max detections ``(1, 10, 100)`` and reports AP, AP50,
AP75, APS, AR1, AR10, AR100, and ARS for both boxes and masks.  MMDetection's
default ``CocoMetric`` uses ``(100, 300, 1000)`` and does not persist AR for
the bbox/segm tasks, so a plain config is not paper-compatible.
"""

import copy
import tempfile
from pathlib import PurePosixPath
from typing import ClassVar

from mmengine.fileio import load
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class PIDrayPaperMetric(CocoMetric):
    """Evaluate the metrics printed in the PIDray and Occ papers.

    Difficulty subsets are obtained in either of two ways:

    1. ``subset_ann_files`` supplies exact COCO annotation files.
    2. The combined annotation file stores ``easy``, ``hard``, or
       ``hidden`` in an image metadata field or file-name path component.

    Exact annotation files take precedence and are required when the paper's
    original split membership cannot be inferred from the combined file.
    """

    default_prefix: ClassVar[str] = "coco"
    _PAPER_ITEMS: ClassVar[tuple[str, ...]] = (
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_s",
        "AR@100",
        "AR@300",
        "AR@1000",
        "AR_s@1000",
    )

    def __init__(
        self,
        *args,
        metric=("bbox", "segm"),
        subset_ann_files: dict[str, str] | None = None,
        infer_difficulty_splits: bool = True,
        **kwargs,
    ) -> None:
        kwargs.pop("proposal_nums", None)
        kwargs.pop("metric_items", None)
        super().__init__(
            *args,
            metric=list(metric),
            proposal_nums=(1, 10, 100),
            metric_items=self._PAPER_ITEMS,
            **kwargs,
        )
        self.subset_ann_files = subset_ann_files or {}
        self.infer_difficulty_splits = infer_difficulty_splits

    @staticmethod
    def _paper_aliases(results: dict) -> dict:
        """Add paper notation and correct MMDetection's fixed AR labels."""
        for task in ("bbox", "segm"):
            aliases = {
                "AP": f"{task}_mAP",
                "AP50": f"{task}_mAP_50",
                "AP75": f"{task}_mAP_75",
                "APS": f"{task}_mAP_s",
            }
            for paper_name, source_name in aliases.items():
                if source_name in results:
                    results[f"{task}_{paper_name}"] = results[source_name]

            # CocoMetric's keys are hard-coded for maxDets=(100,300,1000).
            # Here their values correspond to our configured (1,10,100).
            ar_aliases = {
                "AR1": f"{task}_AR@100",
                "AR10": f"{task}_AR@300",
                "AR100": f"{task}_AR@1000",
                "ARS": f"{task}_AR_s@1000",
            }
            for paper_name, source_name in ar_aliases.items():
                if source_name in results:
                    results[f"{task}_{paper_name}"] = results.pop(source_name)
        return results

    @staticmethod
    def _normalize_split(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "easy": "easy",
            "hard": "hard",
            "hide": "hidden",
            "hidden": "hidden",
            "occ": "occlusion",
            "occluded": "occlusion",
            "occlusion": "occlusion",
            "non_occ": "non_occlusion",
            "non_occluded": "non_occlusion",
            "non_occlusion": "non_occlusion",
            "validation": "non_occlusion",
            "val": "non_occlusion",
        }
        return aliases.get(normalized)

    def _image_split(self, image: dict) -> str | None:
        for field in ("difficulty", "subset", "split", "mode"):
            split = self._normalize_split(image.get(field))
            if split is not None:
                return split

        file_name = str(image.get("file_name", "")).replace("\\", "/")
        for component in PurePosixPath(file_name).parts:
            split = self._normalize_split(component)
            if split in {"easy", "hard", "hidden"}:
                return split
        basename = PurePosixPath(file_name).name.lower()
        for split in ("easy", "hard", "hidden"):
            if basename.startswith(f"xray_{split}"):
                return split
        return None

    @staticmethod
    def _subset_dataset(dataset: dict, image_ids: set[int]) -> dict:
        return {
            "info": copy.deepcopy(dataset.get("info", {})),
            "licenses": copy.deepcopy(dataset.get("licenses", [])),
            "images": [
                copy.deepcopy(image)
                for image in dataset.get("images", [])
                if image["id"] in image_ids
            ],
            "annotations": [
                copy.deepcopy(annotation)
                for annotation in dataset.get("annotations", [])
                if annotation["image_id"] in image_ids
            ],
            "categories": copy.deepcopy(dataset.get("categories", [])),
        }

    def _difficulty_datasets(self) -> dict[str, dict]:
        if self.subset_ann_files:
            if self._coco_api is None:
                return {}
            combined = self._coco_api.dataset
            combined_ids = {
                image["file_name"]: image["id"]
                for image in combined.get("images", [])
            }
            remapped = {}
            for name, path in self.subset_ann_files.items():
                subset = load(path)
                old_to_new = {
                    image["id"]: combined_ids[image["file_name"]]
                    for image in subset.get("images", [])
                    if image["file_name"] in combined_ids
                }
                subset["images"] = [
                    {**image, "id": old_to_new[image["id"]]}
                    for image in subset.get("images", [])
                    if image["id"] in old_to_new
                ]
                subset["annotations"] = [
                    {**annotation, "image_id": old_to_new[annotation["image_id"]]}
                    for annotation in subset.get("annotations", [])
                    if annotation["image_id"] in old_to_new
                ]
                remapped[self._normalize_split(name) or name] = subset
            return remapped
        if not self.infer_difficulty_splits or self._coco_api is None:
            return {}

        dataset = self._coco_api.dataset
        groups: dict[str, set[int]] = {
            "easy": set(),
            "hard": set(),
            "hidden": set(),
        }
        for image in dataset.get("images", []):
            split = self._image_split(image)
            if split in groups:
                groups[split].add(image["id"])
        return {
            name: self._subset_dataset(dataset, image_ids)
            for name, image_ids in groups.items()
            if image_ids
        }

    def _evaluate_subset(
        self,
        dataset: dict,
        predictions_by_task: dict[str, list[dict]],
    ) -> dict[str, float]:
        image_ids = {image["id"] for image in dataset.get("images", [])}
        if not image_ids or not dataset.get("annotations"):
            return {}

        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        category_ids = [category["id"] for category in dataset["categories"]]
        output = {}

        for task in self.metrics:
            predictions = [
                copy.deepcopy(prediction)
                for prediction in predictions_by_task[task]
                if prediction["image_id"] in image_ids
            ]
            if task == "segm":
                for prediction in predictions:
                    prediction.pop("bbox", None)
            if not predictions:
                output.update(
                    {
                        f"{task}_AP": 0.0,
                        f"{task}_AP50": 0.0,
                        f"{task}_AP75": 0.0,
                    }
                )
                continue

            coco_dt = coco_gt.loadRes(predictions)
            evaluator = COCOeval(coco_gt, coco_dt, task)
            evaluator.params.catIds = category_ids
            evaluator.params.imgIds = sorted(image_ids)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.params.iouThrs = self.iou_thrs
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            output.update(
                {
                    f"{task}_AP": float(f"{evaluator.stats[0]:.3f}"),
                    f"{task}_AP50": float(f"{evaluator.stats[1]:.3f}"),
                    f"{task}_AP75": float(f"{evaluator.stats[2]:.3f}"),
                }
            )
        return output

    def compute_metrics(self, results: list[tuple]) -> dict:
        overall = self._paper_aliases(super().compute_metrics(results))
        subsets = self._difficulty_datasets()
        if not subsets:
            MMLogger.get_current_instance().warning(
                "No PIDray difficulty split metadata was found. Overall paper "
                "metrics are valid, but Easy/Hard/Hidden AP requires exact "
                "split annotation files or split names in image metadata."
            )
            return overall

        _, predictions = zip(*results)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_files = self.results2json(
                predictions, f"{tmp_dir}/pidray_paper_results"
            )
            predictions_by_task = {
                task: load(result_files[task]) for task in self.metrics
            }

            for subset_name, dataset in subsets.items():
                metrics = self._evaluate_subset(dataset, predictions_by_task)
                for metric_name, value in metrics.items():
                    overall[f"{metric_name}_{subset_name}"] = value

        return overall
