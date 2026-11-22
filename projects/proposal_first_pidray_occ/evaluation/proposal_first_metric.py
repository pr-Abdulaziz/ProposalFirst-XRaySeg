"""Runtime and screening metrics for proposal-first inference."""

from pathlib import Path
from typing import ClassVar

import numpy as np
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class ProposalFirstRuntimeMetric(BaseMetric):
    """Measure image screening and optional model-only inference efficiency.

    ``final_*`` uses the PIDray paper's image decision: an image is positive
    when at least one final detection has confidence greater than 0.5.
    ``r0_*`` evaluates the proposed coarse node independently.
    """

    default_prefix: ClassVar[str] = "proposal_first"

    def __init__(
        self,
        r0_threshold: float = 0.5,
        detection_threshold: float = 0.5,
        timing_warmup: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if timing_warmup < 0:
            raise ValueError("timing_warmup must be non-negative.")
        self.r0_threshold = r0_threshold
        self.detection_threshold = detection_threshold
        self.timing_warmup = timing_warmup

    @staticmethod
    def _difficulty(img_path: str) -> str:
        basename = Path(img_path).name.lower()
        for difficulty in ("easy", "hard", "hidden"):
            if basename.startswith(f"xray_{difficulty}"):
                return difficulty
        return "unknown"

    def process(self, data_batch: dict, data_samples: list) -> None:
        for sample in data_samples:
            # MMEngine converts BaseDataElement outputs to plain dictionaries
            # before passing them to metrics. Keep direct calls with
            # DetDataSample supported as well for focused tests.
            if isinstance(sample, dict):
                gt_instances = sample.get("gt_instances", {})
                pred_instances = sample.get("pred_instances", {})
                metainfo = sample.get("metainfo", sample)
                scores = pred_instances.get("scores")
            else:
                gt_instances = sample.gt_instances
                pred_instances = sample.pred_instances
                metainfo = sample.metainfo
                scores = getattr(pred_instances, "scores", None)

            def instance_count(instances) -> int:
                if isinstance(instances, dict):
                    for field in ("labels", "bboxes", "scores", "masks"):
                        if field in instances:
                            return len(instances[field])
                    return 0
                return len(instances)

            gt_count = instance_count(gt_instances)
            pred_count = instance_count(pred_instances)
            # Absent instrumentation is recorded as None, not as a measured
            # value. Defaulting r0_score to 0.0 and sam_executed to False made
            # a model with no R0 module report r0_recall 0.0, and a model whose
            # SAM encoder runs on every image report sam_execution_rate 0.0 --
            # both read as measurements rather than as "not instrumented".
            r0_score = metainfo.get("r0_score")
            r0_score = None if r0_score is None else float(r0_score)
            sam_executed = metainfo.get("sam_executed")
            screening_route = metainfo.get("screening_route", "unknown")
            final_positive = bool(
                scores is not None
                and len(scores) > 0
                and (scores > self.detection_threshold).any().item()
            )
            self.results.append(
                {
                    "has_prohibited": gt_count > 0,
                    "r0_positive": (None if r0_score is None
                                    else r0_score > self.r0_threshold),
                    "final_positive": final_positive,
                    "sam_executed": (None if sam_executed is None
                                     else bool(sam_executed)),
                    "screening_route": screening_route,
                    "rescue_executed": bool(metainfo.get("rescue_executed", False)),
                    "detections_before_screening": int(
                        metainfo.get("detections_before_screening", pred_count)
                    ),
                    "detections_after_screening": int(
                        metainfo.get("detections_after_screening", pred_count)
                    ),
                    "difficulty": self._difficulty(str(metainfo.get("img_path", ""))),
                    "stage_times_ms": metainfo.get("stage_times_ms", {}),
                    "peak_gpu_memory_mb": metainfo.get("peak_gpu_memory_mb"),
                    "conditional_hard_count": int(
                        metainfo.get("conditional_hard_count", 0)
                    ),
                    "conditional_total_count": int(
                        metainfo.get("conditional_total_count", 0)
                    ),
                }
            )

    @staticmethod
    def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
        true_positive = int(np.logical_and(truth, predicted).sum())
        true_negative = int(np.logical_and(~truth, ~predicted).sum())
        false_positive = int(np.logical_and(~truth, predicted).sum())
        false_negative = int(np.logical_and(truth, ~predicted).sum())

        def safe(numerator: int, denominator: int) -> float:
            # An undefined ratio is None, not zero. PIXray has no
            # prohibited-item-free images, so specificity and false-alarm rate
            # have an empty denominator and are genuinely unmeasurable here.
            return float(numerator / denominator) if denominator else None

        precision = safe(true_positive, true_positive + false_positive)
        recall = safe(true_positive, true_positive + false_negative)
        return {
            "accuracy": safe(true_positive + true_negative, len(truth)),
            "error_rate": safe(false_positive + false_negative, len(truth)),
            "recall": recall,
            "precision": precision,
            "specificity": safe(true_negative, true_negative + false_positive),
            "false_alarm_rate": safe(false_positive, false_positive + true_negative),
            "false_negative_rate": safe(false_negative, false_negative + true_positive),
            # Undefined when either component is undefined.
            "f1": (None if precision is None or recall is None
                   else safe(2 * precision * recall, precision + recall)),
        }

    def compute_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {}
        truth = np.asarray([row["has_prohibited"] for row in results], dtype=bool)
        # A field is *instrumented* only if at least one sample reported it.
        # Otherwise it is unavailable and every metric derived from it is
        # reported as None rather than as a measured zero.
        r0_instrumented = any(row["r0_positive"] is not None for row in results)
        sam_instrumented = any(row["sam_executed"] is not None for row in results)
        r0_predicted = np.asarray(
            [bool(row["r0_positive"]) for row in results], dtype=bool
        )
        final_predicted = np.asarray(
            [row["final_positive"] for row in results], dtype=bool
        )
        executed = np.asarray(
            [bool(row["sam_executed"]) for row in results], dtype=float
        )
        rescue_executed = np.asarray(
            [row["rescue_executed"] for row in results], dtype=bool
        )
        routes = np.asarray([row["screening_route"] for row in results], dtype=object)
        difficulties = np.asarray([row["difficulty"] for row in results], dtype=object)
        detections_before = np.asarray(
            [row["detections_before_screening"] for row in results], dtype=float
        )
        detections_after = np.asarray(
            [row["detections_after_screening"] for row in results], dtype=float
        )
        conditional_hard = sum(row["conditional_hard_count"] for row in results)
        conditional_total = sum(row["conditional_total_count"] for row in results)
        def rate(values, mask=None):
            """Mean over a subset, or None when the subset or field is empty."""
            if not sam_instrumented:
                return None
            subset = values if mask is None else values[mask]
            return float(subset.mean()) if subset.size else None

        metrics = {
            "sam_execution_rate": rate(executed),
            "sam_execution_rate_normal": rate(executed, ~truth),
            "sam_execution_rate_prohibited": rate(executed, truth),
            "rescue_execution_rate": float(rescue_executed.mean()),
            "detections_before_screening_mean": float(detections_before.mean()),
            "detections_after_screening_mean": float(detections_after.mean()),
            "retained_proposals_per_image": float(detections_after.mean()),
            "conditional_hard_route_rate": (
                float(conditional_hard / conditional_total)
                if conditional_total
                else 0.0
            ),
        }
        for route in ("normal", "rescue", "rejected"):
            route_mask = routes == route
            metrics[f"route_{route}_rate"] = float(route_mask.mean())
            metrics[f"detections_after_screening_{route}_mean"] = (
                float(detections_after[route_mask].mean()) if route_mask.any() else 0.0
            )

        for difficulty in ("easy", "hard", "hidden"):
            difficulty_mask = difficulties == difficulty
            positive_mask = np.logical_and(difficulty_mask, truth)
            false_negatives = np.logical_and(positive_mask, ~r0_predicted)
            metrics[f"r0_false_negative_rate_{difficulty}"] = (
                float(false_negatives.sum() / positive_mask.sum())
                if positive_mask.any()
                else 0.0
            )
            metrics[f"rescue_rate_{difficulty}"] = (
                float(rescue_executed[difficulty_mask].mean())
                if difficulty_mask.any()
                else 0.0
            )
        metrics.update(
            {
                f"r0_{name}": value
                for name, value in (
                    self._binary_metrics(truth, r0_predicted)
                    if r0_instrumented
                    # No R0 module reported a score, so every R0 statistic is
                    # unavailable rather than zero.
                    else dict.fromkeys(
                        self._binary_metrics(truth, r0_predicted), None)
                ).items()
            }
        )
        metrics.update(
            {
                f"final_image_{name}": value
                for name, value in self._binary_metrics(truth, final_predicted).items()
            }
        )

        timing_results = results[self.timing_warmup :]
        timing_names = sorted(
            {name for row in timing_results for name in row["stage_times_ms"]}
        )
        for name in timing_names:
            values = np.asarray(
                [
                    row["stage_times_ms"][name]
                    for row in timing_results
                    if name in row["stage_times_ms"]
                ],
                dtype=float,
            )
            metrics[f"{name}_ms_mean"] = float(values.mean())
            metrics[f"{name}_ms_median"] = float(np.median(values))
            metrics[f"{name}_ms_p95"] = float(np.percentile(values, 95))

        total_times = np.asarray(
            [
                row["stage_times_ms"]["model_total"]
                for row in timing_results
                if "model_total" in row["stage_times_ms"]
            ],
            dtype=float,
        )
        if total_times.size:
            metrics["model_throughput_images_per_second"] = float(
                1000.0 / total_times.mean()
            )
        for route in ("normal", "rescue"):
            route_times = np.asarray(
                [
                    row["stage_times_ms"]["model_total"]
                    for row in timing_results
                    if row["screening_route"] == route
                    and "model_total" in row["stage_times_ms"]
                ],
                dtype=float,
            )
            if route_times.size:
                metrics[f"model_total_ms_{route}_mean"] = float(route_times.mean())

        memory = np.asarray(
            [
                row["peak_gpu_memory_mb"]
                for row in timing_results
                if row["peak_gpu_memory_mb"] is not None
            ],
            dtype=float,
        )
        if memory.size:
            metrics["peak_gpu_memory_mb_mean"] = float(memory.mean())
            metrics["peak_gpu_memory_mb_max"] = float(memory.max())
        return metrics
