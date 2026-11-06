"""Dependency-free, fail-closed checks for the additive PIDray analysis tools."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

# Absolute tolerance on AP in [0, 1], not percentage points. These checks compare
# the same predictions/annotations, not independent training runs.
AP_TOLERANCE = 1e-10
IOU_TYPES = ("segm", "bbox")
PIDRAY_VALIDATION_CASES = (("03", "hard"), ("09M", "hard"), ("09M", "hidden"))


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tp_count(value) -> bool:
    # The frozen scorer serializes its numpy accumulator as a JSON float.
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0 and float(value).is_integer())


def coverage_errors(coverage) -> list[str]:
    """Require every expected image exactly once, including empty detections."""
    if not isinstance(coverage, Mapping):
        return ["coverage: missing or invalid record"]
    errors = []
    if coverage.get("images_match") is not True:
        errors.append("coverage.images_match must be true")
    keys = ("prediction_images", "unique_images", "test_images")
    counts = [coverage.get(key) for key in keys]
    if not all(_count(value) for value in counts):
        errors.append("coverage image counts must be nonnegative integers")
    elif not (counts[0] == counts[1] == counts[2]):
        errors.append("coverage requires prediction_images == unique_images == test_images")
    return errors


def diagnostic_errors(splits, expected_splits=("easy", "hard", "hidden")) -> list[str]:
    """A diagnostic passes only when all expected splits pass both checks."""
    if not isinstance(splits, Mapping):
        return ["splits: missing or invalid record"]
    errors = []
    if set(splits) != set(expected_splits):
        errors.append("splits do not match the expected test splits")
    for split in expected_splits:
        record = splits.get(split)
        if not isinstance(record, Mapping):
            errors.append(f"{split}: missing or invalid record")
            continue
        errors.extend(f"{split}.{error}" for error in coverage_errors(record.get("coverage")))
        checks = record.get("check", {})
        for iou_type in IOU_TYPES:
            check = checks.get(iou_type) if isinstance(checks, Mapping) else None
            if not isinstance(check, Mapping) or check.get("matches") is not True:
                errors.append(f"{split}.{iou_type}: scorer/MMDetection agreement failed")
    return errors


def _difference_errors(record, field: str, prefix: str, tolerance: float) -> list[str]:
    group = record.get(field)
    errors = []
    for iou_type in IOU_TYPES:
        value = group.get(iou_type) if isinstance(group, Mapping) else None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= tolerance):
            errors.append(f"{prefix}.{field}.{iou_type}: expected finite difference <= {tolerance:g}, got {value!r}")
    return errors


def mapped_analysis_errors(report, tolerance: float = AP_TOLERANCE) -> list[str]:
    """Validate every required regression flag, AP check, and PIDray TP count."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("AP tolerance must be finite and nonnegative")
    if not isinstance(report, Mapping):
        return ["report: missing or invalid record"]
    errors = []
    pixray = report.get("pixray_regression")
    if not isinstance(pixray, Mapping):
        errors.append("pixray_regression: missing or invalid record")
    else:
        for name in ("mapping_is_identity", "errors_identical", "mask_quality_identical",
                     "pr_curves_identical", "per_gt_identical"):
            if pixray.get(name) is not True:
                errors.append(f"pixray_regression.{name} must be true")
        for field in ("ap_cache_matches_scorer", "ap_cache_self_check"):
            errors.extend(_difference_errors(pixray, field, "pixray_regression", tolerance))
    rows = report.get("pidray")
    if not isinstance(rows, list):
        return errors + ["pidray: missing or invalid records"]
    cases = []
    for index, row in enumerate(rows):
        prefix = f"pidray[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}: invalid record")
            continue
        cases.append((row.get("method"), row.get("split")))
        for field in ("ap_cache_vs_rescored", "ap_cache_self_check"):
            errors.extend(_difference_errors(row, field, prefix, tolerance))
        actual, expected = row.get("tp_mapped_analysis"), row.get("tp_corrected_scorer")
        if not (_tp_count(actual) and _tp_count(expected) and actual == expected):
            errors.append(f"{prefix}: TP counts must be nonnegative integers and equal ({actual!r} vs {expected!r})")
    if (len(cases) != len(PIDRAY_VALIDATION_CASES)
            or any(cases.count(case) != 1 for case in PIDRAY_VALIDATION_CASES)):
        errors.append("pidray: expected each required method/split case exactly once")
    return errors


def write_new_report(path: Path, report) -> None:
    """Create a report exclusively; never overwrite an existing historical report."""
    text = json.dumps(report, indent=1, default=float) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
