"""Small, dependency-light experiment reporting helpers.

The model and evaluators return ordinary metric dictionaries. This module keeps
report formatting outside the training code: it stores those raw dictionaries
as JSON and appends them to a single Excel workbook after evaluation.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHEET_HEADERS = {
    "runs": [
        "run_id", "timestamp_utc", "dataset", "split", "config",
        "checkpoint", "seed", "status",
    ],
    "metrics_raw": [
        "run_id", "dataset", "split", "family", "metric", "value",
        "unit", "source",
    ],
    "detection_summary": ["run_id", "dataset", "split", "metric", "value"],
    "segmentation_summary": ["run_id", "dataset", "split", "metric", "value"],
    "runtime_efficiency": ["run_id", "dataset", "split", "metric", "value", "unit"],
    "per_class": ["run_id", "dataset", "split", "metric", "value"],
    "ablations": ["run_id", "dataset", "config", "seed"],
    "environment": ["run_id", "name", "value"],
    "notes": ["run_id", "note"],
}


def _as_cell(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _as_cell(value.item())
    if hasattr(value, "tolist"):
        return _as_cell(value.tolist())
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _metric_family(name: str) -> str:
    normalized = name.lower()
    # An explicit segmentation token settles the family first. Mask average
    # recall is named ``segm_AR@100``, and testing for ``ar@`` before that
    # would file it under detection.
    segmentation = any(
        token in normalized
        for token in ("segm", "mask", "dice", "iou", "boundary")
    )
    if not segmentation and (
        "bbox" in normalized
        or "proposal_recall" in normalized
        or "ar@" in normalized
    ):
        return "detection"
    if segmentation:
        return "segmentation"
    if any(
        token in normalized
        for token in (
            "_ms", "throughput", "memory", "proposal", "sam_execution",
            "training_seconds", "gpu_hours", "parameters",
        )
    ):
        return "runtime"
    return "other"


def _metric_unit(name: str) -> str:
    normalized = name.lower()
    if "_ms" in normalized:
        return "ms"
    if "memory" in normalized:
        return "MB"
    if "throughput" in normalized:
        return "images/s"
    if "training_seconds" in normalized:
        return "s"
    if "gpu_hours" in normalized:
        return "GPU-h"
    if "parameters" in normalized:
        return "count"
    ratios = ("ap", "ar", "iou", "dice", "rate", "accuracy", "recall", "precision", "f1")
    return "ratio" if any(token in normalized for token in ratios) else ""


def _environment() -> dict[str, str]:
    packages = ("torch", "mmcv", "mmengine", "mmdet", "transformers", "peft")
    values = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not installed"
    try:
        import torch

        values["cuda_runtime"] = str(torch.version.cuda)
        values["cuda_device_count"] = str(torch.cuda.device_count())
        if torch.cuda.is_available():
            values["gpu_name"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            values["gpu_memory_gib"] = f"{properties.total_memory / 1024**3:.2f}"
    except (ImportError, RuntimeError):
        values["cuda_runtime"] = "unavailable"
    return values


def _workbook(path: Path):
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError as error:
        raise RuntimeError(
            "Excel export requires openpyxl. Install the project requirements "
            "or run `python -m pip install openpyxl`."
        ) from error

    if path.exists():
        workbook = load_workbook(path)
    else:
        workbook = Workbook()
        workbook.remove(workbook.active)

    for sheet_name, headers in SHEET_HEADERS.items():
        sheet = (
            workbook[sheet_name]
            if sheet_name in workbook.sheetnames
            else workbook.create_sheet(sheet_name)
        )
        if sheet.max_row == 1 and sheet.cell(1, 1).value is None:
            for column, header in enumerate(headers, start=1):
                sheet.cell(1, column).value = header
        elif [cell.value for cell in sheet[1]] != headers:
            raise ValueError(f"Unexpected header in worksheet: {sheet_name}")
    return workbook


def _append(sheet, row: list[Any]) -> None:
    sheet.append([_as_cell(value) for value in row])


def write_experiment_report(
    output_path: str | Path,
    metrics: dict[str, Any],
    *,
    config: str,
    checkpoint: str,
    seed: int | None,
    dataset: str,
    split: str = "test",
    notes: str = "",
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Append one evaluation result to an Excel workbook and raw JSON file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat(timespec="seconds")
    run_stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = run_id or f"{run_stamp}_{Path(config).stem}_seed{seed}"
    raw_path = output.parent / "raw_metrics" / f"{run_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    raw_payload = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "dataset": dataset,
        "split": split,
        "config": config,
        "checkpoint": checkpoint,
        "seed": seed,
        "metrics": {name: _as_cell(value) for name, value in metrics.items()},
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    workbook = _workbook(output)
    _append(workbook["runs"], [
        run_id, timestamp, dataset, split, config, checkpoint, seed, "completed",
    ])
    _append(workbook["ablations"], [run_id, dataset, Path(config).stem, seed])
    if notes:
        _append(workbook["notes"], [run_id, notes])

    for name, value in sorted(metrics.items()):
        family = _metric_family(name)
        unit = _metric_unit(name)
        _append(
            workbook["metrics_raw"],
            [run_id, dataset, split, family, name, value, unit, "runner.test"],
        )
        summary_row = [run_id, dataset, split, name, value]
        if family == "detection":
            _append(workbook["detection_summary"], summary_row)
        elif family == "segmentation":
            _append(workbook["segmentation_summary"], summary_row)
        elif family == "runtime":
            _append(workbook["runtime_efficiency"], summary_row + [unit])
        normalized = name.lower()
        if "class" in normalized or (
            any(token in normalized for token in ("bbox", "segm"))
            and any(token in normalized for token in ("_precision", "_recall"))
        ):
            _append(workbook["per_class"], summary_row)

    for name, value in _environment().items():
        _append(workbook["environment"], [run_id, name, value])
    workbook.save(output)
    return output, raw_path
