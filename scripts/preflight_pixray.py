#!/usr/bin/env python3
"""Validate one controlled PIXray/PIXray-A experiment configuration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = (
    "Bat", "Battery", "Dart", "Fireworks", "Gun", "Hammer", "Knife",
    "Lighter", "Pliers", "Pressure_vessel", "Razor_blade", "Saw_blade",
    "Scissors", "Screwdriver", "Wrench",
)
EXPECTED_COUNTS = {"train": 3560, "test": 1486}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-model", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def leaf_dataset(value: Mapping) -> Mapping:
    while isinstance(value.get("dataset"), Mapping):
        value = value["dataset"]
    return value


def path_for(value: str, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root or ROOT) / path
    return path.resolve()


def validate_json(annotation: Path, image_root: Path, expected: int, overlap: bool) -> list[str]:
    errors: list[str] = []
    if not annotation.is_file() or not image_root.is_dir():
        return [f"Missing annotation or image root: {annotation}, {image_root}"]
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = tuple(category.get("name") for category in payload.get("categories", []))
    if len(images) != expected:
        errors.append(f"{annotation.name}: expected {expected} images, found {len(images)}")
    if categories != CLASSES:
        errors.append(f"{annotation.name}: categories differ from canonical PIXray order")
    available = {item.name for item in image_root.iterdir() if item.is_file()}
    missing = [item.get("file_name") for item in images if Path(str(item.get("file_name"))).name not in available]
    if missing:
        errors.append(f"{annotation.name}: {len(missing)} image files missing; examples={missing[:5]}")
    required = {"image_id", "category_id", "bbox", "segmentation"}
    if overlap:
        required.add("bg_object_segmentation")
    invalid = sum(not required.issubset(item) for item in annotations)
    if invalid:
        errors.append(f"{annotation.name}: {invalid} annotations lack required fields")
    return errors


def main() -> int:
    args = parse_args()
    os.environ.setdefault("XRAY_PROJECT_ROOT", str(ROOT))
    os.environ.setdefault("PIXRAY_DATA_ROOT", str(ROOT / "data/PIXray"))
    os.environ.setdefault("PIXRAYA_ANN_ROOT", str(ROOT / "data/PIXray-A/annotations"))
    config_path = path_for(args.config)
    errors: list[str] = []
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path, import_custom_modules=True)
        copy.deepcopy(cfg)
    except Exception as exc:
        print(f"ERROR: cannot load/deepcopy {config_path}: {exc}")
        return 1

    if cfg.get("randomness", {}).get("seed") != 42:
        errors.append("Seed must equal 42")
    if cfg.get("max_epochs") != 60 or cfg.train_cfg.get("max_epochs") != 60:
        errors.append("Training duration must equal 60 epochs")
    wrapper = cfg.optim_wrapper
    optimizer = wrapper.optimizer
    if wrapper.get("type") != "AmpOptimWrapper" or wrapper.get("dtype") != "float16":
        errors.append("Training must use AmpOptimWrapper with float16")
    if optimizer.get("type") != "AdamW" or float(optimizer.get("lr", -1)) != 1e-4:
        errors.append("Optimizer must be AdamW with base LR 1e-4")
    if float(optimizer.get("weight_decay", -1)) != 0.05:
        errors.append("Weight decay must equal 0.05")
    physical = int(os.getenv("XRAY_BATCH_SIZE_PER_GPU", "2"))
    effective = int(os.getenv("XRAY_EFFECTIVE_BATCH_SIZE", "16"))
    if cfg.train_dataloader.batch_size != physical:
        errors.append("Resolved physical batch differs from XRAY_BATCH_SIZE_PER_GPU")
    if wrapper.get("accumulative_counts") != effective // physical:
        errors.append("Gradient accumulation does not produce the effective batch")
    clip = wrapper.get("clip_grad", {})
    if float(clip.get("max_norm", -1)) != 1.0:
        errors.append("Gradient clipping max_norm must equal 1.0")
    if cfg.get("load_from") is not None or bool(cfg.get("resume", False)):
        errors.append("Fresh comparison requires load_from=None and resume=False")
    if any(cfg.get(name) is not None for name in ("val_cfg", "val_dataloader", "val_evaluator")):
        errors.append("Validation must remain disabled for the controlled protocol")

    if not args.config_only:
        for split in ("train", "test"):
            dataset = leaf_dataset(cfg.get(f"{split}_dataloader", {}).get("dataset", {}))
            dataset_type = str(dataset.get("type", ""))
            if dataset_type not in {"PIXrayDataset", "PIXrayOcclusionDataset"}:
                errors.append(f"Unsupported {split} dataset type: {dataset_type}")
                continue
            data_root = path_for(str(dataset.get("data_root", "")))
            annotation = path_for(str(dataset.get("ann_file", "")), data_root)
            prefix = dataset.get("data_prefix", {}).get("img", "")
            image_root = path_for(str(prefix), data_root)
            if tuple(dataset.get("metainfo", {}).get("classes", ())) != CLASSES:
                errors.append(f"{split} class order differs from canonical PIXray")
            errors.extend(validate_json(annotation, image_root, EXPECTED_COUNTS[split], "Occlusion" in dataset_type))

        evaluators = cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]
        # XRayCocoMetric is the CocoMetric subclass this project scores with; it
        # recovers per-iou-type classwise AP and mask recall, which the upstream
        # metric drops. Accept either so the check tracks the annotation file,
        # which is what matters for comparability, not the class name.
        coco_metric_types = ("CocoMetric", "XRayCocoMetric")
        coco = next(
            (item for item in evaluators if item.get("type") in coco_metric_types),
            None,
        )
        canonical = (ROOT / "data/PIXray/annotations/test.json").resolve()
        if coco is None or path_for(str(coco.get("ann_file", ""))) != canonical:
            errors.append(
                "Primary COCO metric must use data/PIXray/annotations/test.json"
            )

    if args.build_model and not errors:
        try:
            import torch
            from mmengine.utils import import_modules_from_strings
            from mmdet.registry import MODELS
            from mmdet.utils import register_all_modules
            if not torch.cuda.is_available():
                errors.append("CUDA is not available")
            else:
                register_all_modules(init_default_scope=True)
                custom_imports = cfg.get("custom_imports")
                if custom_imports:
                    import_modules_from_strings(**custom_imports)
                model = MODELS.build(cfg.model)
                del model
        except Exception as exc:
            traceback.print_exc()
            errors.append(f"Model construction failed: {exc}")

    print(f"Config: {config_path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PIXray experiment preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
