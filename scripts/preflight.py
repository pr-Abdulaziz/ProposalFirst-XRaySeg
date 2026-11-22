#!/usr/bin/env python3
"""Validate a PIDray-30 experiment before launching it on a remote GPU.

The active protocol uses the deterministic PIDray training subset and the
complete Easy, Hard, and Hidden test splits, with paired PIDray-A annotations.
Use ``--data-only`` on a CPU/local machine to validate paths and annotations
without requiring CUDA or pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/occlusion/pidray30_s2_occ_bisam.py"
)
EXPECTED_SEED = 42
PIDRAY_CATEGORIES = (
    (1, "Baton"),
    (2, "Pliers"),
    (3, "Hammer"),
    (4, "Powerbank"),
    (5, "Scissors"),
    (6, "Wrench"),
    (7, "Gun"),
    (8, "Bullet"),
    (9, "Sprayer"),
    (10, "HandCuffs"),
    (11, "Knife"),
    (12, "Lighter"),
)
EXPECTED_CATEGORY_TABLES = {
    "PIDrayDataset": PIDRAY_CATEGORIES,
    "PIDrayOcclusionDataset": PIDRAY_CATEGORIES,
}
EXPECTED_PIDRAY30_COUNTS = {
    "train": 23074,
    "easy": 24758,
    "hard": 9746,
    "hidden": 13069,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Also construct the complete model after static validation.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate configs, paths, and annotations without CUDA/assets.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate the resolved config without scanning dataset files.",
    )
    args = parser.parse_args()
    if sum((args.data_only, args.config_only, args.build_model)) > 1:
        parser.error(
            "--data-only, --config-only, and --build-model are mutually exclusive"
        )
    return args


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def unwrap_dataset(dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the leaf dataset beneath wrappers such as RepeatDataset."""
    current = dataset
    while isinstance(current.get("dataset"), Mapping):
        current = current["dataset"]
    return current


def resolve_config_path(value: str, data_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if data_root is not None:
        return (data_root / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def dataset_paths(dataset: Mapping[str, Any]) -> tuple[Path, Path]:
    data_root = resolve_config_path(str(dataset.get("data_root", "")))
    ann_path = resolve_config_path(str(dataset.get("ann_file", "")), data_root)
    prefix = dataset.get("data_prefix", {})
    image_prefix = prefix.get("img", "") if isinstance(prefix, Mapping) else ""
    image_root = resolve_config_path(str(image_prefix), data_root)
    return ann_path, image_root


def index_image_names(image_root: Path) -> set[str]:
    """Index flat images and PIDray's train0/train1 layout in one pass."""
    available: set[str] = set()
    for directory in (image_root, image_root / "train0", image_root / "train1"):
        if not directory.is_dir():
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    available.add(entry.name)
    return available


def validate_annotations(
    ann_path: Path,
    image_root: Path,
    expected_categories: tuple[tuple[int, str], ...],
    require_overlap_masks: bool,
    errors: list[str],
) -> dict[str, int]:
    stats = {"images": 0, "annotations": 0, "normal_images": 0}
    if not ann_path.is_file():
        errors.append(f"Annotation file does not exist: {ann_path}")
        return stats
    if not image_root.is_dir():
        errors.append(f"Image directory does not exist: {image_root}")
        return stats

    try:
        with ann_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read {ann_path}: {exc}")
        return stats

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    stats["images"] = len(images)
    stats["annotations"] = len(annotations)

    category_table = tuple(
        (category.get("id"), category.get("name")) for category in categories
    )
    require(
        len(categories) == len(expected_categories),
        f"{ann_path.name}: expected {len(expected_categories)} categories, "
        f"found {len(categories)}",
        errors,
    )
    require(
        category_table == expected_categories,
        f"{ann_path.name}: category IDs/names differ from the canonical "
        "dataset definition",
        errors,
    )

    image_ids = {image.get("id") for image in images}
    annotated_image_ids = {ann.get("image_id") for ann in annotations}
    available_images = index_image_names(image_root)
    missing_images: list[str] = []
    for image in images:
        file_name = image.get("file_name")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name not in available_images
        ) and len(missing_images) < 5:
            missing_images.append(str(file_name))
    if missing_images:
        errors.append(
            f"{ann_path.name}: images are missing below {image_root}; "
            f"examples: {missing_images}"
        )

    required_fields = {"image_id", "category_id", "bbox", "segmentation"}
    if require_overlap_masks:
        required_fields.add("bg_object_segmentation")
    invalid_annotations = 0
    for ann in annotations:
        if not required_fields.issubset(ann):
            invalid_annotations += 1
            continue
        bbox = ann.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or bbox[2] <= 0
            or bbox[3] <= 0
            or ann.get("image_id") not in image_ids
        ):
            invalid_annotations += 1
    require(
        invalid_annotations == 0,
        f"{ann_path.name}: {invalid_annotations} annotations have missing fields, "
        "invalid boxes, or unknown image IDs",
        errors,
    )
    stats["normal_images"] = len(image_ids - annotated_image_ids)
    return stats


def contains_key(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in keys or contains_key(child, keys)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_key(child, keys) for child in value)
    return False


def contains_value(value: Any, expected: str) -> bool:
    if isinstance(value, Mapping):
        return any(contains_value(child, expected) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_value(child, expected) for child in value)
    return value == expected


def validate_runtime_contract(config: Any, errors: list[str]) -> None:
    randomness = config.get("randomness", {})
    require(
        randomness.get("seed") == EXPECTED_SEED,
        f"All experiments must use seed {EXPECTED_SEED}",
        errors,
    )

    custom_hooks = config.get("custom_hooks", [])
    default_hooks = config.get("default_hooks", {})
    progress_hook = default_hooks.get("progress", {})
    has_tqdm = any(
        hook.get("type") == "TqdmProgressHook" for hook in custom_hooks
    ) or (
        isinstance(progress_hook, Mapping)
        and progress_hook.get("type") == "TqdmProgressHook"
    )
    require(has_tqdm, "TqdmProgressHook is missing from the runtime hooks", errors)

    uses_validation = bool(config.get("uses_validation", False))
    validation_parts = (
        config.get("val_cfg"),
        config.get("val_dataloader"),
        config.get("val_evaluator"),
    )
    if uses_validation:
        require(
            all(part is not None for part in validation_parts),
            "uses_validation=True requires val_cfg, val_dataloader, and val_evaluator",
            errors,
        )
    else:
        require(
            all(part is None for part in validation_parts),
            "Set uses_validation=True or disable every validation component",
            errors,
        )

    if config.model.get("type") == "ProposalFirstPIDrayOcc":
        require(
            config.model.get("training_gate") == "none",
            "Hybrid training_gate must remain 'none' so positive RoIs are supervised",
            errors,
        )

    data_root_value = str(config.get("data_root", ""))
    require(
        Path(data_root_value).name == "PIDray",
        "Active data_root must resolve to the PIDray directory",
        errors,
    )


def validate_assets(config: Any, errors: list[str]) -> dict[str, bool]:
    model = config.model
    needs_sam = contains_key(
        model,
        {"sam_image_encoder", "sam_positional_embedding", "hf_pretrain_name"},
    )
    needs_resnet101 = contains_value(model, "torchvision://resnet101")

    if needs_sam:
        sam_dir = resolve_config_path(
            os.getenv("SAM_VIT_LARGE_DIR", "pretrained/sam_vit_large")
        )
        sam_checkpoint = resolve_config_path(
            os.getenv(
                "SAM_VIT_LARGE_CKPT",
                str(sam_dir / "pytorch_model_new.bin"),
            )
        )
        require((sam_dir / "config.json").is_file(), f"Missing SAM config: {sam_dir / 'config.json'}", errors)
        require(sam_checkpoint.is_file(), f"Missing converted SAM checkpoint: {sam_checkpoint}", errors)

    if needs_resnet101:
        torch_home = resolve_config_path(
            os.getenv("TORCH_HOME", "pretrained/torch")
        )
        resnet_checkpoints = list(
            (torch_home / "hub/checkpoints").glob("resnet101-*.pth")
        )
        require(
            bool(resnet_checkpoints),
            "Missing cached ImageNet ResNet-101 checkpoint below "
            f"{torch_home / 'hub/checkpoints'}",
            errors,
        )
    return {"sam": needs_sam, "resnet101": needs_resnet101}


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    config_path = resolve_config_path(args.config)
    require(config_path.is_file(), f"Config does not exist: {config_path}", errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    try:
        from mmengine.config import Config
    except ImportError as exc:
        print(f"ERROR: MMEngine is not importable: {exc}")
        return 1

    try:
        config = Config.fromfile(
            config_path,
            import_custom_modules=not (args.data_only or args.config_only),
        )
    except Exception as exc:  # noqa: BLE001 - preserve config diagnostics
        print(f"ERROR: could not load config {config_path}: {exc}")
        return 1

    validate_runtime_contract(config, errors)

    dataset_summaries: list[tuple[str, Path, Path, dict[str, int]]] = []
    test_dataset: Mapping[str, Any] | None = None
    test_categories: tuple[tuple[int, str], ...] = ()
    test_image_root: Path | None = None
    for split in (() if args.config_only else ("train", "test")):
        dataloader = config.get(f"{split}_dataloader")
        if not isinstance(dataloader, Mapping):
            errors.append(f"{split}_dataloader is missing")
            continue
        dataset = unwrap_dataset(dataloader.get("dataset", {}))
        classes = tuple(dataset.get("metainfo", {}).get("classes", ()))
        ann_path, image_root = dataset_paths(dataset)
        dataset_type = str(dataset.get("type", ""))
        expected_categories = EXPECTED_CATEGORY_TABLES.get(dataset_type, ())
        expected_classes = tuple(name for _, name in expected_categories)
        require(
            bool(expected_categories),
            f"No canonical category table is defined for {dataset_type!r}",
            errors,
        )
        require(
            classes == expected_classes,
            f"{split} config classes differ from the canonical "
            f"{dataset_type} definition",
            errors,
        )
        stats = validate_annotations(
            ann_path,
            image_root,
            expected_categories,
            require_overlap_masks="Occlusion" in dataset_type,
            errors=errors,
        )
        dataset_summaries.append((split, ann_path, image_root, stats))
        if split == "train":
            require(
                stats["images"] == EXPECTED_PIDRAY30_COUNTS["train"],
                "Training annotations must contain exactly 23,074 PIDray-30 images",
                errors,
            )
        if split == "test":
            test_dataset = dataset
            test_categories = expected_categories
            test_image_root = image_root

    evaluator = config.get("test_evaluator", {})
    evaluators = evaluator if isinstance(evaluator, list) else [evaluator]
    if test_dataset is not None and test_image_root is not None:
        require_overlap_masks = "Occlusion" in str(test_dataset.get("type", ""))
        for item in evaluators:
            if not isinstance(item, Mapping):
                continue
            subset_files = item.get("subset_ann_files", {})
            if not isinstance(subset_files, Mapping):
                continue
            for subset, path_value in subset_files.items():
                subset_path = resolve_config_path(str(path_value))
                stats = validate_annotations(
                    subset_path,
                    test_image_root,
                    test_categories,
                    require_overlap_masks=require_overlap_masks,
                    errors=errors,
                )
                dataset_summaries.append(
                    (f"test_{subset}", subset_path, test_image_root, stats)
                )

        split_files = config.get("test_split_ann_files", {})
        metric_files = config.get("metric_split_ann_files", split_files)
        require(
            set(split_files) == set(EXPECTED_PIDRAY30_COUNTS) - {"train"},
            "Config must define Easy, Hard, and Hidden dataset annotations",
            errors,
        )
        require(
            set(metric_files) == set(EXPECTED_PIDRAY30_COUNTS) - {"train"},
            "Config must define Easy, Hard, and Hidden metric annotations",
            errors,
        )
        for split, path_value in split_files.items():
            split_path = resolve_config_path(str(path_value))
            stats = validate_annotations(
                split_path,
                test_image_root,
                test_categories,
                require_overlap_masks=require_overlap_masks,
                errors=errors,
            )
            require(
                stats["images"] == EXPECTED_PIDRAY30_COUNTS[split],
                f"{split} must contain {EXPECTED_PIDRAY30_COUNTS[split]} images",
                errors,
            )
            dataset_summaries.append(
                (f"test_{split}_dataset", split_path, test_image_root, stats)
            )
        for split, path_value in metric_files.items():
            metric_path = resolve_config_path(str(path_value))
            stats = validate_annotations(
                metric_path,
                test_image_root,
                PIDRAY_CATEGORIES,
                require_overlap_masks=False,
                errors=errors,
            )
            require(
                stats["images"] == EXPECTED_PIDRAY30_COUNTS[split],
                f"{split} metric file has the wrong image count",
                errors,
            )

    assets = {"sam": False, "resnet101": False}
    if not args.data_only and not args.config_only:
        assets = validate_assets(config, errors)
        try:
            import torch

            require(torch.cuda.is_available(), "CUDA is not available", errors)
        except ImportError as exc:
            errors.append(f"PyTorch is not importable: {exc}")

    if args.build_model and not errors:
        try:
            from mmengine.utils import import_modules_from_strings

            from mmdet.registry import MODELS
            from mmdet.utils import register_all_modules

            register_all_modules(init_default_scope=True)
            custom_imports = config.get("custom_imports")
            if custom_imports:
                import_modules_from_strings(**custom_imports)
            model = MODELS.build(config.model)
            del model
        except Exception as exc:  # noqa: BLE001 - registry errors vary by model
            traceback.print_exc()
            errors.append(f"Complete model construction failed: {exc}")

    print(f"Config: {config_path}")
    print(f"Model: {config.model.get('type')}")
    for split, ann_path, image_root, stats in dataset_summaries:
        print(
            f"{split}: annotations={ann_path} images={image_root} "
            f"count={stats['images']} instances={stats['annotations']} "
            f"normal={stats['normal_images']}"
        )
    if not args.data_only and not args.config_only:
        print(
            "Required pretrained assets: "
            f"SAM={'yes' if assets['sam'] else 'no'}, "
            f"ResNet-101={'yes' if assets['resnet101'] else 'no'}"
        )

    if errors:
        print("\nPreflight failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    if args.config_only:
        mode = "config"
    elif args.data_only:
        mode = "data/config"
    else:
        mode = "full runtime"
    print(f"\nPreflight passed ({mode}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
