"""Validate the local PIXray and PIXray-A files used by all experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CLASSES = (
    "Bat",
    "Battery",
    "Dart",
    "Fireworks",
    "Gun",
    "Hammer",
    "Knife",
    "Lighter",
    "Pliers",
    "Pressure_vessel",
    "Razor_blade",
    "Saw_blade",
    "Scissors",
    "Screwdriver",
    "Wrench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/PIXray"))
    parser.add_argument(
        "--occ-ann-root",
        type=Path,
        default=Path("data/PIXray-A/annotations"),
    )
    parser.add_argument(
        "--check-content-hashes",
        action="store_true",
        help="Also detect byte-identical images shared by train and test.",
    )
    return parser.parse_args()


def validate(path: Path, image_root: Path, overlap: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    expected_categories = list(enumerate(CLASSES))
    actual_categories = [(item.get("id"), item.get("name")) for item in categories]
    if actual_categories != expected_categories:
        raise RuntimeError(f"Category mismatch in {path}: {actual_categories}")

    image_ids = [item.get("id") for item in images]
    annotation_ids = [item.get("id") for item in annotations]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"Duplicate image IDs in {path}")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise RuntimeError(f"Duplicate annotation IDs in {path}")
    known_images = set(image_ids)
    known_categories = set(range(len(CLASSES)))
    required = {"image_id", "category_id", "bbox", "segmentation"}
    if overlap:
        required.add("bg_object_segmentation")

    for annotation in annotations:
        if not required.issubset(annotation):
            raise RuntimeError(
                f"Annotation {annotation.get('id')} is incomplete in {path}"
            )
        if annotation["image_id"] not in known_images:
            raise RuntimeError(f"Unknown image ID in annotation {annotation.get('id')}")
        if annotation["category_id"] not in known_categories:
            raise RuntimeError(f"Unknown category in annotation {annotation.get('id')}")
        bbox = annotation["bbox"]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise RuntimeError(f"Invalid box in annotation {annotation.get('id')}")

    missing = [
        item["file_name"]
        for item in images
        if not (image_root / item["file_name"]).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} images referenced by {path} are missing; examples={missing[:5]}"
        )
    return payload


def _project_records(
    records: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[Any, tuple[Any, ...]]:
    return {
        record.get("id"): tuple(record.get(key) for key in keys)
        for record in records
    }


def assert_target_parity(
    standard: dict[str, Any], occlusion: dict[str, Any], split: str
) -> None:
    """Ensure PIXray-A adds overlap labels without changing target labels."""
    image_keys = ("file_name", "width", "height")
    standard_images = _project_records(standard["images"], image_keys)
    occlusion_images = _project_records(occlusion["images"], image_keys)
    if standard_images != occlusion_images:
        raise RuntimeError(
            f"{split}: PIXray and PIXray-A image records are not identical"
        )
    def group_targets(payload: dict[str, Any]) -> dict[tuple[Any, ...], list[dict]]:
        grouped: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            key = (
                annotation.get("image_id"),
                annotation.get("category_id"),
                annotation.get("iscrowd", 0),
                json.dumps(
                    annotation.get("segmentation"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            grouped[key].append(annotation)
        return grouped

    standard_targets = group_targets(standard)
    occlusion_targets = group_targets(occlusion)
    if standard_targets.keys() != occlusion_targets.keys():
        raise RuntimeError(
            f"{split}: PIXray-A changes target categories or masks instead of "
            "only adding overlap supervision"
        )
    for key, standard_group in standard_targets.items():
        occlusion_group = occlusion_targets[key]
        if len(standard_group) != len(occlusion_group):
            raise RuntimeError(f"{split}: target multiplicity differs for {key[:3]}")
        standard_group.sort(key=lambda item: tuple(item["bbox"]))
        occlusion_group.sort(key=lambda item: tuple(item["bbox"]))
        for standard_ann, occlusion_ann in zip(standard_group, occlusion_group):
            if any(
                abs(float(left) - float(right)) > 1e-3
                for left, right in zip(
                    standard_ann["bbox"], occlusion_ann["bbox"]
                )
            ):
                raise RuntimeError(f"{split}: target boxes differ for {key[:3]}")
            standard_area = float(standard_ann.get("area", 0.0))
            occlusion_area = float(occlusion_ann.get("area", 0.0))
            area_tolerance = max(1e-2, abs(standard_area) * 1e-6)
            if abs(standard_area - occlusion_area) > area_tolerance:
                raise RuntimeError(f"{split}: target areas differ for {key[:3]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_split_isolation(
    train: dict[str, Any], test: dict[str, Any], data_root: Path, check_hashes: bool
) -> None:
    train_names = {Path(item["file_name"]).name for item in train["images"]}
    test_names = {Path(item["file_name"]).name for item in test["images"]}
    shared_names = sorted(train_names & test_names)
    if shared_names:
        raise RuntimeError(
            "PIXray train/test filename leakage; examples=" f"{shared_names[:5]}"
        )
    # COCO image IDs are scoped to each annotation file and legitimately
    # restart at zero in PIXray. Filenames and optional content hashes are the
    # meaningful cross-split leakage checks.
    if not check_hashes:
        return

    train_by_size: dict[int, list[Path]] = defaultdict(list)
    test_by_size: dict[int, list[Path]] = defaultdict(list)
    for item in train["images"]:
        path = data_root / "train" / item["file_name"]
        train_by_size[path.stat().st_size].append(path)
    for item in test["images"]:
        path = data_root / "test" / item["file_name"]
        test_by_size[path.stat().st_size].append(path)

    duplicates: list[tuple[str, str]] = []
    for size in train_by_size.keys() & test_by_size.keys():
        train_hashes: dict[str, Path] = {
            _sha256(path): path for path in train_by_size[size]
        }
        for path in test_by_size[size]:
            match = train_hashes.get(_sha256(path))
            if match is not None:
                duplicates.append((match.name, path.name))
                if len(duplicates) >= 5:
                    break
        if len(duplicates) >= 5:
            break
    if duplicates:
        raise RuntimeError(
            f"PIXray train/test contain byte-identical images: {duplicates}"
        )


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    occ_root = args.occ_ann_root.resolve()
    standard_payloads: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        standard = validate(
            data_root / "annotations" / f"{split}.json",
            data_root / split,
            overlap=False,
        )
        occlusion = validate(
            occ_root / f"{split}_occ.json",
            data_root / split,
            overlap=True,
        )
        assert_target_parity(standard, occlusion, split)
        standard_payloads[split] = standard
        print(
            f"{split}: images={len(standard['images'])}, "
            f"instances={len(standard['annotations'])}, target_parity=PASS"
        )
    assert_split_isolation(
        standard_payloads["train"],
        standard_payloads["test"],
        data_root,
        args.check_content_hashes,
    )
    isolation = "names/hashes" if args.check_content_hashes else "names"
    print(f"train/test isolation ({isolation}): PASS")
    print("PIXray/PIXray-A audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
