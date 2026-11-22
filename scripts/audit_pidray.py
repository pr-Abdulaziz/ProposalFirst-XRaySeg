#!/usr/bin/env python3
"""Read-only audit of the PIDray-30/PIDray-A paired experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
    "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter",
)
EXPECTED_IMAGES = {
    "train": 23074,
    "easy": 24758,
    "hard": 9746,
    "hidden": 13069,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/PIDray"
    )
    parser.add_argument(
        "--occ-ann-root", type=Path,
        default=ROOT / "data/PIDray-A/annotations",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "docs/pidray30_protocol_manifest.json",
    )
    parser.add_argument(
        "--skip-fingerprints", action="store_true",
        help="Skip annotation SHA-256 checks during repeated local audits.",
    )
    parser.add_argument(
        "--check-content-hashes", action="store_true",
        help="Also hash same-size cross-split images (slower).",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing annotation file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def image_index(root: Path, train: bool) -> dict[str, Path]:
    directories = (root / "train0", root / "train1") if train else (root,)
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            raise RuntimeError(f"Missing image directory: {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if entry.name in result:
                    duplicates.add(entry.name)
                result[entry.name] = Path(entry.path)
    if duplicates:
        raise RuntimeError(
            f"Duplicate filenames across PIDray shards: {sorted(duplicates)[:5]}"
        )
    return result


def validate(
    payload: dict[str, Any], path: Path, available: dict[str, Path],
    expected_count: int, overlap: bool,
) -> None:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if len(images) != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} images, found {len(images)}"
        )
    category_table = tuple((item.get("id"), item.get("name")) for item in categories)
    if category_table != tuple(enumerate(CLASSES, start=1)):
        raise RuntimeError(f"{path}: category IDs/names are not canonical PIDray")
    image_ids = [item.get("id") for item in images]
    annotation_ids = [item.get("id") for item in annotations]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"{path}: duplicate image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise RuntimeError(f"{path}: duplicate annotation IDs")
    known_images = set(image_ids)
    required = {"image_id", "category_id", "bbox", "segmentation"}
    if overlap:
        required.add("bg_object_segmentation")
    for annotation in annotations:
        if not required.issubset(annotation):
            raise RuntimeError(
                f"{path}: annotation {annotation.get('id')} is incomplete"
            )
        bbox = annotation["bbox"]
        if (
            annotation["image_id"] not in known_images
            or annotation["category_id"] not in range(1, 13)
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise RuntimeError(
                f"{path}: annotation {annotation.get('id')} is invalid"
            )
    missing = [
        item.get("file_name") for item in images
        if Path(str(item.get("file_name"))).name not in available
    ]
    if missing:
        raise RuntimeError(
            f"{path}: {len(missing)} referenced images are missing; "
            f"examples={missing[:5]}"
        )


def grouped_targets(payload: dict[str, Any]) -> dict[tuple[Any, ...], list[dict]]:
    grouped: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        key = (
            annotation.get("image_id"),
            annotation.get("category_id"),
            annotation.get("iscrowd", 0),
            json.dumps(
                annotation.get("segmentation"), sort_keys=True,
                separators=(",", ":"),
            ),
        )
        grouped[key].append(annotation)
    return grouped


def assert_pairing(
    standard: dict[str, Any], occlusion: dict[str, Any], split: str
) -> int:
    image_fields = ("id", "file_name", "width", "height")
    standard_images = [
        tuple(item.get(key) for key in image_fields) for item in standard["images"]
    ]
    occlusion_images = [
        tuple(item.get(key) for key in image_fields) for item in occlusion["images"]
    ]
    if standard_images != occlusion_images:
        raise RuntimeError(f"{split}: PIDray/PIDray-A image records differ")
    standard_targets = grouped_targets(standard)
    occlusion_targets = grouped_targets(occlusion)
    if standard_targets.keys() != occlusion_targets.keys():
        raise RuntimeError(f"{split}: PIDray/PIDray-A target masks differ")
    area_metadata_differences = 0
    for key, left in standard_targets.items():
        right = occlusion_targets[key]
        left_values = sorted(left, key=lambda item: tuple(item["bbox"]))
        right_values = sorted(right, key=lambda item: tuple(item["bbox"]))
        if len(left_values) != len(right_values):
            raise RuntimeError(f"{split}: target multiplicity differs")
        for standard_ann, occlusion_ann in zip(left_values, right_values):
            if any(
                abs(float(a) - float(b)) > 1e-3
                for a, b in zip(standard_ann["bbox"], occlusion_ann["bbox"])
            ):
                raise RuntimeError(f"{split}: PIDray/PIDray-A target boxes differ")
            if standard_ann.get("area") != occlusion_ann.get("area"):
                area_metadata_differences += 1
    return area_metadata_differences


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_annotation_manifest(path: Path) -> None:
    manifest = load(path.resolve())
    files = manifest.get("annotation_files", {})
    if len(files) != 8:
        raise RuntimeError(f"{path}: expected eight annotation fingerprints")
    for relative_path, expected in files.items():
        annotation = (ROOT / relative_path).resolve()
        if not annotation.is_file():
            raise RuntimeError(f"Manifest annotation is missing: {annotation}")
        if annotation.stat().st_size != expected.get("bytes"):
            raise RuntimeError(f"Annotation size changed: {annotation}")
        if sha256(annotation) != expected.get("sha256"):
            raise RuntimeError(f"Annotation content changed: {annotation}")


def assert_isolation(
    split_paths: dict[str, list[Path]],
    positive_names: dict[str, set[str]],
    check_hashes: bool,
) -> None:
    names = {split: {path.name for path in paths} for split, paths in split_paths.items()}
    split_names = list(names)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = names[left] & names[right]
            if overlap:
                raise RuntimeError(
                    f"{left}/{right} filename leakage: {sorted(overlap)[:5]}"
                )
    if not check_hashes:
        return
    size_owners: dict[int, set[str]] = defaultdict(set)
    for split, paths in split_paths.items():
        for path in paths:
            size_owners[path.stat().st_size].add(split)
    shared_sizes = {
        size for size, owners in size_owners.items() if len(owners) > 1
    }
    candidate_count = sum(
        path.stat().st_size in shared_sizes
        for paths in split_paths.values()
        for path in paths
    )
    print(
        "content-leakage candidates: "
        f"{candidate_count}/{sum(map(len, split_paths.values()))} images",
        flush=True,
    )
    hashes: dict[str, dict[str, Path]] = {}
    for split, paths in split_paths.items():
        hashes[split] = {
            sha256(path): path
            for path in paths
            if path.stat().st_size in shared_sizes
        }
    violations = []
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = hashes[left].keys() & hashes[right].keys()
            if overlap:
                def involves_positive(value: str) -> bool:
                    return (
                        hashes[left][value].name in positive_names[left]
                        or hashes[right][value].name in positive_names[right]
                    )

                positive_involved = sum(
                    involves_positive(value)
                    for value in overlap
                )
                ordered = sorted(overlap, key=lambda value: not involves_positive(value))
                examples = [
                    (hashes[left][value].name, hashes[right][value].name)
                    for value in ordered[:5]
                ]
                violations.append(
                    f"{left}/{right}={len(overlap)} exact duplicates; "
                    f"positive-involved={positive_involved}; "
                    f"examples={examples}"
                )
    if violations:
        raise RuntimeError("Cross-split byte leakage: " + "; ".join(violations))


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    occ_root = args.occ_ann_root.resolve()
    if not args.skip_fingerprints:
        verify_annotation_manifest(args.manifest)
        print("annotation fingerprints: PASS")
    train_files = image_index(data_root / "train", train=True)
    test_files = image_index(data_root / "test", train=False)
    split_paths: dict[str, list[Path]] = {}
    positive_names: dict[str, set[str]] = {}
    for split in ("train", "easy", "hard", "hidden"):
        standard_name = "train.json" if split == "train" else f"test_{split}.json"
        occ_name = "train_occ.json" if split == "train" else f"test_{split}_occ.json"
        standard_path = data_root / "annotations" / standard_name
        occ_path = occ_root / occ_name
        available = train_files if split == "train" else test_files
        standard = load(standard_path)
        occlusion = load(occ_path)
        validate(
            standard, standard_path, available, EXPECTED_IMAGES[split], overlap=False
        )
        validate(
            occlusion, occ_path, available, EXPECTED_IMAGES[split], overlap=True
        )
        area_differences = assert_pairing(standard, occlusion, split)
        image_names = [Path(item["file_name"]).name for item in standard["images"]]
        split_paths[split] = [available[name] for name in image_names]
        annotated = {item["image_id"] for item in standard["annotations"]}
        positive_names[split] = {
            Path(item["file_name"]).name
            for item in standard["images"]
            if item["id"] in annotated
        }
        normal = sum(item["id"] not in annotated for item in standard["images"])
        overlap_instances = sum(
            bool(item.get("bg_object_segmentation"))
            for item in occlusion["annotations"]
        )
        print(
            f"{split}: images={len(standard['images'])}, "
            f"instances={len(standard['annotations'])}, normal={normal}, "
            "target parity=PASS, "
            f"overlap_instances={overlap_instances}, "
            f"area_metadata_differences={area_differences}"
        )
    assert_isolation(split_paths, positive_names, args.check_content_hashes)
    if sum(EXPECTED_IMAGES[name] for name in ("easy", "hard", "hidden")) != 47573:
        raise AssertionError("Internal test-count contract is invalid")
    isolation = "filenames+content" if args.check_content_hashes else "filenames"
    print(f"split isolation ({isolation}): PASS")
    print("PIDray-30/PIDray-A audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
