"""Audit the public XSeg layout before selecting a training formulation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/XSeg"))
    parser.add_argument("--sample-masks", type=int, default=200)
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing XSeg split: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_stem(directory: Path, stem: str, suffixes: tuple[str, ...]) -> Path:
    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No file for {stem!r} below {directory}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    image_dir = root / "JPEGImages"
    mask_dir = root / "SegmentationClass"
    split_dir = root / "ImageSets" / "Segmentation"

    splits = {
        name: read_split(split_dir / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    sets = {name: set(items) for name, items in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise RuntimeError(
                f"XSeg {left}/{right} overlap contains {len(overlap)} names"
            )

    all_names = sorted(set().union(*sets.values()))
    value_counts: Counter[int] = Counter()
    for index, stem in enumerate(all_names):
        resolve_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
        mask_path = resolve_stem(mask_dir, stem, (".png",))
        if index < args.sample_masks:
            values = np.unique(np.asarray(Image.open(mask_path)))
            value_counts.update(int(value) for value in values)

    print(f"XSeg root: {root}")
    for name, items in splits.items():
        print(f"{name}: {len(items)} images")
    print(f"sampled mask values: {sorted(value_counts)}")
    print("Layout and split references: PASS")
    if len(value_counts) <= 2:
        print(
            "WARNING: sampled masks look binary; instance/category experiments "
            "must remain disabled."
        )
    else:
        print(
            "NOTICE: multiple mask values exist, but their category/instance "
            "semantics still require documentation before COCO conversion."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
