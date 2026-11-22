"""Clip invalid PIDray COCO geometry without changing split membership."""

import argparse
import json
from pathlib import Path

from pycocotools import mask as mask_utils


def clip_polygon(segmentation: list, width: int, height: int) -> list:
    clipped = []
    for polygon in segmentation:
        coordinates = list(polygon)
        for index in range(0, len(coordinates), 2):
            coordinates[index] = min(max(float(coordinates[index]), 0.0), width)
            coordinates[index + 1] = min(
                max(float(coordinates[index + 1]), 0.0), height
            )
        clipped.append(coordinates)
    return clipped


def polygon_area(segmentation: list, width: int, height: int) -> float:
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return float(mask_utils.area(mask_utils.merge(rles)))


def repair(payload: dict) -> tuple[dict, list[int]]:
    images = {image["id"]: image for image in payload["images"]}
    changed = []
    for annotation in payload["annotations"]:
        image = images[annotation["image_id"]]
        width, height = image["width"], image["height"]
        x, y, box_width, box_height = map(float, annotation["bbox"])
        x1 = min(max(x, 0.0), width)
        y1 = min(max(y, 0.0), height)
        x2 = min(max(x + box_width, 0.0), width)
        y2 = min(max(y + box_height, 0.0), height)
        clipped_bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

        segmentation = annotation.get("segmentation")
        clipped_segmentation = (
            clip_polygon(segmentation, width, height)
            if isinstance(segmentation, list)
            else segmentation
        )
        if clipped_bbox == annotation["bbox"] and clipped_segmentation == segmentation:
            continue

        annotation["bbox"] = clipped_bbox
        if clipped_segmentation is not None:
            annotation["segmentation"] = clipped_segmentation
        if isinstance(clipped_segmentation, list) and clipped_segmentation:
            annotation["area"] = polygon_area(
                clipped_segmentation, width, height
            )
        changed.append(annotation["id"])
    return payload, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_files", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in args.annotation_files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        payload, changed = repair(payload)
        print(f"{path}: repaired={len(changed)} ids={changed[:10]}")
        if changed and args.overwrite:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
