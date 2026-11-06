"""Prepare the full PIDray-A training annotation used by every experiment."""

import argparse
import copy
import json
from pathlib import Path


def clip_out_of_bounds_geometry(payload: dict) -> tuple[dict, list[int]]:
    """Clip only annotations that truly extend beyond their image bounds."""
    payload = copy.deepcopy(payload)
    images = {image["id"]: image for image in payload["images"]}
    changed = []
    for annotation in payload["annotations"]:
        image = images[annotation["image_id"]]
        width, height = float(image["width"]), float(image["height"])
        x, y, box_width, box_height = map(float, annotation["bbox"])
        geometry_changed = (
            x < 0.0
            or y < 0.0
            or x + box_width > width
            or y + box_height > height
            or box_width < 0.0
            or box_height < 0.0
        )

        clipped_segments = []
        for polygon in annotation.get("segmentation", []):
            clipped = []
            for index, coordinate in enumerate(polygon):
                limit = width if index % 2 == 0 else height
                value = float(coordinate)
                geometry_changed |= value < 0.0 or value > limit
                clipped.append(min(max(value, 0.0), limit))
            clipped_segments.append(clipped)

        if not geometry_changed:
            continue

        x1 = min(max(x, 0.0), width)
        y1 = min(max(y, 0.0), height)
        x2 = min(max(x + box_width, 0.0), width)
        y2 = min(max(y + box_height, 0.0), height)
        clipped_box = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        annotation["bbox"] = clipped_box
        annotation["segmentation"] = clipped_segments
        annotation["area"] = clipped_box[2] * clipped_box[3]
        changed.append(annotation["id"])
    return payload, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("data/PIDray-A/annotations"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    annotation_root = args.annotation_root.resolve()
    source = annotation_root / "train_occ.json"
    output = annotation_root / "train_occ_runtime.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; use --overwrite.")

    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    payload, changed = clip_out_of_bounds_geometry(payload)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    annotated_images = {ann["image_id"] for ann in payload["annotations"]}
    print(
        f"{output}: images={len(payload['images'])}, "
        f"annotations={len(payload['annotations'])}, "
        f"normal={len(payload['images']) - len(annotated_images)}, "
        f"repaired={len(changed)}"
    )


if __name__ == "__main__":
    main()
