#!/usr/bin/env python3
"""Render a publication-style segmentation comparison on identical images."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from analyze_segmentation_errors import array, decode, get, mask_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="LABEL=PREDICTIONS.pkl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument(
        "--sheet-name", default="Qualitative",
        help="Excel worksheet name used when --workbook is supplied.",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--fade", type=float, default=0.45,
        help="Lighten the X-ray plate toward white before overlaying masks (0 disables).",
    )
    parser.add_argument("--image-ids", type=int, nargs="*")
    return parser.parse_args()


def prediction_index(path: Path) -> dict[int, Any]:
    with path.open("rb") as stream:
        samples = pickle.load(stream)
    return {
        int(get(sample, "img_id", get(get(sample, "metainfo", {}), "img_id", -1))): sample
        for sample in samples
    }


def predicted_masks(sample: Any, threshold: float, height: int, width: int) -> list[np.ndarray]:
    instances = get(sample, "pred_instances", {}) if sample is not None else {}
    scores = array(get(instances, "scores"), float).reshape(-1)
    masks = mask_list(get(instances, "masks"), height, width)
    keep = np.flatnonzero(scores >= threshold)
    return [masks[index] for index in keep if index < len(masks)]


def predicted_overlap_masks(sample: Any, threshold: float, height: int, width: int) -> list[np.ndarray]:
    """Predicted occluder layer, saved by the bilayer models as ``overlap_masks``.

    Only S2, S3-P, S3-MBoost and E3 decode a second layer; every other arm
    returns an empty list, so the same call is safe for all eleven methods.
    """
    instances = get(sample, "pred_instances", {}) if sample is not None else {}
    scores = array(get(instances, "scores"), float).reshape(-1)
    masks = mask_list(get(instances, "overlap_masks"), height, width)
    if not masks:
        return []
    keep = np.flatnonzero(scores >= threshold)
    return [masks[index] for index in keep if index < len(masks)]


def hard_route_counts(sample: Any, threshold: float) -> tuple[int, int]:
    """E3 only: how many retained instances the difficulty gate sent the hard way."""
    instances = get(sample, "pred_instances", {}) if sample is not None else {}
    routes = array(get(instances, "mask_hard_routes"))
    if routes.ndim != 1 or routes.shape[0] == 0:
        return (0, 0)
    scores = array(get(instances, "scores"), float).reshape(-1)
    keep = np.flatnonzero(scores >= threshold)
    keep = keep[keep < routes.shape[0]]
    if keep.size == 0:
        return (0, 0)
    return (int(np.asarray(routes[keep]).astype(bool).sum()), int(keep.size))


def occlusion_rank(
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
) -> tuple[list[int], str]:
    ranked = []
    for image_id, info in images.items():
        height, width = int(info["height"]), int(info["width"])
        target_area = overlap_area = 0
        for annotation in annotations.get(image_id, []):
            target_area += int(decode(annotation.get("segmentation"), height, width).sum())
            overlap_area += int(
                decode(annotation.get("bg_object_segmentation"), height, width).sum()
            )
        ranked.append((overlap_area / max(target_area, 1), overlap_area, image_id))
    if any(overlap_area > 0 for _, overlap_area, _ in ranked):
        return (
            [image_id for _, _, image_id in sorted(ranked, reverse=True)],
            "highest overlap ratio",
        )
    target_rank = []
    for image_id, info in images.items():
        height, width = int(info["height"]), int(info["width"])
        target_area = sum(
            int(decode(item.get("segmentation"), height, width).sum())
            for item in annotations.get(image_id, [])
        )
        target_rank.append((target_area, image_id))
    return (
        [image_id for _, image_id in sorted(target_rank, reverse=True)],
        "largest target-mask area",
    )


TARGET_FILL = (20, 70, 220, 105)        # blue: the object being segmented
TARGET_LINE = (0, 0, 0, 255)
OVERLAP_FILL = (230, 130, 20, 120)      # amber: what covers it
OVERLAP_LINE = (150, 70, 0, 255)


def _layer(
    canvas: Image.Image,
    size: tuple[int, int],
    masks: list[np.ndarray],
    fill_rgba: tuple[int, int, int, int],
    line_rgba: tuple[int, int, int, int],
) -> Image.Image:
    if not masks:
        return canvas
    union = np.any(masks, axis=0)
    mask = Image.fromarray((union * 255).astype(np.uint8), mode="L")
    fill = Image.new("RGBA", size, fill_rgba)
    canvas = Image.composite(Image.alpha_composite(canvas, fill), canvas, mask)
    inner = mask.filter(ImageFilter.MinFilter(5))
    boundary = np.asarray(mask, dtype=np.uint8) > np.asarray(inner, dtype=np.uint8)
    outline = Image.fromarray((boundary * 255).astype(np.uint8), mode="L")
    return Image.composite(Image.new("RGBA", size, line_rgba), canvas, outline)


def render_overlay(
    image: Image.Image,
    masks: list[np.ndarray],
    overlap_masks: list[np.ndarray] | None = None,
    fade: float = 0.45,
) -> Image.Image:
    """Draw the target layer, then the occluder layer on top of it.

    The occluder is composited last so that where the two intersect the amber
    reads as covering the blue -- which is the whole point of bilayer decoding.
    Methods without a second decoder simply pass no overlap masks.

    PIXray plates are already saturated green, blue and orange, so translucent
    overlays vanish into them. ``fade`` lightens the plate toward white first;
    the scan stays legible while the mask colours separate from it.
    """
    canvas = image.convert("RGBA")
    if fade > 0:
        canvas = Image.blend(canvas, Image.new("RGBA", image.size, (255, 255, 255, 255)), fade)
    canvas = _layer(canvas, image.size, masks, TARGET_FILL, TARGET_LINE)
    canvas = _layer(canvas, image.size, overlap_masks or [], OVERLAP_FILL, OVERLAP_LINE)
    return canvas.convert("RGB")


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    result.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return result


def main() -> int:
    args = parse_args()
    methods = []
    for value in args.method:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected LABEL=PATH, received: {value}")
        methods.append((label, prediction_index(Path(path))))

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in coco["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    ranked_ids, selection = occlusion_rank(images, annotations)
    image_ids = args.image_ids or ranked_ids[: args.rows]
    if args.image_ids:
        selection = "explicit image IDs"
    image_ids = image_ids[: args.rows]
    columns = [("Ground Truth", None), *methods]
    cell_width, cell_height, title_height = 360, 300, 36
    figure = Image.new(
        "RGB",
        (cell_width * len(columns), (cell_height + title_height) * len(image_ids)),
        "white",
    )
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default()

    for row, image_id in enumerate(image_ids):
        info = images[image_id]
        image = Image.open(args.image_root / info["file_name"]).convert("RGB")
        gt_masks = [
            decode(item.get("segmentation"), image.height, image.width)
            for item in annotations.get(image_id, [])
        ]
        # Ground truth shows both layers too, so a bilayer prediction is compared
        # against the occluder annotation rather than against target masks alone.
        gt_overlap = [
            decode(item.get("bg_object_segmentation"), image.height, image.width)
            for item in annotations.get(image_id, [])
            if item.get("bg_object_segmentation")
        ]
        for column, (label, index) in enumerate(columns):
            caption = label
            if index is None:
                masks, overlap = gt_masks, gt_overlap
            else:
                sample = index.get(image_id)
                masks = predicted_masks(sample, args.score_threshold, image.height, image.width)
                overlap = predicted_overlap_masks(sample, args.score_threshold, image.height, image.width)
                hard, total = hard_route_counts(sample, args.score_threshold)
                if total:
                    caption = f"{label}  (hard {hard}/{total})"
            rendered = fit(render_overlay(image, masks, overlap, args.fade), cell_width, cell_height)
            x = column * cell_width
            y = row * (cell_height + title_height)
            figure.paste(rendered, (x, y + title_height))
            box = draw.textbbox((0, 0), caption, font=font)
            text_width = box[2] - box[0]
            draw.text((x + (cell_width - text_width) / 2, y + 8), caption, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    figure.save(args.output.with_suffix(".pdf"), "PDF", resolution=300)
    args.output.with_suffix(".json").write_text(
        json.dumps({"image_ids": image_ids, "selection": selection}, indent=2),
        encoding="utf-8",
    )
    if args.workbook:
        from openpyxl import load_workbook
        from openpyxl.drawing.image import Image as WorksheetImage

        workbook = load_workbook(args.workbook)
        sheet_name = args.sheet_name[:31]
        if sheet_name in workbook.sheetnames:
            del workbook[sheet_name]
        sheet = workbook.create_sheet(sheet_name)
        sheet["A1"] = f"PIDray instance segmentation: {args.sheet_name}"
        sheet["A2"] = (
            "Blue regions are predicted or ground-truth masks; identical images "
            "are used across methods."
        )
        worksheet_image = WorksheetImage(str(args.output))
        worksheet_image.width = 1200
        worksheet_image.height = int(1200 * figure.height / figure.width)
        sheet.add_image(worksheet_image, "A4")
        workbook.save(args.workbook)
    print(f"Saved qualitative comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
