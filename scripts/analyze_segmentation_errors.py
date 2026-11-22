#!/usr/bin/env python3
"""Analyze instance-segmentation errors and render representative failures."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--occlusion-annotations", type=Path,
        help="Paired *_occ.json supplying bg_object_segmentation for "
             "occluded/non-occluded recall. Targets still come from --annotations.",
    )
    return parser.parse_args()


def get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if value is None:
        result = np.empty((0,))
    elif hasattr(value, "to_ndarray"):
        result = value.to_ndarray()
    elif hasattr(value, "tensor"):
        result = value.tensor.detach().cpu().numpy()
    elif hasattr(value, "detach"):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    return result.astype(dtype) if dtype is not None else result


def decode(segmentation: Any, height: int, width: int) -> np.ndarray:
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    rle = mask_utils.frPyObjects(segmentation, height, width) if isinstance(segmentation, list) else segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def mask_list(value: Any, height: int, width: int) -> list[np.ndarray]:
    """Normalise a predictions ``masks`` field into a list of boolean arrays.

    ``tools/test.py`` dumps instance masks as COCO RLE dictionaries, not as a
    stacked array. Passing those through ``np.asarray(...).astype(bool)`` yields
    a 0-d object array that is ``True`` for every non-empty dict, which silently
    marks the whole image as foreground: precision collapses, recall reads ~1.0,
    and no prediction ever matches a target. Decode the RLE instead.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if isinstance(value[0], dict):
            return [decode(item, height, width) for item in value]
    data = array(value)
    if data.dtype == object:
        return [decode(item, height, width) for item in data.tolist()]
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        return []
    return [np.asarray(item).astype(bool) for item in data]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def instance_key(annotation: dict[str, Any]) -> tuple:
    """Identify a ground-truth instance by content rather than by id.

    Annotation ids are not shared across the target and occlusion annotation
    tracks, but the instances themselves are identical, so category, box and
    mask together name the same object in both files.
    """
    return (
        int(annotation["image_id"]),
        int(annotation["category_id"]),
        tuple(round(float(value), 4) for value in annotation["bbox"]),
        json.dumps(annotation.get("segmentation"), sort_keys=True),
    )


def size_group(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlay_sample(
    image_path: Path,
    output_path: Path,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    caption: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image).copy()
    gt_union = np.any(gt_masks, axis=0) if gt_masks else np.zeros(pixels.shape[:2], bool)
    pred_union = np.any(pred_masks, axis=0) if pred_masks else np.zeros(pixels.shape[:2], bool)
    pixels[gt_union] = (0.55 * pixels[gt_union] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    pixels[pred_union] = (0.55 * pixels[pred_union] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    rendered = Image.fromarray(pixels)
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, rendered.width, 22), fill="black")
    draw.text((5, 5), caption, fill="white")
    rendered.save(output_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    # Occluded/non-occluded recall needs bg_object_segmentation, which lives in
    # the PIXray-A/PIDray-A track, not in the primary target annotations. Merge
    # only that field so targets and the evaluator stay exactly as the protocol
    # requires while occlusion grouping becomes possible.
    #
    # The two tracks number their annotations differently -- PIXray-A ids are
    # offset by one -- so an id-keyed merge lands 93% of the occluder masks on
    # the wrong instance, 27% of them in a different image entirely. Key on the
    # instance content instead, which is byte-identical across the two files and
    # collision-free on both PIXray splits.
    if args.occlusion_annotations:
        overlay = json.loads(args.occlusion_annotations.read_text(encoding="utf-8"))
        overlap_by_instance = {
            instance_key(item): item.get("bg_object_segmentation")
            for item in overlay.get("annotations", [])
        }
        merged = 0
        for annotation in coco["annotations"]:
            segmentation = overlap_by_instance.get(instance_key(annotation))
            if segmentation:
                annotation["bg_object_segmentation"] = segmentation
                merged += 1
        unmatched = len(coco["annotations"]) - sum(
            1 for a in coco["annotations"] if instance_key(a) in overlap_by_instance
        )
        if unmatched:
            raise SystemExit(
                f"{unmatched} target annotations have no counterpart in "
                f"{args.occlusion_annotations}; the two files must describe the "
                "same instances for occlusion grouping to be meaningful"
            )
        print(f"Merged {merged} occluder masks from {args.occlusion_annotations}")
    images = {int(item["id"]): item for item in coco["images"]}
    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    label_to_category = {index: int(item["id"]) for index, item in enumerate(categories)}
    category_names = {int(item["id"]): item["name"] for item in categories}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        gt_by_image[int(annotation["image_id"])].append(annotation)

    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)

    totals = Counter()
    class_stats: dict[int, Counter] = defaultdict(Counter)
    size_stats: dict[str, Counter] = defaultdict(Counter)
    confusion = Counter()
    error_rows: list[dict[str, Any]] = []
    sample_payloads: list[tuple[float, int, list[np.ndarray], list[np.ndarray], str]] = []

    for sample in predictions:
        image_id = int(get(sample, "img_id", get(get(sample, "metainfo", {}), "img_id", -1)))
        if image_id not in images:
            continue
        info = images[image_id]
        height, width = int(info["height"]), int(info["width"])
        gt_annotations = gt_by_image.get(image_id, [])
        gt_masks = [decode(item.get("segmentation"), height, width) for item in gt_annotations]
        gt_labels = [int(item["category_id"]) for item in gt_annotations]

        instances = get(sample, "pred_instances", {})
        scores = array(get(instances, "scores"), float).reshape(-1)
        labels = array(get(instances, "labels"), int).reshape(-1)
        masks = mask_list(get(instances, "masks"), height, width)
        totals["predictions"] += len(array(get(instances, "scores")))
        totals["predictions_with_mask"] += len(masks)
        keep = np.flatnonzero(scores >= args.score_threshold)
        keep = keep[keep < len(masks)]
        order = keep[np.argsort(scores[keep])[::-1]] if len(keep) else keep
        pred_masks = [masks[index] for index in order]
        pred_labels = [label_to_category.get(int(labels[index]), int(labels[index])) for index in order]

        gt_union = np.any(gt_masks, axis=0) if gt_masks else np.zeros((height, width), bool)
        pred_union = np.any(pred_masks, axis=0) if pred_masks else np.zeros((height, width), bool)
        totals["pixel_tp"] += int(np.logical_and(gt_union, pred_union).sum())
        totals["pixel_fp"] += int(np.logical_and(~gt_union, pred_union).sum())
        totals["pixel_fn"] += int(np.logical_and(gt_union, ~pred_union).sum())
        totals["pixel_tn"] += int(np.logical_and(~gt_union, ~pred_union).sum())

        matched_gt: set[int] = set()
        tp_ious: list[float] = []
        reasons = Counter()
        for pred_mask, pred_label in zip(pred_masks, pred_labels):
            ious = [mask_iou(pred_mask, target) for target in gt_masks]
            same = [idx for idx, label in enumerate(gt_labels) if label == pred_label and idx not in matched_gt]
            best_same = max(same, key=lambda idx: ious[idx]) if same else None
            if best_same is not None and ious[best_same] >= args.match_iou:
                matched_gt.add(best_same)
                value = ious[best_same]
                tp_ious.append(value)
                totals["matched_iou_sum"] += value
                totals["matched_iou_75"] += int(value >= 0.75)
                totals["tp"] += 1
                class_stats[pred_label]["tp"] += 1
                class_stats[pred_label]["iou_sum"] += value
                size_stats[size_group(float(gt_annotations[best_same].get("area", gt_masks[best_same].sum())))]["tp"] += 1
                continue

            totals["fp"] += 1
            class_stats[pred_label]["fp"] += 1
            best = int(np.argmax(ious)) if ious else -1
            best_iou = ious[best] if best >= 0 else 0.0
            if best >= 0 and best_iou >= args.match_iou and gt_labels[best] != pred_label:
                reason = "class_confusion"
                confusion[(gt_labels[best], pred_label)] += 1
            elif best >= 0 and best_iou >= args.match_iou:
                reason = "duplicate_prediction"
            elif best_iou >= 0.10:
                reason = "poor_mask_localization"
            else:
                reason = "background_false_positive"
            reasons[reason] += 1
            totals[reason] += 1

        for index, (annotation, category_id) in enumerate(zip(gt_annotations, gt_labels)):
            class_stats[category_id]["gt"] += 1
            group = size_group(float(annotation.get("area", gt_masks[index].sum())))
            size_stats[group]["gt"] += 1
            overlap = decode(annotation.get("bg_object_segmentation"), height, width).any()
            totals["occluded_gt" if overlap else "non_occluded_gt"] += 1
            if index not in matched_gt:
                totals["fn"] += 1
                class_stats[category_id]["fn"] += 1
                size_stats[group]["fn"] += 1
                totals["occluded_fn" if overlap else "non_occluded_fn"] += 1
                reasons[f"missed_{group}_object"] += 1

        low_quality = sum(value < 0.75 for value in tp_ious)
        severity = 3 * (len(gt_masks) - len(matched_gt)) + 2 * sum(reasons.values()) + low_quality
        dominant = reasons.most_common(1)[0][0] if reasons else "correct"
        error_rows.append({
            "image_id": image_id,
            "file_name": info["file_name"],
            "gt_instances": len(gt_masks),
            "pred_instances": len(pred_masks),
            "true_positives": len(matched_gt),
            "false_negatives": len(gt_masks) - len(matched_gt),
            "false_positives": sum(reasons.values()),
            "mean_matched_mask_iou": float(np.mean(tp_ious)) if tp_ious else 0.0,
            "dominant_error": dominant,
            "severity": severity,
        })
        if severity > 0:
            sample_payloads.append((severity, image_id, gt_masks, pred_masks, dominant))

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    pixel_precision = totals["pixel_tp"] / max(totals["pixel_tp"] + totals["pixel_fp"], 1)
    pixel_recall = totals["pixel_tp"] / max(totals["pixel_tp"] + totals["pixel_fn"], 1)
    pixel_specificity = totals["pixel_tn"] / max(totals["pixel_tn"] + totals["pixel_fp"], 1)
    pixel_dice = 2 * totals["pixel_tp"] / max(
        2 * totals["pixel_tp"] + totals["pixel_fp"] + totals["pixel_fn"], 1
    )
    pixel_iou = totals["pixel_tp"] / max(
        totals["pixel_tp"] + totals["pixel_fp"] + totals["pixel_fn"], 1
    )
    if totals["predictions"] and not totals["predictions_with_mask"]:
        raise SystemExit(
            "No prediction in this file carries a mask. This analysis matches "
            "instances by mask IoU, so every detection would be discarded and "
            "the summary would report zeros. Check that the evaluated model has "
            "a mask branch."
        )

    occluded_recall = (
        1 - totals["occluded_fn"] / totals["occluded_gt"]
        if totals["occluded_gt"]
        else None
    )
    non_occluded_recall = (
        1 - totals["non_occluded_fn"] / totals["non_occluded_gt"]
        if totals["non_occluded_gt"]
        else None
    )
    summary = {
        "score_threshold": args.score_threshold,
        "match_iou": args.match_iou,
        "images": len(error_rows),
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "instance_false_negative_rate": totals["fn"] / max(totals["tp"] + totals["fn"], 1),
        "mean_matched_mask_iou": totals["matched_iou_sum"] / max(totals["tp"], 1),
        "matched_masks_iou_at_least_0_75": totals["matched_iou_75"] / max(totals["tp"], 1),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_specificity": pixel_specificity,
        "pixel_f1_dice": pixel_dice,
        "pixel_iou": pixel_iou,
        "pixel_balanced_accuracy": 0.5 * (pixel_recall + pixel_specificity),
        "pixel_false_positive_rate": 1 - pixel_specificity,
        "pixel_false_negative_rate": 1 - pixel_recall,
        "background_false_positives": totals["background_false_positive"],
        "class_confusions": totals["class_confusion"],
        "duplicate_predictions": totals["duplicate_prediction"],
        "poor_mask_localizations": totals["poor_mask_localization"],
        "occluded_recall": occluded_recall,
        "non_occluded_recall": non_occluded_recall,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "per_image_errors.csv", error_rows)

    per_class = []
    for category_id, name in category_names.items():
        stat = class_stats[category_id]
        per_class.append({
            "category": name,
            "gt": stat["gt"],
            "tp": stat["tp"],
            "fp": stat["fp"],
            "fn": stat["fn"],
            "precision": stat["tp"] / max(stat["tp"] + stat["fp"], 1),
            "recall": stat["tp"] / max(stat["gt"], 1),
            "mean_matched_mask_iou": stat["iou_sum"] / max(stat["tp"], 1),
        })
    write_csv(args.output_dir / "per_class.csv", per_class)

    per_size = []
    for group in ("small", "medium", "large"):
        stat = size_stats[group]
        per_size.append({
            "size": group,
            "gt": stat["gt"],
            "tp": stat["tp"],
            "fn": stat["fn"],
            "recall": stat["tp"] / max(stat["gt"], 1),
        })
    write_csv(args.output_dir / "per_size.csv", per_size)

    confusion_rows = [
        {"ground_truth": category_names.get(gt, str(gt)), "predicted": category_names.get(pred, str(pred)), "count": count}
        for (gt, pred), count in confusion.most_common()
    ]
    write_csv(args.output_dir / "class_confusion.csv", confusion_rows)

    for rank, (_, image_id, gt_masks, pred_masks, reason) in enumerate(
        sorted(sample_payloads, reverse=True)[: args.samples], start=1
    ):
        info = images[image_id]
        overlay_sample(
            args.image_root / info["file_name"],
            sample_dir / f"{rank:02d}_{Path(info['file_name']).stem}_{reason}.png",
            gt_masks,
            pred_masks,
            f"GT=green Pred=red | {reason}",
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
