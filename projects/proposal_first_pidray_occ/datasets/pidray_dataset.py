"""PIDray dataset metadata and local archive path handling."""

import os.path as osp
from typing import ClassVar

from mmdet.datasets import CocoDataset
from mmdet.occlu.occlu import OcCocoDataset
from mmdet.registry import DATASETS

PIDRAY_CLASSES = (
    "Baton",
    "Pliers",
    "Hammer",
    "Powerbank",
    "Scissors",
    "Wrench",
    "Gun",
    "Bullet",
    "Sprayer",
    "HandCuffs",
    "Knife",
    "Lighter",
)

PIDRAY_PALETTE = (
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
)

PIXRAY_CLASSES = (
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

PIXRAY_PALETTE = (
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
)


class _PIDrayShardPathMixin:
    """Resolve the journal archive's ``train/train0`` and ``train/train1``.

    The official COCO files contain bare names such as ``xray_00000.png``,
    while the distributed journal archive divides the training images between
    two directories.  Keep the source JSON unchanged and repair only the
    runtime ``img_path`` produced by the parent COCO dataset parser.
    """

    _TRAIN_SHARDS = ("train0", "train1")

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        img_path = data_info["img_path"]
        if osp.isfile(img_path):
            return data_info

        image_root = osp.dirname(img_path)
        file_name = osp.basename(img_path)
        matches = [
            osp.join(image_root, shard, file_name)
            for shard in self._TRAIN_SHARDS
            if osp.isfile(osp.join(image_root, shard, file_name))
        ]
        if len(matches) == 1:
            data_info["img_path"] = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                f"PIDray image appears in multiple training shards: {file_name}"
            )
        return data_info


@DATASETS.register_module()
class PIDrayDataset(_PIDrayShardPathMixin, CocoDataset):
    """COCO-format PIDray data, including images with no prohibited item.

    Set ``filter_cfg=dict(filter_empty_gt=False, min_size=0)`` in every
    training dataloader.  This is required for the PIDray R0 normal-image
    supervision and replaces the released code's globally disabled filtering.
    """

    METAINFO: ClassVar[dict] = {
        "classes": PIDRAY_CLASSES,
        "palette": PIDRAY_PALETTE,
    }


@DATASETS.register_module()
class PIDrayOcclusionDataset(_PIDrayShardPathMixin, OcCocoDataset):
    """Occ-format PIDray annotations with target and overlap masks.

    The current ``*_occ.json`` files are algorithmically derived from PIDray
    instance polygons.  This class does not imply that they are the authors'
    manually annotated PIDray-A release.

    Every segmented instance must provide ``bg_object_segmentation`` in the
    COCO annotation. Normal images may contain an empty annotation list and
    are retained for PIDray R0 supervision.
    """

    METAINFO: ClassVar[dict] = {
        "classes": PIDRAY_CLASSES,
        "palette": PIDRAY_PALETTE,
    }


@DATASETS.register_module()
class PIXrayDataset(CocoDataset):
    """The 15-category COCO-format PIXray instance dataset."""

    METAINFO: ClassVar[dict] = {
        "classes": PIXRAY_CLASSES,
        "palette": PIXRAY_PALETTE,
    }


@DATASETS.register_module()
class PIXrayOcclusionDataset(OcCocoDataset):
    """Official PIXray-A annotations with occluder-overlap masks."""

    METAINFO: ClassVar[dict] = {
        "classes": PIXRAY_CLASSES,
        "palette": PIXRAY_PALETTE,
    }
