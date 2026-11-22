"""Detector baseline with occlusion-aware bilayer mask decoding.

Identical to ``pixray_detector_base`` -- same ResNet-101, same cascade bbox
stages, same schedule -- except that the FCN mask head gains a parallel
occluder branch supervised by ``bg_object_segmentation``. The data path moves
to the occlusion track because that is the only way the occluder masks reach
the model; the target annotations themselves are unchanged, the two files
having byte-identical instance lists.

This isolates occluder supervision from the SAM decoder that the published
occlusion-aware models bundle it with.
"""

import os as _os
from pathlib import Path as _Path

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *

# OcLoadAnnotations, OcRandomCrop, OcPackDetInputs and OcDataPreprocessor all
# live in mmdet.occlu; the detector base only imports the project package.
custom_imports = dict(
    imports=["mmdet.occlu", "projects.proposal_first_pidray_occ"],
    allow_failed_imports=False,
)

dataset_ann_root = str(
    _Path(
        _os.getenv(
            "PIXRAYA_ANN_ROOT",
            f"{project_root}/data/PIXray-A/annotations",
        )
    ).expanduser().resolve()
)

# Occluder masks must survive flipping, resizing and cropping alongside the
# target masks, which is what the Oc transforms add.
model["data_preprocessor"]["type"] = "OcDataPreprocessor"
model["roi_head"]["type"] = "BilayerCascadeRoIHead"
model["roi_head"]["mask_head"].update(
    type="BilayerFCNMaskHead",
    # 0.25 is the weight the upstream occlusion-aware head applies to its
    # occluder term, so this arm tests that method's own setting.
    bo_loss_weight=0.25,
    with_residual_guidance=True,
    bo_class_agnostic=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="OcRandomFlip", prob=0.5),
    dict(
        type="RandomResize",
        scale=(512, 512),
        ratio_range=(0.5, 1.5),
        resize_type="OcResize",
        keep_ratio=True,
    ),
    dict(
        type="OcRandomCrop",
        crop_size=(512, 512),
        crop_type="absolute",
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(
        type="FilterAnnotations",
        min_gt_bbox_wh=(1e-5, 1e-5),
        by_mask=True,
        keep_empty=True,
    ),
    dict(type="OcPackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="Pad",
        size=(512, 512),
        pad_val=dict(img=(103.53, 116.28, 123.675), masks=0),
    ),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="OcPackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader["dataset"] = dict(
    type="PIXrayOcclusionDataset",
    data_root=data_root,
    ann_file=f"{dataset_ann_root}/train_occ.json",
    data_prefix=dict(img="train/"),
    metainfo=metainfo,
    pipeline=train_pipeline,
)
test_dataloader["dataset"] = dict(
    type="PIXrayOcclusionDataset",
    data_root=data_root,
    ann_file=f"{dataset_ann_root}/test_occ.json",
    data_prefix=dict(img="test/"),
    metainfo=metainfo,
    test_mode=True,
    pipeline=test_pipeline,
)
# Scored against the target annotations, exactly as every other arm is.

# The parent deletes these; this config re-imports them, so it must too --
# a live module object in the config namespace makes the config undeepcopyable.
del _os, _Path
