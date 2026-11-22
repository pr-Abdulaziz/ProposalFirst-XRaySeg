"""Shared 12-class detector protocol for the deterministic PIDray-30 track."""

import os as _os
from pathlib import Path as _Path

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

project_root = str(
    _Path(
        _os.getenv(
            "XRAY_PROJECT_ROOT",
            str(_Path(__file__).resolve().parents[3]),
        )
    ).expanduser().resolve()
)
data_root = str(
    _Path(_os.getenv("PIDRAY_DATA_ROOT", f"{project_root}/data/PIDray"))
    .expanduser().resolve()
)
physical_batch_size = int(_os.getenv("XRAY_BATCH_SIZE_PER_GPU", "2"))
effective_batch_size = int(_os.getenv("XRAY_EFFECTIVE_BATCH_SIZE", "16"))
train_num_workers = int(_os.getenv("XRAY_NUM_WORKERS", "4"))
if physical_batch_size <= 0 or effective_batch_size % physical_batch_size:
    raise ValueError(
        "XRAY_BATCH_SIZE_PER_GPU must be positive and evenly divide "
        "XRAY_EFFECTIVE_BATCH_SIZE"
    )
accumulation_steps = effective_batch_size // physical_batch_size
uses_validation = False
metainfo = dict(classes=(
    "Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
    "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter",
))
test_split_ann_files = {
    split: f"{data_root}/annotations/test_{split}.json"
    for split in ("easy", "hard", "hidden")
}
metric_split_ann_files = dict(test_split_ann_files)

# P1--P4 are ordinary Cascade Mask R-CNN detector baselines. R0 belongs only
# to the proposal-first family and is removed here to isolate architecture.
model["type"] = "CascadeRCNN"
model.pop("r0_head", None)
model.pop("training_gate", None)
model.pop("gate_during_inference", None)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomResize", scale=(512, 512), ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop", crop_size=(512, 512), crop_type="absolute",
        recompute_bbox=True, allow_negative_crop=True,
    ),
    dict(
        type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5),
        by_mask=True, keep_empty=True,
    ),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="Pad", size=(512, 512),
        pad_val=dict(img=(103.53, 116.28, 123.675), masks=0),
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id", "img_path", "ori_shape", "img_shape", "pad_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=physical_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="PIDrayDataset",
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PIDrayDataset",
        data_root=data_root,
        ann_file=test_split_ann_files["easy"],
        data_prefix=dict(img="test/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
val_dataloader = None
val_evaluator = None
val_cfg = None
# Mirrors the PIXray detector base: the baselines carry the same image-level
# metric as the hybrids so no row of the comparison table is left empty.
test_evaluator = [
    dict(
        type="XRayCocoMetric",
        ann_file=test_split_ann_files["easy"],
        metric=["bbox", "segm", "proposal"],
        classwise=True,
    ),
    dict(
        type="ProposalFirstRuntimeMetric",
        r0_threshold=0.5,
        detection_threshold=0.5,
        timing_warmup=50,
    ),
]

base_lr = 1e-4
max_epochs = 60
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=accumulation_steps,
)
auto_scale_lr = dict(enable=False, base_batch_size=effective_batch_size)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type="CosineAnnealingLR", eta_min=1e-7, begin=1, end=max_epochs,
        T_max=max_epochs, by_epoch=True,
    ),
]
default_hooks["checkpoint"] = dict(
    type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True
)
default_hooks["logger"] = dict(type="LoggerHook", interval=50)
default_hooks["progress"] = dict(
    type="TqdmProgressHook", mininterval=1.0, leave=False
)
custom_hooks = [dict(
    type="PIDrayCheckpointProvenanceHook",
    training_scope="deterministic_pidray30_seed42",
    annotation_file=f"{data_root}/annotations/train.json",
)]
load_from = None
resume = False

del _os, _Path
