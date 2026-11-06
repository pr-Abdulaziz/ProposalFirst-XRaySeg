"""S2: Occ-PIDray bilayer SAM under the shared PIDray-30 protocol."""

import os as _os
from pathlib import Path as _Path

_base_ = ["pid.py"]

custom_imports = dict(
    imports=["mmdet.occlu", "projects.proposal_first_pidray_occ"],
    allow_failed_imports=False,
)

project_root = str(
    _Path(
        _os.getenv(
            "XRAY_PROJECT_ROOT",
            str(_Path("{{fileDirname}}").resolve().parents[1]),
        )
    ).expanduser().resolve()
)
data_root = str(
    _Path(_os.getenv("PIDRAY_DATA_ROOT", f"{project_root}/data/PIDray"))
    .expanduser().resolve()
)
dataset_ann_root = str(
    _Path(
        _os.getenv(
            "PIDRAYA_ANN_ROOT", f"{project_root}/data/PIDray-A/annotations"
        )
    ).expanduser().resolve()
)
sam_model_dir = str(
    _Path(
        _os.getenv("SAM_VIT_LARGE_DIR", f"{project_root}/pretrained/sam_vit_large")
    ).expanduser().resolve()
)
sam_checkpoint = str(
    _Path(
        _os.getenv(
            "SAM_VIT_LARGE_CKPT", f"{sam_model_dir}/pytorch_model_new.bin"
        )
    ).expanduser().resolve()
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
test_split_ann_files = {
    split: f"{dataset_ann_root}/test_{split}_occ.json"
    for split in ("easy", "hard", "hidden")
}
metric_split_ann_files = {
    split: f"{data_root}/annotations/test_{split}.json"
    for split in ("easy", "hard", "hidden")
}

CLASSES = (
    "Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
    "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter",
)
model = dict(
    shared_image_embedding=dict(
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
    ),
    backbone=dict(
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
    ),
    roi_head=dict(mask_head=dict(mask_decoder=dict(
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
    ))),
)

# Match every other primary experiment's augmentation and empty-image policy.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None, to_float32=True),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="OcRandomFlip", prob=0.5),
    dict(
        type="RandomResize", scale=(512, 512), ratio_range=(0.5, 1.5),
        resize_type="OcResize", keep_ratio=True,
    ),
    dict(
        type="OcRandomCrop", crop_size=(512, 512), crop_type="absolute",
        recompute_bbox=True, allow_negative_crop=True,
    ),
    dict(
        type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5),
        by_mask=True, keep_empty=True,
    ),
    dict(type="OcPackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None, to_float32=True),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="Pad", size=(512, 512),
        pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0),
    ),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="OcPackDetInputs",
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
        type="PIDrayOcclusionDataset",
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/train_occ.json",
        data_prefix=dict(img="train/"),
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
        type="PIDrayOcclusionDataset",
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=test_split_ann_files["easy"],
        data_prefix=dict(img="test/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
val_dataloader = None
val_evaluator = None
val_cfg = None
test_evaluator = dict(
    _delete_=True,
    type="XRayCocoMetric",
    ann_file=metric_split_ann_files["easy"],
    metric=["bbox", "segm", "proposal"],
    classwise=True,
)

base_lr = 1e-4
max_epochs = 60
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type="CosineAnnealingLR", eta_min=1e-7, begin=1, end=max_epochs,
        T_max=max_epochs, by_epoch=True,
    ),
]
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=accumulation_steps,
)
auto_scale_lr = dict(enable=False, base_batch_size=effective_batch_size)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    progress=dict(type="TqdmProgressHook", mininterval=1.0, leave=False),
)
custom_hooks = [dict(
    type="PIDrayCheckpointProvenanceHook",
    training_scope="deterministic_pidray30_seed42",
    annotation_file=f"{dataset_ann_root}/train_occ.json",
)]
load_from = None
resume = False
work_dir = f"{project_root}/work_dirs/pidray30_s2"

del _os, _Path
