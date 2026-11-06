"""Primary PIXray-A Occ baseline.

This preserves the authors' PIXray architecture and 60-epoch protocol. The
images and occlusion annotations are separate in the local workspace:

    <project>/data/PIXray/{train,test}
    <project>/data/PIXray-A/annotations/{train_occ,test_occ}.json

This is the default executable experiment after removing PIDray from the data
protocol. Override ``PIXRAY_DATA_ROOT`` and ``PIXRAYA_ANN_ROOT`` only when the
dataset is mounted elsewhere. ``OCC_PIXRAY_CKPT`` is optional and is used only
for checkpoint evaluation or explicitly reported fine-tuning.
"""

import os as _os
from pathlib import Path as _Path

from mmengine.config import read_base

with read_base():
    from .base import *

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
work_dir = f"{project_root}/work_dirs/pixray_s2"
code_root = project_root
dataset_type = "PIXrayOcclusionDataset"
EXPERIMENT_SEED = 42
randomness = dict(
    seed=EXPERIMENT_SEED,
    deterministic=False,
    diff_rank_seed=False,
)
uses_validation = False
# Defaults move with the project folder. Environment overrides support datasets
# or pretrained weights mounted elsewhere without editing this config.
data_root = str(
    _Path(_os.getenv("PIXRAY_DATA_ROOT", f"{project_root}/data/PIXray"))
    .expanduser()
    .resolve()
)
dataset_ann_root = str(
    _Path(
        _os.getenv(
            "PIXRAYA_ANN_ROOT",
            f"{project_root}/data/PIXray-A/annotations",
        )
    ).expanduser().resolve()
)
evaluator_ann_root = f"{data_root}/annotations"
sam_model_dir = str(
    _Path(
        _os.getenv(
            "SAM_VIT_LARGE_DIR",
            f"{project_root}/pretrained/sam_vit_large",
        )
    ).expanduser().resolve()
)
sam_checkpoint = str(
    _Path(
        _os.getenv(
            "SAM_VIT_LARGE_CKPT",
            f"{sam_model_dir}/pytorch_model_new.bin",
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


default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    progress=dict(type="TqdmProgressHook", mininterval=1.0, leave=False),
)

num_classes = 15
prompt_shape = (70, 5)
crop_size = (512, 512)

batch_augments = [dict(
    type="OcBatchFixedSizePad",
    size=crop_size,
    img_pad_value=0,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
)]

data_preprocessor = dict(
    type="OcDataPreprocessor",
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments,
)

model["decoder_freeze"] = False
model["data_preprocessor"] = data_preprocessor
model["shared_image_embedding"] = dict(
    type="RSSamPositionalEmbedding",
    hf_pretrain_name=sam_model_dir,
    init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
)
model["backbone"] = dict(
    img_size=crop_size[0],
    type="MMPretrainSamVisionEncoder",
    hf_pretrain_name=sam_model_dir,
    init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
    peft_config=dict(
        peft_type="LORA",
        r=16,
        target_modules=["qkv"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    ),
)
model["neck"]["feature_aggregator"] = dict(
    type="OcFeatureAggregator",
    in_channels=256,
    hidden_channels=512,
    out_channels=256,
)
model["roi_head"]["type"] = "OcPrompterAnchorRoIPromptHead"
model["roi_head"]["bbox_head"]["num_classes"] = num_classes
mask_head = model["roi_head"]["mask_head"]
mask_head["type"] = "OcPrompterAnchorMaskHead"
mask_head["mask_decoder"] = dict(
    type="OcRSSamMaskDecoder",
    hf_pretrain_name=sam_model_dir,
    init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
)
mask_head["per_pointset_point"] = prompt_shape[1]
mask_head["with_sincos"] = True
mask_head["loss_mask"] = dict(
    type="CrossEntropyLoss", use_mask=True, loss_weight=1.0
)
# The upstream head never builds a separate occluder-loss object: the config
# entry that used to sit here was swallowed by **kwargs and self.loss_mask_bo
# is referenced nowhere. What is actually applied is self.loss_mask scaled by
# a hardcoded 0.25. WeightedOcPrompterAnchorMaskHead exposes that scalar so
# the configuration states what the model does; the default reproduces the
# released behaviour exactly, so this checkpoint remains valid.
mask_head["type"] = "WeightedOcPrompterAnchorMaskHead"
mask_head["bo_loss_weight"] = 0.25
model["train_cfg"]["rcnn"]["mask_size"] = crop_size

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args, to_float32=True),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="OcRandomFlip", prob=0.5),
    dict(
        type="RandomResize",
        scale=crop_size,
        # Controlled comparison: match P1/P2/S1/S3 augmentation strength.
        ratio_range=(0.5, 1.5),
        resize_type="OcResize",
        keep_ratio=True,
    ),
    dict(
        type="OcRandomCrop",
        crop_size=crop_size,
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
    dict(type="LoadImageFromFile", backend_args=backend_args, to_float32=True),
    dict(type="Resize", scale=crop_size, keep_ratio=True),
    dict(
        type="Pad",
        size=crop_size,
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

CLASSES = (
    "Bat", "Battery", "Dart", "Fireworks", "Gun", "Hammer", "Knife",
    "Lighter", "Pliers", "Pressure_vessel", "Razor_blade", "Saw_blade",
    "Scissors", "Screwdriver", "Wrench",
)

batch_size_per_gpu = physical_batch_size
num_workers = train_num_workers
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type="PIXrayOcclusionDataset",
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/train_occ.json",
        data_prefix=dict(img="train/"),
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    # Keep inference timing and memory comparable across every experiment.
    batch_size=1,
    num_workers=2,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PIXrayOcclusionDataset",
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/test_occ.json",
        data_prefix=dict(img="test/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
val_dataloader = None
val_evaluator = None
val_cfg = None

test_evaluator = [
    dict(
        type="XRayCocoMetric",
        ann_file=f"{evaluator_ann_root}/test.json",
        metric=["bbox", "segm", "proposal"],
        classwise=True,
        # Adds compat_zeroid/* keys beside the official ones. Official
        # values are unchanged - see tests/test_zero_id_compat.py.
        zero_id_compat=True,
    ),
    # S2 is a different model family, but it is scored with the same metric set
    # as every other arm so the comparison table has no empty rows.
    dict(
        type="ProposalFirstRuntimeMetric",
        detection_threshold=0.5,
        timing_warmup=50,
    ),
]

load_from = None
resume = False

base_lr = 0.0001
max_epochs = 60
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
    ),
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=accumulation_steps,
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
)
auto_scale_lr = dict(enable=False, base_batch_size=effective_batch_size)

del _os, _Path
