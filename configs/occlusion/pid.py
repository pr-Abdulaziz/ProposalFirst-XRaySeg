"""Occ PIDray-A baseline with local path-safe defaults."""

import os

_base_ = ["base.py"]

work_dir = "./work_dirs/pid"
EXPERIMENT_SEED = 42
randomness = dict(
    seed=EXPERIMENT_SEED,
    deterministic=False,
    diff_rank_seed=False,
)

custom_imports = dict(
    imports=["mmdet.occlu", "projects.proposal_first_pidray_occ"],
    allow_failed_imports=False,
)

# Change only these paths when the project is moved to another machine.
data_root = os.path.abspath(os.getenv("OCC_PIDRAY_DATA_ROOT", "data"))
pidraya_ann_root = os.path.abspath(os.getenv(
    "PIDRAYA_ANN_ROOT", f"{data_root}/PIDray-A/annotations"
))
sam_model_dir = os.path.abspath(os.getenv(
    "SAM_VIT_LARGE_DIR", "pretrained/sam_vit_large"
))
sam_checkpoint = os.path.abspath(os.getenv(
    "SAM_VIT_LARGE_CKPT", f"{sam_model_dir}/pytorch_model_new.bin"
))
subset_ann_files = {
    split: os.path.join(pidraya_ann_root, filename)
    for split, filename in (
        ("easy", "test_easy_occ.json"),
        ("hard", "test_hard_occ.json"),
        ("hidden", "test_hidden_occ.json"),
    )
}

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    progress=dict(type="TqdmProgressHook", mininterval=1.0, leave=False),
)

num_classes = 12
prompt_shape = (70, 5)

crop_size = (512, 512)

batch_augments = [
    dict(
        type="OcBatchFixedSizePad",
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )
]

data_preprocessor = dict(
    type="OcDataPreprocessor",
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments,
)

model = dict(
    decoder_freeze=False,
    data_preprocessor=data_preprocessor,
    shared_image_embedding=dict(
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
    ),
    backbone=dict(
        _delete_=True,
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
    ),
    neck=dict(
        feature_aggregator=dict(
            _delete_=True,
            type="OcFeatureAggregator",
            in_channels=256,
            hidden_channels=512,
            out_channels=256,
        ),
    ),
    roi_head=dict(
        type="OcPrompterAnchorRoIPromptHead",
        bbox_head=dict(
            num_classes=num_classes,
        ),
        mask_head=dict(
            type="OcPrompterAnchorMaskHead",
            mask_decoder=dict(
                type="OcRSSamMaskDecoder",
                hf_pretrain_name=sam_model_dir,
                init_cfg=dict(type="Pretrained", checkpoint=sam_checkpoint),
            ),
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            loss_mask_bo=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=crop_size,
        )
    ),
)


backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args, to_float32=True),
    dict(type="OcLoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="OcRandomFlip", prob=0.5),
    dict(
        type="RandomResize",
        scale=crop_size,
        ratio_range=(0.1, 2.0),
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
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
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
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
        ),
    ),
]

dataset_type = "PIDrayOcclusionDataset"

CLASSES = (
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

batch_size_per_gpu = 2
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=os.path.join(pidraya_ann_root, "train_occ_runtime.json"),
        data_prefix=dict(img="train/"),
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file=os.path.join(pidraya_ann_root, "test_occ.json"),
        data_prefix=dict(img="test/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_dataloader = None
val_evaluator = None
val_cfg = None
test_evaluator = dict(
    type="PIDrayPaperMetric",
    ann_file=os.path.join(pidraya_ann_root, "test_occ.json"),
    metric=["bbox", "segm"],
    classwise=True,
    subset_ann_files=subset_ann_files,
)

base_lr = 0.0001
max_epochs = 50

train_cfg = dict(
    _delete_=True,
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
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
)

custom_hooks = [dict(
    type="PIDrayCheckpointProvenanceHook",
    training_scope="official_full_train",
    annotation_file=os.path.join(pidraya_ann_root, "train_occ_runtime.json"),
)]
load_from = None
resume = False
