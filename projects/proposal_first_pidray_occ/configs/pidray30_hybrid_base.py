"""Shared proposal-first protocol on paired PIDray-30/PIDray-A records."""

import os as _os
from pathlib import Path as _Path

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

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
metainfo = dict(classes=(
    "Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench",
    "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter",
))
test_split_ann_files = {
    split: f"{dataset_ann_root}/test_{split}_occ.json"
    for split in ("easy", "hard", "hidden")
}
metric_split_ann_files = {
    split: f"{data_root}/annotations/test_{split}.json"
    for split in ("easy", "hard", "hidden")
}

# R0 learns from the retained normal images, but never gates positive RoIs
# during training. This protects mask supervision while preserving the
# proposal-first inference mechanism.
model.update(
    training_gate="none",
    gate_during_inference=False,
    use_r0_screening=True,
    use_proposal_screening=True,
    r0_rescue_enabled=False,
    sam_encoder_frozen=False,
    sam_decoder_frozen=False,
)
model["r0_head"]["loss_weight"] = 0.1
model["sam_image_encoder"]["hf_pretrain_name"] = sam_model_dir
model["sam_image_encoder"]["init_cfg"]["checkpoint"] = sam_checkpoint
model["sam_positional_embedding"]["hf_pretrain_name"] = sam_model_dir
model["sam_positional_embedding"]["init_cfg"]["checkpoint"] = sam_checkpoint
model["roi_head"]["mask_head"]["mask_decoder"]["hf_pretrain_name"] = sam_model_dir
model["roi_head"]["mask_head"]["mask_decoder"]["init_cfg"][
    "checkpoint"
] = sam_checkpoint
model["roi_head"]["mask_head"]["overlap_loss_weight"] = 1.0
model["roi_head"].update(
    use_prompt_adapter=False,
    adapter_hidden_channels=32,
    use_mask_booster=False,
    mask_booster_hidden_channels=8,
    mask_booster_strength=1.0,
    hard_aware_loss=False,
    dice_loss_weight=1.0,
    occlusion_loss_weight=1.0,
    boundary_loss_weight=0.25,
    conditional_bisam=False,
    difficulty_threshold=0.05,
    difficulty_gate_threshold=0.5,
    difficulty_loss_weight=0.25,
    difficulty_hidden_channels=64,
)

train_dataloader = dict(
    batch_size=physical_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="PIDrayOcclusionDataset",
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/train_occ.json",
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
        type="PIDrayOcclusionDataset",
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
test_evaluator = [
    dict(
        type="XRayCocoMetric", ann_file=metric_split_ann_files["easy"],
        metric=["bbox", "segm", "proposal"], classwise=True,
    ),
    dict(
        type="ProposalFirstRuntimeMetric", r0_threshold=0.5,
        detection_threshold=0.5, timing_warmup=50,
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
    annotation_file=f"{dataset_ann_root}/train_occ.json",
)]
load_from = None
resume = False

# Stage-level inference timings are a test-time measurement only, so the
# training graph is unaffected. The evaluation stage sets XRAY_PROFILE_INFERENCE.
model["profile_inference"] = _os.getenv("XRAY_PROFILE_INFERENCE", "0") == "1"

del _os, _Path
