"""Shared proposal-first bilayer model on official PIXray-A annotations."""

import os as _os
from pathlib import Path as _Path

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

project_root = str(
    _Path(
        _os.getenv(
            "XRAY_PROJECT_ROOT",
            str(_Path("{{fileDirname}}").resolve().parents[2]),
        )
    ).expanduser().resolve()
)
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
uses_validation = False
subset_ann_files = {}
metainfo = dict(
    classes=(
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
)

# PIXray-A contains no normal-image negatives. Keep the R0 module only as a
# zero-weight compatibility feature and never use it for training or inference.
model["training_gate"] = "none"
model["gate_during_inference"] = False
model["use_r0_screening"] = False
model["r0_head"]["loss_weight"] = 0.0
model["roi_head"]["mask_head"]["overlap_loss_weight"] = 1.0
model["sam_image_encoder"]["hf_pretrain_name"] = sam_model_dir
model["sam_image_encoder"]["init_cfg"]["checkpoint"] = sam_checkpoint
model["sam_positional_embedding"]["hf_pretrain_name"] = sam_model_dir
model["sam_positional_embedding"]["init_cfg"]["checkpoint"] = sam_checkpoint
model["roi_head"]["mask_head"]["mask_decoder"][
    "hf_pretrain_name"
] = sam_model_dir
model["roi_head"]["mask_head"]["mask_decoder"]["init_cfg"][
    "checkpoint"
] = sam_checkpoint
for bbox_head in model["roi_head"]["bbox_head"]:
    bbox_head["num_classes"] = 15
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
        type="PIXrayOcclusionDataset",
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/train_occ.json",
        data_prefix=dict(img="train/"),
        metainfo=metainfo,
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
        type="PIXrayOcclusionDataset",
        data_root=data_root,
        ann_file=f"{dataset_ann_root}/test_occ.json",
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
        type="XRayCocoMetric",
        ann_file=f"{evaluator_ann_root}/test.json",
        metric=["bbox", "segm", "proposal"],
        classwise=True,
    ),
    dict(type="ProposalFirstRuntimeMetric", detection_threshold=0.5, timing_warmup=50),
]

max_epochs = 60
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs)
param_scheduler[1].update(end=max_epochs, T_max=max_epochs)
optim_wrapper["accumulative_counts"] = accumulation_steps
auto_scale_lr = dict(enable=False, base_batch_size=effective_batch_size)
default_hooks["logger"]["interval"] = 50
custom_hooks = []
load_from = None
resume = False

# Stage-level inference timings (rpn / sam / decoder breakdown) are recorded
# only when profiling is switched on. It stays off by default so the training
# graph is identical to the runs already completed, and the evaluation stage
# turns it on through XRAY_PROFILE_INFERENCE.
model["profile_inference"] = _os.getenv("XRAY_PROFILE_INFERENCE", "0") == "1"

del _os, _Path
