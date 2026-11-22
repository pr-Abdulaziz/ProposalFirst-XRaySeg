"""Development arm A: Method 03, no point refinement.

Reference arm of the three-way development comparison. Identical to Method 03
in architecture; the only differences from ``pixray_p2_cascade_attfpn`` are the
development split, the dev-val validation loop and the removal of the official
test loader.

This arm is *retrained* on dev-train. The historical Method 03 checkpoint
cannot serve as the development baseline: it saw all 3560 training images,
including the 534 dev-val images, so comparing a dev-train arm against it would
compare two different training sets rather than two architectures.

A vs B isolates the effect of adding point refinement at all.

All three development arms share one overlay, generated from a single template by
scripts/write_dev_configs.py, so they can differ only in the architecture they
inherit. Data, schedule, optimiser, batch, augmentation, evaluation and
checkpoint policy are identical by construction.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_p2_cascade_attfpn import *

import os as _os
from pathlib import Path as _Path

# The split lives outside data_root on purpose: the released dataset directory
# stays exactly as distributed, and the split is an external artefact that can
# be regenerated from scripts/build_dev_split.py and checked against its
# manifest. Absolute paths, so ann_file is not joined onto data_root.
dev_split_root = str(
    _Path(
        _os.getenv("PIXRAY_DEV_SPLIT_ROOT", f"{project_root}/splits/pixray_dev_v1")
    ).expanduser().resolve()
)

uses_validation = True

# dev-train only. The 534 dev-val images are absent from training for every arm.
train_dataloader["dataset"]["ann_file"] = f"{dev_split_root}/dev_train.json"

# dev-val images are physically in data/PIXray/train/, so data_prefix is "train/".
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PIXrayDataset",
        data_root=data_root,
        ann_file=f"{dev_split_root}/dev_val.json",
        data_prefix=dict(img="train/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
# Model selection uses mask AP on dev-val and nothing else. Runtime metrics
# belong to the test protocol; measuring them here would only add noise to a
# selection signal. dev_val.json happens to contain no annotation with id 0, so
# the compat keys coincide with the official ones -- enabled anyway so the same
# evaluator code path runs everywhere.
val_evaluator = dict(
    type="XRayCocoMetric",
    ann_file=f"{dev_split_root}/dev_val.json",
    metric=["bbox", "segm"],
    classwise=True,
    zero_id_compat=True,
)
val_cfg = dict(type="ValLoop")
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=2)

# A development run must be structurally incapable of reading the official test
# split. Removing the loader and the loop is that guarantee; it is not a matter
# of remembering to pass the right subcommand.
test_dataloader = None
test_evaluator = None
test_cfg = None

default_hooks["checkpoint"] = dict(
    type="CheckpointHook",
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
    save_best="coco/segm_mAP",
    rule="greater",
)

# Every arm starts from the same ImageNet ResNet-101 initialisation. The
# historical epoch-60 checkpoints were trained on all 3560 training images,
# including the 534 now held out, so using one as a development starting point
# would leak dev-val into all three arms before the first step.
load_from = None
resume = False

del _os, _Path

work_dir = f"{project_root}/work_dirs/pixray_dev_a_cascade_attfpn"
