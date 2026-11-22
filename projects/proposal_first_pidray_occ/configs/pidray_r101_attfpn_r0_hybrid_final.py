"""Paper-style alias for the full-training proposal-first hybrid."""

import os

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

train_dataloader['dataset']['ann_file'] = os.path.join(
    pidraya_ann_root, 'train_occ_runtime.json'
)
val_dataloader = None
val_evaluator = None
val_cfg = None
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=50,
)
default_hooks['checkpoint'] = dict(
    type='CheckpointHook',
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
)
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file=os.path.join(pidraya_ann_root, 'train_occ_runtime.json'),
)]
load_from = None
resume = False
work_dir = './work_dirs/proposal_first_pidray_occ_final'
