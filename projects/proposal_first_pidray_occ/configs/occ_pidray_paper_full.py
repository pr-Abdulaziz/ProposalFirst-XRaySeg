"""Occ architecture on the official manual PIDray-A annotations."""

import os

_base_ = ['./occ_pidray_env.py']

data_root = os.getenv('OCC_PIDRAY_DATA_ROOT', 'data')
pidraya_ann_root = os.getenv(
    'PIDRAYA_ANN_ROOT', f'{data_root}/PIDray-A/annotations'
)

train_dataloader = dict(dataset=dict(
    ann_file=os.path.join(pidraya_ann_root, 'train_occ_runtime.json'),
))
val_dataloader = None
val_evaluator = None
val_cfg = None
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=50,
)
default_hooks = dict(checkpoint=dict(
    _delete_=True,
    type='CheckpointHook',
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
))
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file=os.path.join(pidraya_ann_root, 'train_occ_runtime.json'),
)]
load_from = None
resume = False
work_dir = './work_dirs/occ_pidray_paper_full'
