"""Path-safe wrapper around the released Occ PIDray configuration.

Environment variables:
    OCC_PIDRAY_DATA_ROOT: PIDray root with annotations/train/test.
    SAM_VIT_LARGE_DIR: local Hugging Face SAM ViT-L directory.
    SAM_VIT_LARGE_CKPT: converted SAM checkpoint; defaults inside the directory.
"""

import os

_base_ = [
    '../../../configs/occlusion/pid.py',
]

custom_imports = dict(
    imports=['mmdet.occlu', 'projects.proposal_first_pidray_occ'],
    allow_failed_imports=False,
)

data_root = os.getenv('OCC_PIDRAY_DATA_ROOT', 'data')
pidraya_ann_root = os.getenv(
    'PIDRAYA_ANN_ROOT', f'{data_root}/PIDray-A/annotations'
)
subset_ann_files = {
    split: os.path.abspath(os.getenv(env_name, os.path.join(
        pidraya_ann_root, filename)))
    for split, env_name, filename in (
        ('easy', 'PIDRAY_EASY_ANN', 'test_easy_occ.json'),
        ('hard', 'PIDRAY_HARD_ANN', 'test_hard_occ.json'),
        ('hidden', 'PIDRAY_HIDDEN_ANN', 'test_hidden_occ.json'),
    )
}

train_dataloader = dict(dataset=dict(
    type='PIDrayOcclusionDataset',
    data_root=data_root,
    ann_file=os.path.join(pidraya_ann_root, 'train_occ_runtime.json'),
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=0),
))
val_dataloader = None
test_dataloader = dict(dataset=dict(
    type='PIDrayOcclusionDataset',
    data_root=data_root,
    ann_file=os.path.join(pidraya_ann_root, 'test_occ.json'),
    data_prefix=dict(img='test/'),
    test_mode=True,
))
val_evaluator = None
val_cfg = None
test_evaluator = dict(
    type='PIDrayPaperMetric',
    ann_file=os.path.join(pidraya_ann_root, 'test_occ.json'),
    metric=['bbox', 'segm'],
    classwise=True,
    subset_ann_files=subset_ann_files,
)
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file=os.path.join(pidraya_ann_root, 'train_occ_runtime.json'),
)]
work_dir = './work_dirs/occ_pidray_env'
