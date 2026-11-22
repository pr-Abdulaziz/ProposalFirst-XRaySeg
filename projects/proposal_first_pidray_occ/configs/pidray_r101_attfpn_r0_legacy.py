"""PIDray R0/R1 reproduction on the repository's MMDetection 3.2 runtime.

Set ``PIDRAY_DATA_ROOT`` to the directory containing ``annotations/``,
``train/``, and ``test/``.  No dataset or checkpoint path is hard-coded.
"""

from mmengine.config import read_base

with read_base():
    from ....mmdet.configs._base_.default_runtime import *
    from ....mmdet.configs._base_.models.cascade_mask_rcnn_r50_fpn import *

custom_imports = dict(
    imports=['projects.proposal_first_pidray_occ'],
    allow_failed_imports=False,
)
default_scope = 'mmdet'

dataset_type = 'PIDrayDataset'
data_root = 'data/'
EXPERIMENT_SEED = 42
randomness = dict(
    seed=EXPERIMENT_SEED,
    deterministic=False,
    diff_rank_seed=False,
)
subset_ann_files = {
    'easy': f'{data_root}annotations/test_easy.json',
    'hard': f'{data_root}annotations/test_hard.json',
    'hidden': f'{data_root}annotations/test_hidden.json',
}
metainfo = dict(classes=(
    'Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench',
    'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife', 'Lighter',
))

_model_overrides = dict(
    type='PIDrayCascadeRCNN',
    training_gate='legacy',
    gate_during_inference=True,
    r0_head=dict(
        type='PIDrayR0Head',
        hidden_dim=256,
        threshold=0.5,
        loss_weight=0.1,
        loss_type='bce',
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    neck=dict(type='AttFPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=12,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=12,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=12,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(num_classes=12),
    ),
)

# Pure-Python MMEngine configs use ordinary Python update semantics. Preserve
# the complete Cascade Mask R-CNN base structure while applying the PIDray
# changes that the legacy dict-merging syntax previously supplied.
model.update(
    type=_model_overrides['type'],
    training_gate=_model_overrides['training_gate'],
    gate_during_inference=_model_overrides['gate_during_inference'],
    r0_head=_model_overrides['r0_head'],
)
model['data_preprocessor'].update(_model_overrides['data_preprocessor'])
model['backbone'].update(_model_overrides['backbone'])
model['neck'].update(_model_overrides['neck'])
model['roi_head']['bbox_head'] = _model_overrides['roi_head']['bbox_head']
model['roi_head']['mask_head'].update(
    _model_overrides['roi_head']['mask_head']
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
    ),
)

val_dataloader = None
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = None
test_evaluator = dict(
    type='PIDrayPaperMetric',
    ann_file=f'{data_root}annotations/test.json',
    metric=['bbox', 'segm'],
    classwise=True,
    subset_ann_files=subset_ann_files,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=12,
        milestones=[8, 11],
        gamma=0.1,
    ),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12)
val_cfg = None
test_cfg = dict(type='TestLoop')

_pidray_default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
    progress=dict(type='TqdmProgressHook', mininterval=1.0, leave=False),
)
default_hooks.update(_pidray_default_hooks)
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file='annotations/train.json',
)]
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './work_dirs/pidray_r101_attfpn_r0_legacy'
