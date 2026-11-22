"""Complete proposal-first PIDray--Occ bilayer SAM experiment.

All experiments use the full official training set and evaluate the official
test set only after training, matching the parent-paper comparison protocol.
"""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

custom_imports = dict(
    imports=['mmdet.occlu', 'projects.proposal_first_pidray_occ'],
    allow_failed_imports=False,
)

crop_size = (512, 512)
data_root = 'data'
pidraya_ann_root = f'{data_root}/PIDray-A/annotations'
sam_model_dir = 'pretrained/sam_vit_large'
sam_checkpoint = f'{sam_model_dir}/pytorch_model_new.bin'

batch_augments = [dict(
    type='OcBatchFixedSizePad',
    size=crop_size,
    img_pad_value=0,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
)]

profile_inference = False

# Optional exact paper split files. If they are not set, PIDrayPaperMetric
# attempts to infer Easy/Hard/Hidden from image metadata or path components.
subset_ann_files = {
    'easy': f'{pidraya_ann_root}/test_easy_occ.json',
    'hard': f'{pidraya_ann_root}/test_hard_occ.json',
    'hidden': f'{pidraya_ann_root}/test_hidden_occ.json',
}

# Pre-registered primary screening policy. R0=0.5 follows PIDray and the
# detector score=0.05 follows the released MMDetection test configuration.
# The combined score is used only for ranking in the primary experiment;
# threshold 0.0 deliberately disables an additional, unvalidated q gate.
PRIMARY_R0_THRESHOLD = 0.5
PRIMARY_DET_THRESHOLD = 0.05
PRIMARY_COMBINED_THRESHOLD = 0.0
PRIMARY_MAX_SAM_PROPOSALS = 70

_hybrid_model_overrides = dict(
    type='ProposalFirstPIDrayOcc',
    training_gate='none',
    gate_during_inference=False,
    det_score_threshold=PRIMARY_DET_THRESHOLD,
    combined_score_threshold=PRIMARY_COMBINED_THRESHOLD,
    max_sam_proposals=PRIMARY_MAX_SAM_PROPOSALS,
    use_r0_screening=True,
    use_proposal_screening=True,
    r0_rescue_enabled=False,
    rescue_det_threshold=0.20,
    rescue_topk=5,
    sam_encoder_frozen=False,
    sam_positional_frozen=True,
    sam_decoder_frozen=False,
    profile_inference=profile_inference,
    data_preprocessor=dict(
        _delete_=True,
        type='OcDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
        batch_augments=batch_augments,
    ),
    sam_image_encoder=dict(
        type='MMPretrainSamVisionEncoder',
        img_size=crop_size[0],
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type='Pretrained', checkpoint=sam_checkpoint),
        peft_config=dict(
            peft_type='LORA',
            r=16,
            target_modules=['qkv'],
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
        ),
    ),
    sam_positional_embedding=dict(
        type='RSSamPositionalEmbedding',
        hf_pretrain_name=sam_model_dir,
        init_cfg=dict(type='Pretrained', checkpoint=sam_checkpoint),
    ),
    roi_head=dict(
        type='ProposalFirstCascadeRoIHead',
        max_mask_rois_per_image=64,
        use_occluder_decoder=True,
        use_residual_guidance=True,
        mask_head=dict(
            _delete_=True,
            type='ProposalFirstBilayerMaskHead',
            mask_decoder=dict(
                type='OcRSSamMaskDecoder',
                hf_pretrain_name=sam_model_dir,
                init_cfg=dict(type='Pretrained', checkpoint=sam_checkpoint),
            ),
            in_channels=256,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            mask_size=crop_size,
            overlap_loss_weight=0.25,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        ),
    ),
)

model.update(
    type=_hybrid_model_overrides['type'],
    training_gate=_hybrid_model_overrides['training_gate'],
    gate_during_inference=_hybrid_model_overrides['gate_during_inference'],
    det_score_threshold=_hybrid_model_overrides['det_score_threshold'],
    combined_score_threshold=_hybrid_model_overrides[
        'combined_score_threshold'
    ],
    max_sam_proposals=_hybrid_model_overrides['max_sam_proposals'],
    use_r0_screening=_hybrid_model_overrides['use_r0_screening'],
    use_proposal_screening=_hybrid_model_overrides[
        'use_proposal_screening'
    ],
    r0_rescue_enabled=_hybrid_model_overrides['r0_rescue_enabled'],
    rescue_det_threshold=_hybrid_model_overrides['rescue_det_threshold'],
    rescue_topk=_hybrid_model_overrides['rescue_topk'],
    sam_encoder_frozen=_hybrid_model_overrides['sam_encoder_frozen'],
    sam_positional_frozen=_hybrid_model_overrides[
        'sam_positional_frozen'
    ],
    sam_decoder_frozen=_hybrid_model_overrides['sam_decoder_frozen'],
    profile_inference=_hybrid_model_overrides['profile_inference'],
    sam_image_encoder=_hybrid_model_overrides['sam_image_encoder'],
    sam_positional_embedding=_hybrid_model_overrides[
        'sam_positional_embedding'
    ],
)
model['data_preprocessor'] = dict(
    _hybrid_model_overrides['data_preprocessor']
)
model['data_preprocessor'].pop('_delete_', None)
model['r0_head']['threshold'] = PRIMARY_R0_THRESHOLD
model['roi_head'].update(
    type=_hybrid_model_overrides['roi_head']['type'],
    max_mask_rois_per_image=_hybrid_model_overrides['roi_head'][
        'max_mask_rois_per_image'
    ],
    use_occluder_decoder=_hybrid_model_overrides['roi_head'][
        'use_occluder_decoder'
    ],
    use_residual_guidance=_hybrid_model_overrides['roi_head'][
        'use_residual_guidance'
    ],
)
model['roi_head']['mask_head'] = dict(
    _hybrid_model_overrides['roi_head']['mask_head']
)
model['roi_head']['mask_head'].pop('_delete_', None)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='OcLoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='OcRandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.5),
        resize_type='OcResize',
        keep_ratio=True,
    ),
    dict(
        type='OcRandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1e-5, 1e-5),
        by_mask=True,
        keep_empty=True,
    ),
    dict(type='OcPackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(
        type='Pad',
        size=crop_size,
        pad_val=dict(img=(103.53, 116.28, 123.675), masks=0),
    ),
    dict(type='OcLoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='OcPackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor'),
    ),
]

metainfo = dict(classes=(
    'Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench',
    'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife', 'Lighter',
))

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='PIDrayOcclusionDataset',
        data_root=data_root,
        ann_file=f'{pidraya_ann_root}/train_occ_runtime.json',
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
        _delete_=True,
        type='PIDrayOcclusionDataset',
        data_root=data_root,
        ann_file=f'{pidraya_ann_root}/test_occ.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = None
test_evaluator = [
    dict(
        type='PIDrayPaperMetric',
        ann_file=f'{pidraya_ann_root}/test_occ.json',
        metric=['bbox', 'segm'],
        classwise=True,
        subset_ann_files=subset_ann_files,
    ),
    dict(
        type='ProposalFirstRuntimeMetric',
        r0_threshold=0.5,
        detection_threshold=0.5,
        timing_warmup=50,
    ),
]

base_lr = 1e-4
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
max_epochs = 50
param_scheduler = [
    # Match the released Occ PIDray config and paper: 50 warm-up iterations.
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
    ),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
val_cfg = None
_hybrid_default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
    ),
    logger=dict(type='LoggerHook', interval=20),
)
default_hooks.update(_hybrid_default_hooks)
custom_hooks = [dict(
    type='PIDrayCheckpointProvenanceHook',
    training_scope='official_full_train',
    annotation_file=f'{pidraya_ann_root}/train_occ_runtime.json',
)]
find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=2)
# Primary training leaves this unset: ResNet-101 uses ImageNet initialization,
# PIDray-specific detector heads train jointly, and SAM uses ViT-L weights.
load_from = None
resume = False
work_dir = './work_dirs/proposal_first_pidray_occ'
