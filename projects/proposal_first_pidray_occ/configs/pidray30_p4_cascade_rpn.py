"""P4: Cascade-AttFPN with learned two-stage Cascade RPN on PIDray-30."""

from mmengine.config import read_base

with read_base():
    from .pidray30_p2_cascade_attfpn import *

_rpn_weight = 0.7
_p2_rpn_proposal_cfg = model["train_cfg"]["rpn_proposal"].copy()
_p2_test_rpn_cfg = model["test_cfg"]["rpn"].copy()
model["rpn_head"] = dict(
    type="CascadeRPNHead",
    num_stages=2,
    stages=[
        dict(
            type="StageCascadeRPNHead",
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator", scales=[8], ratios=[1.0],
                strides=[4, 8, 16, 32, 64],
            ),
            adapt_cfg=dict(type="dilation", dilation=3),
            bridged_feature=True,
            with_cls=False,
            reg_decoded_bbox=True,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.5, 0.5),
            ),
            loss_bbox=dict(
                type="IoULoss", linear=True, loss_weight=10.0 * _rpn_weight,
            ),
        ),
        dict(
            type="StageCascadeRPNHead",
            in_channels=256,
            feat_channels=256,
            adapt_cfg=dict(type="offset"),
            bridged_feature=False,
            with_cls=True,
            reg_decoded_bbox=True,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.05, 0.05, 0.1, 0.1),
            ),
            loss_cls=dict(
                type="CrossEntropyLoss", use_sigmoid=True,
                loss_weight=1.0 * _rpn_weight,
            ),
            loss_bbox=dict(
                type="IoULoss", linear=True, loss_weight=10.0 * _rpn_weight,
            ),
        ),
    ],
)
model["train_cfg"]["rpn"] = [
    dict(
        assigner=dict(type="RegionAssigner", center_ratio=0.2, ignore_ratio=0.5),
        allowed_border=-1, pos_weight=-1, debug=False,
    ),
    dict(
        assigner=dict(
            type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.7,
            min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler", num=256, pos_fraction=0.5,
            neg_pos_ub=-1, add_gt_as_proposals=False,
        ),
        allowed_border=-1, pos_weight=-1, debug=False,
    ),
]
model["train_cfg"]["rpn_proposal"] = _p2_rpn_proposal_cfg
model["test_cfg"]["rpn"] = _p2_test_rpn_cfg
work_dir = f"{project_root}/work_dirs/pidray30_p4"

