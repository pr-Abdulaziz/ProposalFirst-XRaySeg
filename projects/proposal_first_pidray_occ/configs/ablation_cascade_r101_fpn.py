"""A1: Cascade Mask R-CNN with ordinary FPN and no PIDray R0."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

model['type'] = 'CascadeRCNN'
model.pop('r0_head', None)
model.pop('training_gate', None)
model.pop('gate_during_inference', None)
model['neck'] = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
)
work_dir = './work_dirs/ablation_cascade_r101_fpn'
