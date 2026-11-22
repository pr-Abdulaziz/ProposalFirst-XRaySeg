"""A2: Cascade Mask R-CNN with AttFPN and no PIDray R0."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

model['type'] = 'CascadeRCNN'
model.pop('r0_head', None)
model.pop('training_gate', None)
model.pop('gate_during_inference', None)
work_dir = './work_dirs/ablation_cascade_r101_attfpn'
