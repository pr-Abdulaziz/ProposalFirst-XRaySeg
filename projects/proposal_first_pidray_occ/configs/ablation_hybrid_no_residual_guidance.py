"""A5: bilayer decoders with overlap supervision but no residual guidance."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

model['roi_head']['use_occluder_decoder'] = True
model['roi_head']['use_residual_guidance'] = False
work_dir = './work_dirs/ablation_hybrid_no_residual_guidance'
