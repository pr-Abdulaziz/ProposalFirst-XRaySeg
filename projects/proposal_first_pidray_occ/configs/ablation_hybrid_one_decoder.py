"""A4: RSPrompter-style target decoder without the occluder decoder."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

model['roi_head']['use_occluder_decoder'] = False
model['roi_head']['use_residual_guidance'] = False
model['roi_head']['mask_head']['overlap_loss_weight'] = 0.0
work_dir = './work_dirs/ablation_hybrid_one_decoder'
