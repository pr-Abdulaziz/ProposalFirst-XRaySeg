"""A6: full hybrid with detector proposal screening but no R0 gate."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

model['use_r0_screening'] = False
work_dir = './work_dirs/ablation_hybrid_no_r0_screening'
