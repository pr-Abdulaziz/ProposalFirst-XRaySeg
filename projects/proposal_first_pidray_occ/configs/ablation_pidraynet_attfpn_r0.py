"""A3: full PIDrayNet with AttFPN and R0."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_legacy import *

work_dir = './work_dirs/ablation_pidraynet_attfpn_r0'
