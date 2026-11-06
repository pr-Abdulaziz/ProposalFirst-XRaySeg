"""A9: full primary proposal-first PIDray--Occ hybrid."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

work_dir = './work_dirs/ablation_hybrid_full'
