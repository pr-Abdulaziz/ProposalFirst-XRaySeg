"""A7: full hybrid with R0 image screening but no custom proposal gate/budget."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

model['use_proposal_screening'] = False
work_dir = './work_dirs/ablation_hybrid_no_proposal_screening'
