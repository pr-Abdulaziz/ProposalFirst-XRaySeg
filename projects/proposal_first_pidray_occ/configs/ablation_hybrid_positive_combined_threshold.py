"""A8: full hybrid with a fixed positive q=phi*s threshold."""

from mmengine.config import read_base

with read_base():
    from .pidray_r101_attfpn_r0_hybrid_base import *

COMBINED_THRESHOLD = 0.25
model['combined_score_threshold'] = COMBINED_THRESHOLD
work_dir = './work_dirs/ablation_hybrid_positive_combined_threshold'
