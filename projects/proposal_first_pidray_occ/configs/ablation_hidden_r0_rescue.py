"""H1: inference-only R0 rescue route using the trained A9 checkpoint."""

from mmengine.config import read_base

with read_base():
    from .ablation_hybrid_full import *

model['r0_rescue_enabled'] = True
model['rescue_det_threshold'] = 0.20
model['rescue_topk'] = 5
work_dir = './work_dirs/ablation_hidden_r0_rescue'
