"""H3: inference-only rescue route using the trained H2 focal checkpoint."""

from mmengine.config import read_base

with read_base():
    from .ablation_hidden_r0_focal import *

model['r0_rescue_enabled'] = True
model['rescue_det_threshold'] = 0.20
model['rescue_topk'] = 5
work_dir = './work_dirs/ablation_hidden_r0_focal_rescue'
