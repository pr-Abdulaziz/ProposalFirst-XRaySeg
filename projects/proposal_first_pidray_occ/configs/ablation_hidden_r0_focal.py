"""H2: train the full hybrid with focal loss applied only to R0."""

from mmengine.config import read_base

with read_base():
    from .ablation_hybrid_full import *

model['r0_head'].update(
    loss_type='focal',
    focal_gamma=2.0,
    focal_alpha=0.25,
)
work_dir = './work_dirs/ablation_hidden_r0_focal'
