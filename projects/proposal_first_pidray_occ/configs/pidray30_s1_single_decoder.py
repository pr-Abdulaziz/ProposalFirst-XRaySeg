"""S1: proposal-first single-decoder segmentation baseline on PIDray-30."""

from mmengine.config import read_base

with read_base():
    from .pidray30_hybrid_base import *

model["roi_head"]["use_occluder_decoder"] = False
model["roi_head"]["use_residual_guidance"] = False
model["roi_head"]["mask_head"]["overlap_loss_weight"] = 0.0
model["use_proposal_screening"] = False
work_dir = f"{project_root}/work_dirs/pidray30_s1"

