"""E1: recall-RPN single decoder with a trainable prompt adapter."""

from mmengine.config import read_base

with read_base():
    from .pidray30_enhancement_base import *

model["roi_head"].update(
    use_occluder_decoder=False,
    use_residual_guidance=False,
    use_mask_booster=False,
    hard_aware_loss=False,
    conditional_bisam=False,
)
model["roi_head"]["mask_head"]["overlap_loss_weight"] = 0.0
work_dir = f"{project_root}/work_dirs/pidray30_e1"

