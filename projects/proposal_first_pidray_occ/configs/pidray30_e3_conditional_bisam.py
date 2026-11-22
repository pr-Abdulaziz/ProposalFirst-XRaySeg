"""E3: conditional guided BiSAM with hard-aware loss and mask correction."""

from mmengine.config import read_base

with read_base():
    from .pidray30_enhancement_base import *

model["roi_head"].update(
    use_occluder_decoder=True,
    use_residual_guidance=True,
    use_mask_booster=True,
    mask_booster_hidden_channels=8,
    mask_booster_strength=1.0,
    hard_aware_loss=True,
    dice_loss_weight=1.0,
    occlusion_loss_weight=1.0,
    boundary_loss_weight=0.25,
    conditional_bisam=True,
    difficulty_threshold=0.05,
    difficulty_gate_threshold=0.5,
    difficulty_loss_weight=0.25,
)
work_dir = f"{project_root}/work_dirs/pidray30_e3"

