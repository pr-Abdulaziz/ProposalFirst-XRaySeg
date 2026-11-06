"""E2: E1 with occlusion- and boundary-aware BCE/Dice supervision."""

from mmengine.config import read_base

with read_base():
    from .pidray30_e1_adapter import *

model["roi_head"].update(
    hard_aware_loss=True,
    dice_loss_weight=1.0,
    occlusion_loss_weight=1.0,
    boundary_loss_weight=0.25,
)
work_dir = f"{project_root}/work_dirs/pidray30_e2"

