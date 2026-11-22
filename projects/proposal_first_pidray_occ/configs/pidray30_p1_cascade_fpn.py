"""P1: Cascade Mask R-CNN with ordinary FPN on PIDray-30."""

from mmengine.config import read_base

with read_base():
    from .pidray30_detector_base import *

model["neck"] = dict(
    type="FPN", in_channels=[256, 512, 1024, 2048],
    out_channels=256, num_outs=5,
)
work_dir = f"{project_root}/work_dirs/pidray30_p1"

