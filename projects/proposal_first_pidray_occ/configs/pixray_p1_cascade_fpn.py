"""P1: Cascade Mask R-CNN with ordinary FPN on PIXray."""

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *

model["neck"] = dict(
    type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
)
work_dir = f"{project_root}/work_dirs/pixray_p1"
