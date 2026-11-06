"""P2: Cascade Mask R-CNN with AttFPN on PIDray-30."""

from mmengine.config import read_base

with read_base():
    from .pidray30_detector_base import *

work_dir = f"{project_root}/work_dirs/pidray30_p2"

