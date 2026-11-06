"""P2: Cascade Mask R-CNN with AttFPN on PIXray."""

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *

work_dir = f"{project_root}/work_dirs/pixray_p2"
