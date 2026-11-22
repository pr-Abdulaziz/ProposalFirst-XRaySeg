"""B6: P1 plus occlusion-aware bilayer mask decoding.

P1 with a second FCN mask output supervised by the occluder annotations.
B6 against P1 is a single variable: occluder supervision on an ordinary
FCN mask head, with no SAM anywhere in the model.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_bilayer_detector_base import *

model["neck"] = dict(
    type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
)
work_dir = f"{project_root}/work_dirs/pixray_b6"
