"""B4: Cascade Mask R-CNN with ResNet-50 and FPN on PIXray.

The literal R-50 form of row 2 of the source paper's table. P1 is the R-101
form, so B4 shows whether even the smaller backbone clears the published
number under a matched training budget.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *

model["backbone"]["depth"] = 50
model["backbone"]["init_cfg"] = dict(
    type="Pretrained", checkpoint="torchvision://resnet50"
)
model["neck"] = dict(
    type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
)

work_dir = f"{project_root}/work_dirs/pixray_b4"
