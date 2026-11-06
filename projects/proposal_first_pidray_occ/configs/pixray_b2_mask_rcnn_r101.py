"""B2: Mask R-CNN with ResNet-101 and FPN on PIXray.

Backbone-matched to P1-P4, so B2 against P1 isolates the cascade refinement
stages rather than confounding them with backbone depth.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_b1_mask_rcnn_r50 import *

model["backbone"]["depth"] = 101
model["backbone"]["init_cfg"] = dict(
    type="Pretrained", checkpoint="torchvision://resnet101"
)

work_dir = f"{project_root}/work_dirs/pixray_b2"
