"""B1: Mask R-CNN with ResNet-50 and FPN on PIXray.

Row 1 of the occlusion-aware paper's comparison table, retrained under this
repository's protocol instead of quoted. The protocol is inherited unchanged
from ``pixray_detector_base``; only the detector is replaced.
"""

from mmengine.config import read_base

with read_base():
    # Protocol first: pipelines, dataloaders, schedule, optimiser, evaluator.
    from .pixray_detector_base import *
    # Then the detector, which replaces ``model`` wholesale. The base config
    # defines nothing else, so nothing from the protocol is disturbed.
    from ....mmdet.configs._base_.models.mask_rcnn_r50_fpn import *

model["roi_head"]["bbox_head"]["num_classes"] = 15
model["roi_head"]["mask_head"]["num_classes"] = 15

work_dir = f"{project_root}/work_dirs/pixray_b1"
