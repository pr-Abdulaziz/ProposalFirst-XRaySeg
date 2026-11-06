"""B3: Cascade R-CNN with ResNet-101 and FPN on PIXray, detection only.

No mask branch. B3 against P1 measures whether carrying a mask head costs
detection accuracy -- a comparison the source paper's table does not contain.
Only box metrics are defined for this arm, so the segmentation metric is
dropped from the evaluator rather than reported as an empty row.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *
    from ....mmdet.configs._base_.models.cascade_rcnn_r50_fpn import *

model["backbone"]["depth"] = 101
model["backbone"]["init_cfg"] = dict(
    type="Pretrained", checkpoint="torchvision://resnet101"
)
for bbox_head in model["roi_head"]["bbox_head"]:
    bbox_head["num_classes"] = 15

# The training pipeline still loads and filters masks so that B3 sees exactly
# the same instances as every other arm; the model simply ignores them.
test_evaluator = [
    dict(
        type="XRayCocoMetric",
        ann_file=f"{data_root}/annotations/test.json",
        metric=["bbox", "proposal"],
        classwise=True,
    ),
    dict(
        type="ProposalFirstRuntimeMetric",
        detection_threshold=0.5,
        timing_warmup=50,
    ),
]

work_dir = f"{project_root}/work_dirs/pixray_b3"
