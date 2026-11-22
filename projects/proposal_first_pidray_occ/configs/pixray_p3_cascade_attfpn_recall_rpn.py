"""P3: Cascade-AttFPN with a recall-enhanced RPN on PIXray."""

from mmengine.config import read_base

with read_base():
    from .pixray_detector_base import *

# Add one smaller anchor scale and retain more candidates before Cascade
# refinement.  No SAM or occlusion-specific component is introduced here.
model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)

work_dir = f"{project_root}/work_dirs/pixray_p3"
