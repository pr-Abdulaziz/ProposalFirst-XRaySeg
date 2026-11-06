"""S3-P: recall-enhanced proposals followed by bilayer SAM."""

from mmengine.config import read_base

with read_base():
    from .pixray_hybrid_base import *

model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)
model["use_proposal_screening"] = False
model["max_sam_proposals"] = 100
model["roi_head"]["use_mask_booster"] = False

work_dir = f"{project_root}/work_dirs/pixray_s3_p"
