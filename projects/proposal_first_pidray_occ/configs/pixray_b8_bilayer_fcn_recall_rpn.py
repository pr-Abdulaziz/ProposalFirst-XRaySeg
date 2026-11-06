"""B8: P3 plus occlusion-aware bilayer mask decoding.

P3's recall-enhanced RPN with the bilayer mask head. P3 gave the best
small-object recall of the eleven controlled arms, so B8 tests whether better
proposals and occluder supervision compound or merely overlap.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_bilayer_detector_base import *

model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)

work_dir = f"{project_root}/work_dirs/pixray_b8"
