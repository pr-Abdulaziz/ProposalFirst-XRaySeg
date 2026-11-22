"""P3: Cascade-AttFPN with recall-enhanced RPN on PIDray-30."""

from mmengine.config import read_base

with read_base():
    from .pidray30_p2_cascade_attfpn import *

model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)
work_dir = f"{project_root}/work_dirs/pidray30_p3"

