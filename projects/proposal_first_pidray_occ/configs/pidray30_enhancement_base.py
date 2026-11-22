"""Shared end-to-end protocol for the E1--E3 PIDray-30 comparisons."""

from mmengine.config import read_base

with read_base():
    from .pidray30_hybrid_base import *

model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)
model.update(
    use_proposal_screening=False,
    max_sam_proposals=100,
)
model["roi_head"].update(
    use_prompt_adapter=True,
    adapter_hidden_channels=32,
)

