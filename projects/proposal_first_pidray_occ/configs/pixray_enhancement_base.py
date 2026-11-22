"""Shared full-training protocol for the E1--E3 enhancement comparisons."""

from mmengine.config import read_base

with read_base():
    from .pixray_hybrid_base import *

# Use the same recall-enhanced proposal design as P3 and S3-P.
model["rpn_head"]["anchor_generator"]["scales"] = [4, 8]
model["train_cfg"]["rpn_proposal"].update(nms_pre=3000, max_per_img=3000)
model["test_cfg"]["rpn"].update(nms_pre=2000, max_per_img=2000)
model.update(
    sam_encoder_frozen=False,
    sam_decoder_frozen=False,
    use_proposal_screening=False,
    max_sam_proposals=100,
)
model["roi_head"].update(
    use_prompt_adapter=True,
    adapter_hidden_channels=32,
)

# E1--E3 train end to end under the same 60-epoch protocol as the full models.
max_epochs = 60
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs)
param_scheduler[1].update(end=max_epochs, T_max=max_epochs)
load_from = None
resume = False
