"""08: PIDray R0/R1, adapted to PIXray.

Architectural source: https://github.com/lutao2021/PIDray, whose README names
``configs/cascade_mask_rcnn_r101_with_R0R1.py`` as the reproduction entry point
under MMDetection 2.25. This project runs MMDetection 3.2 / MMEngine, so the
model computation is reused through the existing port in
``projects/proposal_first_pidray_occ/models/`` rather than by downgrading the
shared runtime.

**This is not an original PIDray benchmark reproduction.** It trains and is
scored on the same immutable PIXray images and target annotations as methods
01-07, with 15 output classes. No PIDray image, PIDray annotation, PIXray-A
auxiliary supervision, invented normal image, or pretrained PIDray checkpoint is
used. Reproducing the PIDray benchmark would need that dataset, its splits and
its own table.

Inherits method 03 (Cascade Mask R-CNN, ResNet-101 + AttFPN) and adds only the
R0/R1 mechanism, so 03 -> 08 is the intended comparison. The scale-8 anchors of
03 are retained deliberately: method 04's recall-enhanced RPN and method 07's
bilayer mask head must not leak in.

Local adaptations relative to the released PIDray protocol, all inherited from
the common local schedule: AdamW at 1e-4 with cosine decay rather than the
released SGD step schedule, 512x512 with RandomResize 0.5-1.5 and RandomCrop,
physical batch 2 with accumulation 8, AMP, and final-epoch checkpoint selection.

R0 on PIXray is a limited test of the released intent. Every original PIXray
image contains target items, so normal-image specificity is undefined on this
test set. Empty training crops can still arise from augmentation and the R0
target is read from post-transform ``gt_instances``; those crops are not
original normal-baggage images and are counted separately in the diagnostics.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_p2_cascade_attfpn import *

# pixray_detector_base removes R0 because PIXray has no normal-image
# supervision. Method 08 exists to exercise it, so it is restored here.
model["type"] = "PIDrayCascadeRCNN"
model["r0_head"] = dict(
    type="PIDrayR0Head",
    hidden_dim=256,
    threshold=0.5,
    loss_weight=0.1,
    loss_type="bce",
)
model["training_gate"] = "legacy"
model["gate_during_inference"] = True

# Batch size one for the primary comparison: the inference gate zeroes scores
# per image, and batching changes nothing about that, but the released
# comparison is single-image and this keeps latency measurement comparable.
test_dataloader["batch_size"] = 1

work_dir = f"{project_root}/work_dirs/pixray_08_pidray_r0r1"
load_from = None
resume = False
