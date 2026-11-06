"""09: Cascade Mask R-CNN, ResNet-101 + AttFPN, with transformer point refinement.

Parent is Method 03. Everything from 03 is retained unchanged -- backbone,
AttFPN, RPN, box heads, coarse mask heads, preprocessing, augmentation, batch,
optimizer, schedule and evaluator. The only addition is one shared
``PointRefineHead`` called by all three cascade stages at two refinement levels.

Research candidate, not a proven improvement and not a reproduction of
PointRend, Mask Transfiner or RefineMask. It uses uncertainty-selected points
and a compact transformer; it has no quadtree, no incoherence detection and no
residual-logit correction.

PIXray only. PIDray is a separate later campaign.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_p2_cascade_attfpn import *

_num_classes = model["roi_head"]["mask_head"]["num_classes"]

# One shared refinement head; the cascade adapter must not build three.
model["roi_head"]["type"] = "PointRefineCascadeRoIHead"
model["roi_head"]["refine_enabled"] = True
model["roi_head"]["fine_level"] = 0          # P2, stride 4
model["roi_head"]["seed"] = 42
model["roi_head"]["point_refine_head"] = dict(
    type="PointRefineHead",
    num_classes=_num_classes,                # from config, never a literal
    output_sizes=(56, 112),                  # 224 needs a documented decision
    train_points=196,
    infer_points=784,
    oversample_ratio=3,
    uncertain_fraction=0.75,
    feature_stride=4,
    embed_dim=128,
    num_heads=4,
    num_blocks=2,
    ffn_dim=512,
    use_stage_embedding=False,               # optional ablation, off initially
)

work_dir = f"{project_root}/work_dirs/pixray_09_point_refine"
load_from = None
resume = False
