"""S3-MBoost: recall-enhanced guided BiSAM plus residual mask correction."""

from mmengine.config import read_base

with read_base():
    from .pixray_s3_proposal_first_bisam import *

model["roi_head"].update(
    use_mask_booster=True,
    mask_booster_hidden_channels=8,
    mask_booster_strength=1.0,
)

work_dir = f"{project_root}/work_dirs/pixray_s3_mboost"
