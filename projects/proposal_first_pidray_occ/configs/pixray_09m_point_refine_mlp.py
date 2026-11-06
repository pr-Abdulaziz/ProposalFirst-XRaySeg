"""09M: Method 09 with the attention replaced by a capacity-matched MLP.

The matched control for Method 09. It differs from ``pixray_09_point_refine``
in exactly one config key -- the point-refinement head's ``type`` -- which
swaps ``PointRefineTransformer`` for ``PointRefineMLP``: same token embedding,
same depth, same pre-LN residual structure, same FFN width, same decoder, same
point budgets, same schedule, and 398,479 parameters against 398,735.

Its purpose is to answer whether Method 09's effect comes from points attending
to each other or merely from the extra per-point capacity. Without this arm a
transformer-versus-baseline difference cannot be attributed to attention.

Not an active registry method. PIXray only.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_09_point_refine import *

model["roi_head"]["point_refine_head"]["type"] = "PointRefineMLPHead"

work_dir = f"{project_root}/work_dirs/pixray_09m_point_refine_mlp"
