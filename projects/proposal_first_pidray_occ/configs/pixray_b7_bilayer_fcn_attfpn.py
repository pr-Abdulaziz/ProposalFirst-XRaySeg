"""B7: P2 plus occlusion-aware bilayer mask decoding.

The AttFPN counterpart of B6, so B7 against P2 isolates occluder supervision
on the strongest detector arm.
"""

from mmengine.config import read_base

with read_base():
    from .pixray_bilayer_detector_base import *

work_dir = f"{project_root}/work_dirs/pixray_b7"
