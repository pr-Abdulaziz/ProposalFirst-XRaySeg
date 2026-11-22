"""Gradient-isolated point refiners (diagnostic variants of Method 09 and 09M).

The point loss of the original refiners reaches the detector through two inputs of the token
embedding: the sampled P2 features (backbone and neck) and the sampled coarse-mask logits (the three
coarse mask heads). These variants detach exactly those two tensors before their embedding
projections. Nothing else changes: same point selection, same coordinates, same targets, same loss,
same state-dict keys, and forward values identical to the originals. The point loss can then update
only the refiner, while the detector's own losses train the detector as before.

Registered through ``custom_imports`` of the diagnostic configs, so the frozen refiner modules and
their package ``__init__`` stay untouched.
"""
from __future__ import annotations

from torch import Tensor

from mmdet.registry import MODELS

from .head import PointRefineHead
from .mlp_control import PointRefineMLPHead


class _DetachedTokensMixin:
    """Detach the sampled P2 features and sampled coarse logits before the token embedding."""

    def _tokens(self, features: Tensor, coarse: Tensor, coords: Tensor, level: int, stage: int):
        return super()._tokens(features.detach(), coarse.detach(), coords, level, stage)


@MODELS.register_module()
class PointRefineHeadDetached(_DetachedTokensMixin, PointRefineHead):
    """Transformer refiner (Method 09) with gradient isolation from the detector."""


@MODELS.register_module()
class PointRefineMLPHeadDetached(_DetachedTokensMixin, PointRefineMLPHead):
    """MLP refiner (Method 09M) with gradient isolation from the detector."""
