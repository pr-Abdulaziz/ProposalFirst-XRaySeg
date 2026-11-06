"""Opt-in uncertainty selection that changes only the inference tie handling.

Import this module explicitly through ``custom_imports`` and select one of its
registered head types. Existing heads, training paths and historical campaign
configs remain unchanged. The subclasses introduce no parameters or buffers;
checkpoints retain exactly the corresponding original head's state-dict keys.

The legacy selector subtracts ``index * dtype.eps`` before ranking, which can
reorder unequal uncertainties. A stable sort instead preserves their values
and uses ascending flattened index only when their uncertainties are equal.
Results obtained with these heads must identify the changed inference rule.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.utils.point_sample import get_uncertainty

from .detached import PointRefineHeadDetached, PointRefineMLPHeadDetached
from .head import PointRefineHead
from .mlp_control import PointRefineMLPHead
from .sampling import (grid_indices_to_roi_coords, sample_coarse_logits,
                       sample_fine_features, scatter_points)


def select_inference_points_stable(
    logits: Tensor, labels: Tensor, num_points: int = 784,
) -> Tensor:
    """Rank unchanged uncertainties, resolving exact ties by grid index.

    Returns unique flattened indices of shape ``[N, min(num_points, H*W)]``.
    CPU half/bfloat16 uncertainties are promoted losslessly to float32 for
    compatibility with older PyTorch sorting kernels. Float32/float64 values
    retain their original dtype and precision. No random state is consumed.
    """
    if num_points < 0:
        raise ValueError("num_points must be nonnegative")
    num_rois, _, height, width = logits.shape
    total = height * width
    k = min(num_points, total)
    if num_rois == 0 or k == 0:
        return torch.empty((num_rois, k), dtype=torch.long, device=logits.device)
    uncertainty = get_uncertainty(logits, labels).view(num_rois, total)
    if uncertainty.device.type == "cpu" and uncertainty.dtype in (
        torch.float16, torch.bfloat16,
    ):
        uncertainty = uncertainty.float()
    return torch.argsort(uncertainty, dim=1, descending=True, stable=True)[:, :k]


class _StableInferenceMixin:
    """Delegate training unchanged; replace only inference point selection."""

    def refine(
        self,
        coarse_logits: Tensor,
        rois: Tensor,
        labels: Tensor,
        fine_features: Tensor,
        training: bool,
        gt_masks: Tensor | None = None,
        stage: int = 0,
        generator: torch.Generator | None = None,
    ) -> tuple:
        if training:
            return super().refine(
                coarse_logits, rois, labels, fine_features, training=True,
                gt_masks=gt_masks, stage=stage, generator=generator)

        # Keep the original inference loop and diagnostics. In particular,
        # uncertainty reads the evolving mask, while tokens always read the
        # original coarse logits. predict_refine retains its empty-ROI guard.
        current = coarse_logits
        losses, diagnostics = [], {}
        for level, size in enumerate(self.output_sizes):
            current = F.interpolate(
                current, size=(size, size), mode="bilinear", align_corners=False)
            indices = select_inference_points_stable(
                current, labels, self.infer_points)
            coords = grid_indices_to_roi_coords(indices, size, size)
            features = sample_fine_features(
                fine_features, rois, coords, self.spatial_scale)
            coarse_tokens = sample_coarse_logits(coarse_logits, coords)
            tokens = self._tokens(features, coarse_tokens, coords, level, stage)
            point_logits = self.transformer(tokens)
            diagnostics[f"point_count_l{level}"] = float(indices.shape[1])
            current = scatter_points(current, indices, point_logits)
        return current, losses, diagnostics


@MODELS.register_module()
class PointRefineHeadStableInference(_StableInferenceMixin, PointRefineHead):
    """Transformer head with stable uncertainty ranking at inference only."""


@MODELS.register_module()
class PointRefineMLPHeadStableInference(_StableInferenceMixin, PointRefineMLPHead):
    """MLP control with stable uncertainty ranking at inference only."""


@MODELS.register_module()
class PointRefineHeadDetachedStableInference(
    _StableInferenceMixin, PointRefineHeadDetached,
):
    """Detached Transformer head with stable inference uncertainty ranking."""


@MODELS.register_module()
class PointRefineMLPHeadDetachedStableInference(
    _StableInferenceMixin, PointRefineMLPHeadDetached,
):
    """Detached MLP head with stable inference uncertainty ranking."""
