"""Opt-in FP32 prediction for stable-selector heads receiving AMP features.

The legacy sampling helper can allocate a half output while MMCV point sampling
produces float32 values under CUDA autocast. These separate head types keep the
entire prediction refinement path in float32, including coordinate sampling.
Training delegates unchanged and is NOT repaired or validated by this module.

Import explicitly through custom_imports. Keep head parameters/buffers float32;
do not call model.half(). Returned prediction logits are float32. No parameters,
buffers, or state-dict keys are added to the stable-selector counterparts.
"""
from __future__ import annotations

import torch
from torch import Tensor

from mmdet.registry import MODELS

from .stable_inference import (
    PointRefineHeadStableInference,
    PointRefineMLPHeadStableInference,
    PointRefineHeadDetachedStableInference,
    PointRefineMLPHeadDetachedStableInference,
)


class _FP32PredictionMixin:
    def _require_fp32_prediction_weights(self) -> None:
        for name, value in list(self.named_parameters()) + list(self.named_buffers()):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise TypeError(
                    f"FP32 prediction requires float32 head weights/buffers; "
                    f"{name} has dtype {value.dtype}. Keep the model in float32 "
                    "and use autocast for upstream inference; do not use model.half().")

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
        self._require_fp32_prediction_weights()
        with torch.autocast(device_type=coarse_logits.device.type, enabled=False):
            return super().refine(
                coarse_logits.float(), rois.float(), labels, fine_features.float(),
                training=False, gt_masks=gt_masks, stage=stage, generator=generator)

    def predict_refine(
        self, coarse_logits: Tensor, rois: Tensor, labels: Tensor,
        fine_features: Tensor, stage: int = 0,
    ) -> Tensor:
        self._require_fp32_prediction_weights()
        with torch.autocast(device_type=coarse_logits.device.type, enabled=False):
            # Include the inherited empty-ROI return in the float32 contract.
            return super().predict_refine(
                coarse_logits.float(), rois.float(), labels, fine_features.float(),
                stage=stage)


@MODELS.register_module()
class PointRefineHeadFP32Inference(
    _FP32PredictionMixin, PointRefineHeadStableInference,
):
    """Stable-selector Transformer with explicit FP32 prediction."""


@MODELS.register_module()
class PointRefineMLPHeadFP32Inference(
    _FP32PredictionMixin, PointRefineMLPHeadStableInference,
):
    """Stable-selector MLP with explicit FP32 prediction."""


@MODELS.register_module()
class PointRefineHeadDetachedFP32Inference(
    _FP32PredictionMixin, PointRefineHeadDetachedStableInference,
):
    """Detached stable-selector Transformer with explicit FP32 prediction."""


@MODELS.register_module()
class PointRefineMLPHeadDetachedFP32Inference(
    _FP32PredictionMixin, PointRefineMLPHeadDetachedStableInference,
):
    """Detached stable-selector MLP with explicit FP32 prediction."""
