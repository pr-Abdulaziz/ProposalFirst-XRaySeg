"""Shared point-refinement head for Method 09.

One instance is shared across all three cascade stages and both refinement
levels. It never applies stage weights itself: ``CascadeRoIHead`` already
multiplies every entry of ``mask_results['loss_mask']`` whose key contains
``loss`` by that stage's weight, so the head returns a single unweighted
``loss_point`` and the wrapper applies 1.0 / 0.5 / 0.25 exactly once.

For the same reason diagnostics are returned under keys that do **not** contain
``loss``: MMEngine's loss parser sums every key containing ``loss``, so emitting
both an aggregate and its components would optimise the term twice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

from .embeddings import PointTokenEmbedding
from .sampling import (grid_indices_to_roi_coords, sample_coarse_logits,
                       sample_fine_features, sample_mask_targets,
                       scatter_points, select_inference_points,
                       select_training_points)
from .transformer import PointRefineTransformer


@MODELS.register_module()
class PointRefineHead(nn.Module):
    """Refine coarse mask logits at successively finer grids.

    Args:
        num_classes: Foreground class count ``C``.
        output_sizes: Refinement grid sizes, coarse to fine. ``(56, 112)``
            initially; 224 requires a documented decision.
        train_points / infer_points: Point budgets. Training deliberately uses a
            smaller budget than inference.
        oversample_ratio / uncertain_fraction: Candidate pool and the fraction
            taken by uncertainty rather than uniformly.
        feature_stride: Stride of the fine feature map (P2): 4.
    """

    def __init__(
        self,
        num_classes: int,
        output_sizes: tuple = (56, 112),
        train_points: int = 196,
        infer_points: int = 784,
        oversample_ratio: int = 3,
        uncertain_fraction: float = 0.75,
        feature_stride: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 512,
        use_stage_embedding: bool = False,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_sizes = tuple(output_sizes)
        self.train_points = train_points
        self.infer_points = infer_points
        self.oversample_ratio = oversample_ratio
        self.uncertain_fraction = uncertain_fraction
        self.spatial_scale = 1.0 / feature_stride
        self.use_stage_embedding = use_stage_embedding
        self.embedding = PointTokenEmbedding(
            num_classes=num_classes, embed_dim=embed_dim,
            num_levels=len(self.output_sizes), num_stages=num_stages,
            use_stage_embedding=use_stage_embedding)
        # Built through a hook so a matched control can swap only the
        # point mixer. The default is byte-equivalent to the direct call.
        self.transformer = self._build_mixer(
            num_classes=num_classes, embed_dim=embed_dim, num_heads=num_heads,
            num_blocks=num_blocks, ffn_dim=ffn_dim)
        self.loss_point = nn.BCEWithLogitsLoss(reduction="mean")

    def _build_mixer(self, num_classes, embed_dim, num_heads, num_blocks,
                     ffn_dim) -> nn.Module:
        """Build the module that maps point tokens to point logits.

        Overridden by the matched control in ``mlp_control.py``. It is a hook
        rather than a constructor flag so Method 09's own source and its
        state-dict keys stay exactly as they were validated.
        """
        return PointRefineTransformer(
            num_classes=num_classes, embed_dim=embed_dim, num_heads=num_heads,
            num_blocks=num_blocks, ffn_dim=ffn_dim)

    # -- helpers ---------------------------------------------------------
    def _zero_loss(self, reference: Tensor) -> Tensor:
        """Finite, graph-connected zero that keeps every shared parameter in the
        autograd graph, so DDP still sees this module participate on a rank
        with no positive ROIs."""
        total = reference.sum() * 0.0
        for parameter in self.parameters():
            total = total + parameter.sum() * 0.0
        return total

    def _tokens(self, features, coarse, coords, level, stage):
        return self.embedding(
            features, coarse, coords, level=level,
            stage=stage if self.use_stage_embedding else None)

    # -- the refinement loop ---------------------------------------------
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
        """Run every refinement level.

        Returns ``(refined_logits, losses, diagnostics)``. ``losses`` is a list
        of per-level scalars, empty at inference.

        The coarse tensor passed in is never modified: token conditioning always
        samples the *original* 28x28 logits, while uncertainty selection uses
        the *current* upsampled grid.
        """
        current = coarse_logits
        losses, diagnostics = [], {}
        for level, size in enumerate(self.output_sizes):
            current = F.interpolate(
                current, size=(size, size), mode="bilinear", align_corners=False)
            if training:
                indices = select_training_points(
                    current, labels, self.train_points, self.oversample_ratio,
                    self.uncertain_fraction, generator)
            else:
                indices = select_inference_points(current, labels, self.infer_points)
            coords = grid_indices_to_roi_coords(indices, size, size)
            features = sample_fine_features(
                fine_features, rois, coords, self.spatial_scale)
            coarse_tokens = sample_coarse_logits(coarse_logits, coords)
            tokens = self._tokens(features, coarse_tokens, coords, level, stage)
            point_logits = self.transformer(tokens)                  # [N, K, C]

            if training:
                if gt_masks is None:
                    raise ValueError("training requires gt_masks for point targets")
                targets = sample_mask_targets(gt_masks, rois, coords)  # [N, K]
                num, k = targets.shape
                gather = labels.view(num, 1, 1).expand(num, k, 1)
                predicted = point_logits.gather(2, gather).squeeze(2)  # [N, K]
                losses.append(self.loss_point(predicted, targets))
                diagnostics[f"point_target_mean_l{level}"] = float(targets.mean())
            diagnostics[f"point_count_l{level}"] = float(indices.shape[1])
            current = scatter_points(current, indices, point_logits)
        return current, losses, diagnostics

    def loss_and_refine(
        self,
        coarse_logits: Tensor,
        rois: Tensor,
        labels: Tensor,
        fine_features: Tensor,
        gt_masks: Tensor,
        stage: int = 0,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Training entry point for one cascade stage.

        Returns a dict destined for ``mask_results['loss_mask']``. Exactly one
        key contains ``loss``; every diagnostic key deliberately does not.
        """
        if coarse_logits.shape[0] == 0:
            return {"loss_point": self._zero_loss(coarse_logits)}
        refined, losses, diagnostics = self.refine(
            coarse_logits, rois, labels, fine_features, training=True,
            gt_masks=gt_masks, stage=stage, generator=generator)
        # Mean over levels; the cascade wrapper applies the stage weight.
        out = {"loss_point": torch.stack(losses).mean()}
        for index, value in enumerate(losses):
            # No 'loss' in the key: reported, never summed or stage-weighted.
            out[f"point_bce_l{index}"] = value.detach()
        out.update({k: torch.as_tensor(v) for k, v in diagnostics.items()})
        out["_refined"] = refined
        return out

    def predict_refine(
        self, coarse_logits: Tensor, rois: Tensor, labels: Tensor,
        fine_features: Tensor, stage: int = 0,
    ) -> Tensor:
        """Inference entry point: refined logits, no targets, no randomness."""
        if coarse_logits.shape[0] == 0:
            return coarse_logits
        refined, _, _ = self.refine(
            coarse_logits, rois, labels, fine_features, training=False, stage=stage)
        return refined
