"""Occlusion-aware bilayer mask decoding for a conventional FCN mask head.

The occlusion-aware literature couples bilayer decoding to a SAM decoder, so
the two changes -- predicting the occluder alongside the target, and replacing
the mask head with SAM -- are never separated. Measurement on PIXray puts the
whole detector/hybrid mask gap inside the SAM decoder: box AR is flat across
arms while the recall lost between box and mask is 0.107-0.110 for detectors
against 0.183-0.203 for the SAM hybrids.

This head applies the occlusion-aware idea to the mask head that actually wins,
so occluder supervision can be tested without SAM. The target branch is the
stock ``FCNMaskHead``, unchanged, which keeps the comparison against P1-P3 to a
single variable: a parallel occluder branch, optionally feeding the target
branch through a residual projection in the way the SAM bilayer decoder feeds
its occluder embeddings forward.

Targets come from ``gt_instances.bo_masks`` and are cropped to the RoI by the
same ``mask_target`` utility the target branch uses, so the occluder is
supervised in RoI-relative coordinates rather than on a global grid.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances  # noqa: F401  (registry warm-up)
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target


@MODELS.register_module()
class BilayerFCNMaskHead(FCNMaskHead):
    """``FCNMaskHead`` with a parallel occluder branch.

    Args:
        bo_loss_weight: Weight on the occluder-mask loss. ``0`` keeps the
            branch in the graph but removes its supervision.
        with_residual_guidance: Add a projection of the occluder branch's
            pre-logit features into the target branch, mirroring how the SAM
            bilayer decoder passes occluder embeddings to the target decoder.
        bo_class_agnostic: Predict a single occluder channel rather than one
            per class. An occluder is frequently a different category from the
            instance it covers, and often not a prohibited item at all, so its
            class is not a meaningful target. Defaults to ``True``.
    """

    def __init__(
        self,
        *args,
        bo_loss_weight: float = 0.25,
        with_residual_guidance: bool = True,
        bo_class_agnostic: bool = True,
        **kwargs,
    ) -> None:
        if bo_loss_weight < 0:
            raise ValueError(
                f"bo_loss_weight must be non-negative, got {bo_loss_weight}"
            )
        super().__init__(*args, **kwargs)
        self.bo_loss_weight = float(bo_loss_weight)
        self.with_residual_guidance = with_residual_guidance
        self.bo_class_agnostic = bo_class_agnostic

        # The occluder branch mirrors the target branch layer for layer, so the
        # two predictions are made at the same resolution from the same features.
        self.bo_convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            self.bo_convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=(self.conv_kernel_size - 1) // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        upsample_cfg = self.upsample_cfg.copy()
        if self.upsample_method == "deconv":
            upsample_cfg.update(
                in_channels=self.conv_out_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
            )
        elif self.upsample_method is not None:
            upsample_cfg.update(scale_factor=self.scale_factor)
        self.bo_upsample = (
            build_upsample_layer(upsample_cfg)
            if self.upsample_method is not None
            else None
        )
        bo_out_channels = 1 if bo_class_agnostic else self.num_classes
        self.conv_logits_bo = build_conv_layer(
            self.predictor_cfg, self.conv_out_channels, bo_out_channels, 1
        )
        self.residual_proj = (
            build_conv_layer(
                self.predictor_cfg,
                self.conv_out_channels,
                self.conv_out_channels,
                1,
            )
            if with_residual_guidance
            else None
        )

    def forward_bilayer(
        self, x: Tensor, with_occluder_logits: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Return the target logits and, optionally, the occluder logits.

        The occluder convolutions always run: residual guidance makes the
        target prediction depend on them, so that cost is real at inference and
        is reported as such. The occluder upsample and classifier are skipped
        when their output is unused, which is every forward pass outside
        training -- computing them would inflate the measured latency of these
        arms against the detector arms they are compared with.
        """
        bo_feat = x
        for conv in self.bo_convs:
            bo_feat = conv(bo_feat)

        target_feat = x
        for conv in self.convs:
            target_feat = conv(target_feat)
        if self.residual_proj is not None:
            # Occluder evidence reaches the target branch before upsampling,
            # the same place the SAM bilayer decoder injects it.
            target_feat = target_feat + self.residual_proj(bo_feat)

        if self.upsample is not None:
            target_feat = self.upsample(target_feat)
            if self.upsample_method == "deconv":
                target_feat = self.relu(target_feat)
        mask_preds = self.conv_logits(target_feat)
        if not with_occluder_logits:
            return mask_preds, None

        if self.upsample is not None:
            bo_feat = self.bo_upsample(bo_feat)
            if self.upsample_method == "deconv":
                bo_feat = self.relu(bo_feat)
        return mask_preds, self.conv_logits_bo(bo_feat)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_bilayer(x, with_occluder_logits=False)[0]

    def get_bo_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: List[InstanceData],
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        """Occluder masks cropped to each positive RoI."""
        for gt_instances in batch_gt_instances:
            if "bo_masks" not in gt_instances:
                raise KeyError(
                    "BilayerFCNMaskHead requires gt_instances.bo_masks. Use "
                    "PIXrayOcclusionDataset with OcLoadAnnotations and "
                    "OcPackDetInputs on *_occ.json annotations."
                )
        return mask_target(
            [res.pos_priors for res in sampling_results],
            [res.pos_assigned_gt_inds for res in sampling_results],
            [res.bo_masks for res in batch_gt_instances],
            rcnn_train_cfg,
        )

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: List[InstanceData],
        rcnn_train_cfg: ConfigDict,
        mask_preds_bo: Tensor = None,
    ) -> dict:
        outputs = super().loss_and_target(
            mask_preds=mask_preds,
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg,
        )
        if mask_preds_bo is None:
            return outputs

        bo_targets = self.get_bo_targets(
            sampling_results, batch_gt_instances, rcnn_train_cfg
        )
        if mask_preds_bo.size(0) == 0 or bo_targets.size(0) == 0:
            # Keep the branch in the graph on an empty batch, as the target
            # branch does, so DDP still sees a gradient for every parameter.
            loss_mask_bo = mask_preds_bo.sum()
        else:
            labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            if self.bo_class_agnostic:
                labels = torch.zeros_like(labels)
            loss_mask_bo = self.loss_mask(mask_preds_bo, bo_targets, labels)
        outputs["loss_mask"]["loss_mask_bo"] = self.bo_loss_weight * loss_mask_bo
        outputs["mask_bo_targets"] = bo_targets
        return outputs
