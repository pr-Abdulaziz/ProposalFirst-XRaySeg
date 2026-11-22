"""Cascade RoI head that carries the occluder prediction to the mask loss.

``CascadeRoIHead._mask_forward`` returns a single ``mask_preds`` tensor, and
``predict_mask`` calls ``.split()`` on it, so a bilayer head cannot simply
return a tuple from ``forward``. This subclass adds the second prediction to
``mask_results`` during training and passes it to the loss; the inference path
is untouched, because ``mask_preds`` stays exactly the tensor it was.
"""

from typing import List, Tuple

from torch import Tensor

from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList

from .bilayer_fcn_mask_head import BilayerFCNMaskHead


@MODELS.register_module()
class BilayerCascadeRoIHead(CascadeRoIHead):
    """``CascadeRoIHead`` whose mask stage may predict an occluder layer."""

    def _mask_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor) -> dict:
        mask_head = self.mask_head[stage]
        if not isinstance(mask_head, BilayerFCNMaskHead):
            return super()._mask_forward(stage, x, rois)
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_feats = mask_roi_extractor(
            x[: mask_roi_extractor.num_inputs], rois
        )
        # predict_mask consumes only mask_preds, so the occluder classifier is
        # computed during training alone.
        mask_preds, mask_preds_bo = mask_head.forward_bilayer(
            mask_feats, with_occluder_logits=self.training
        )
        return dict(mask_preds=mask_preds, mask_preds_bo=mask_preds_bo)

    def mask_loss(
        self,
        stage: int,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
    ) -> dict:
        mask_head = self.mask_head[stage]
        if not isinstance(mask_head, BilayerFCNMaskHead):
            return super().mask_loss(stage, x, sampling_results, batch_gt_instances)
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            mask_preds_bo=mask_results["mask_preds_bo"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage],
        )
        mask_results.update(mask_loss_and_target)
        return mask_results
