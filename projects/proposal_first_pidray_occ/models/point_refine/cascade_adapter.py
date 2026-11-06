"""Cascade RoI head that adds Method 09's shared point refinement.

Exactly one ``PointRefineHead`` instance is created and reused by every cascade
stage and every refinement level; three independently constructed heads would
not be the proposed architecture.

Training: each stage calls the shared head with *its own* positive ROIs, labels
and assignments, since ``N_s`` differs between stages (observed 14, 9, 7). The
point loss is returned inside ``mask_results['loss_mask']`` under a single key
containing ``loss``, so ``CascadeRoIHead`` applies that stage's weight exactly
once and nothing is double-counted.

Inference: each stage's logits are refined, then the baseline sigmoid, stage
merge and paste run unchanged. ``refine_enabled=False`` routes through the
untouched parent implementation for equivalence testing.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.models.utils.misc import empty_instances
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList


@MODELS.register_module()
class PointRefineCascadeRoIHead(CascadeRoIHead):
    """``CascadeRoIHead`` plus a shared transformer point-refinement head.

    Args:
        point_refine_head: Config for the shared :class:`PointRefineHead`.
        refine_enabled: When ``False`` every path falls back to the parent, so
            the model is bit-identical to Method 03 under the same weights.
        fine_level: Index of the feature level the refiner samples. 0 is P2.
        seed: Seed for the point-selection generator, so training-point choice
            is reproducible independently of global RNG consumption.
    """

    def __init__(self, *args, point_refine_head: dict = None,
                 refine_enabled: bool = True, fine_level: int = 0,
                 seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if point_refine_head is None:
            raise ValueError("point_refine_head config is required")
        # One shared instance, deliberately not a ModuleList of three.
        self.point_refine_head = MODELS.build(point_refine_head)
        self.refine_enabled = refine_enabled
        self.fine_level = fine_level
        self._seed = seed
        self._generator = None
        self._pending_generator_state = None

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
            if self._pending_generator_state is not None:
                # A resumed run continues the saved stream instead of restarting it.
                self._generator.set_state(self._pending_generator_state)
                self._pending_generator_state = None
        return self._generator

    def point_generator_state(self) -> dict:
        """Seed and current state of the point-selection generator."""
        if self._generator is not None:
            state = self._generator.get_state()
        else:
            state = self._pending_generator_state
        return dict(seed=self._seed, state=state)

    def load_point_generator_state(self, record: dict) -> None:
        """Restore a state saved by :meth:`point_generator_state`."""
        if record.get("seed") != self._seed:
            raise ValueError(
                f"saved point-generator seed {record.get('seed')} does not match "
                f"the configured seed {self._seed}")
        if record.get("state") is None:
            return
        if self._generator is not None:
            self._generator.set_state(record["state"])
        else:
            self._pending_generator_state = record["state"]

    @staticmethod
    def _positive_gt_masks(
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        device: torch.device,
    ) -> Tensor:
        """Full-resolution transformed masks, one per positive ROI.

        Point targets come from these, not from the 28x28 training target, so
        supervision is not limited by the coarse grid. The released masks are
        only read.
        """
        collected = []
        for result, gt in zip(sampling_results, batch_gt_instances):
            assigned = result.pos_assigned_gt_inds
            if len(assigned) == 0:
                continue
            masks = gt.masks[assigned.cpu().numpy()].to_tensor(
                dtype=torch.float32, device=device)
            collected.append(masks)
        if not collected:
            return torch.zeros(0, 1, 1, device=device)
        return torch.cat(collected, dim=0)

    def mask_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        mask_results = super().mask_loss(
            stage, x, sampling_results, batch_gt_instances)
        if not self.refine_enabled:
            return mask_results

        coarse = mask_results["mask_preds"]
        device = coarse.device
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        labels = torch.cat([res.pos_gt_labels for res in sampling_results]) \
            if coarse.shape[0] else torch.zeros(0, dtype=torch.long, device=device)
        gt_masks = self._positive_gt_masks(
            sampling_results, batch_gt_instances, device)

        refine_out = self.point_refine_head.loss_and_refine(
            coarse_logits=coarse, rois=pos_rois, labels=labels,
            fine_features=x[self.fine_level], gt_masks=gt_masks, stage=stage,
            generator=self._get_generator(device))
        refine_out.pop("_refined", None)
        # Merged into loss_mask so the parent applies this stage's weight once.
        mask_results["loss_mask"].update(refine_out)
        return mask_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Unchanged; refinement happens where labels are available."""
        return super()._mask_forward(stage, x, rois)

    def predict_mask(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                     results_list: InstanceList, rescale: bool = False
                     ) -> InstanceList:
        if not self.refine_enabled:
            return super().predict_mask(x, batch_img_metas, results_list, rescale)

        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas, mask_rois.device, task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)

        num_per_img = [len(res) for res in results_list]
        labels = torch.cat([res.labels for res in results_list])
        aug_masks = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            refined = self.point_refine_head.predict_refine(
                coarse_logits=mask_results["mask_preds"], rois=mask_rois,
                labels=labels, fine_features=x[self.fine_level], stage=stage)
            # Sigmoid once, per stage, before the baseline merge.
            split = refined.split(num_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in split])

        merged = [merge_aug_masks([mask[i] for mask in aug_masks],
                                  batch_img_metas[i])
                  for i in range(len(batch_img_metas))]
        return self.mask_head[-1].predict_by_feat(
            mask_preds=merged, results_list=results_list,
            batch_img_metas=batch_img_metas, rcnn_test_cfg=self.test_cfg,
            rescale=rescale, activate_map=True)
