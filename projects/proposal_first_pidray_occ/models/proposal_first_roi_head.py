"""Cascade proposal bridge for the PIDray--Occ bilayer SAM branch."""

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.occlu.occlu import OcPrompterAnchorMaskHead
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList


@MODELS.register_module()
class ProposalFirstBilayerMaskHead(OcPrompterAnchorMaskHead):
    """Occ bilayer mask head with an explicit full-image target contract.

    The first decoder predicts the overlapping-occluder mask. Its returned
    image and prompt features guide the second decoder, which predicts the
    prohibited-item mask. Both targets are full-image masks, matching the
    released Occ implementation.
    """

    def __init__(
        self, *args, mask_size=(512, 512), overlap_loss_weight: float = 0.25, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mask_size = tuple(mask_size)
        self.overlap_loss_weight = overlap_loss_weight

    def get_targets(
        self,
        sampling_results: list[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg=None,
    ) -> tuple[Tensor, Tensor]:
        pos_proposals = [result.pos_priors for result in sampling_results]
        pos_gt_inds = [result.pos_assigned_gt_inds for result in sampling_results]
        device = pos_proposals[0].device
        mask_targets = []
        overlap_targets = []

        for assigned, gt_instances in zip(pos_gt_inds, batch_gt_instances):
            if "bo_masks" not in gt_instances:
                raise KeyError(
                    "Hybrid training requires gt_instances.bo_masks. Use "
                    "PIDrayOcclusionDataset with OcLoadAnnotations and "
                    "OcPackDetInputs on *_occ.json annotations."
                )
            if len(assigned) == 0:
                shape = (0,) + self.mask_size
                masks = torch.zeros(shape, device=device, dtype=torch.float32)
                overlaps = torch.zeros_like(masks)
            else:
                masks = gt_instances.masks[assigned.cpu()].to_tensor(
                    dtype=torch.float32, device=device
                )
                overlaps = gt_instances.bo_masks[assigned.cpu()].to_tensor(
                    dtype=torch.float32, device=device
                )
                if masks.shape[-2:] != self.mask_size:
                    masks = F.interpolate(
                        masks[:, None], size=self.mask_size, mode="nearest"
                    )[:, 0]
                    overlaps = F.interpolate(
                        overlaps[:, None], size=self.mask_size, mode="nearest"
                    )[:, 0]
            mask_targets.append(masks)
            overlap_targets.append(overlaps)

        return torch.cat(mask_targets), torch.cat(overlap_targets)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        mask_bo_preds: Tensor,
        sampling_results: list[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg=None,
    ) -> dict:
        mask_targets, overlap_targets = self.get_targets(
            sampling_results, batch_gt_instances, rcnn_train_cfg
        )
        pos_labels = torch.cat([result.pos_gt_labels for result in sampling_results])

        mask_preds = F.interpolate(
            mask_preds, size=self.mask_size, mode="bilinear", align_corners=False
        )
        mask_bo_preds = F.interpolate(
            mask_bo_preds, size=self.mask_size, mode="bilinear", align_corners=False
        )

        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
            loss_overlap = mask_bo_preds.sum()
        else:
            labels = torch.zeros_like(pos_labels) if self.class_agnostic else pos_labels
            loss_mask = self.loss_mask(mask_preds, mask_targets, labels)
            loss_overlap = self.loss_mask(mask_bo_preds, overlap_targets, labels)

        return {
            "loss_mask": {"loss_mask": loss_mask},
            "mask_targets": mask_targets,
            "loss_mask_bo": {"loss_mask_bo": self.overlap_loss_weight * loss_overlap},
            "mask_bo_targets": overlap_targets,
        }


@MODELS.register_module()
class ProposalFirstCascadeRoIHead(CascadeRoIHead):
    """Three-stage Cascade boxes followed by one bilayer SAM mask branch.

    Box classification and regression are refined at all Cascade stages. Mask
    supervision is applied only after the final stage, using matched positive
    RoIs during training. At inference the detector screens final detections
    before calling :meth:`predict_mask`.
    """

    def __init__(
        self,
        *args,
        max_mask_rois_per_image: int = 64,
        use_occluder_decoder: bool = True,
        use_residual_guidance: bool = True,
        use_prompt_adapter: bool = False,
        adapter_hidden_channels: int = 32,
        use_mask_booster: bool = False,
        mask_booster_hidden_channels: int = 8,
        mask_booster_strength: float = 1.0,
        hard_aware_loss: bool = False,
        dice_loss_weight: float = 1.0,
        occlusion_loss_weight: float = 1.0,
        boundary_loss_weight: float = 0.25,
        conditional_bisam: bool = False,
        difficulty_threshold: float = 0.05,
        difficulty_gate_threshold: float = 0.5,
        difficulty_loss_weight: float = 0.25,
        difficulty_hidden_channels: int = 64,
        **kwargs,
    ) -> None:
        if max_mask_rois_per_image < 1:
            raise ValueError("max_mask_rois_per_image must be positive.")
        if use_residual_guidance and not use_occluder_decoder:
            raise ValueError("Residual guidance requires the occluder decoder.")
        if use_mask_booster and not use_residual_guidance:
            raise ValueError("The mask booster requires residual-guided decoding.")
        if conditional_bisam and not use_residual_guidance:
            raise ValueError("Conditional BiSAM requires residual-guided decoding.")
        for name, value in (
            ("adapter_hidden_channels", adapter_hidden_channels),
            ("mask_booster_hidden_channels", mask_booster_hidden_channels),
            ("difficulty_hidden_channels", difficulty_hidden_channels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("mask_booster_strength", mask_booster_strength),
            ("dice_loss_weight", dice_loss_weight),
            ("occlusion_loss_weight", occlusion_loss_weight),
            ("boundary_loss_weight", boundary_loss_weight),
            ("difficulty_loss_weight", difficulty_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        for name, value in (
            ("difficulty_threshold", difficulty_threshold),
            ("difficulty_gate_threshold", difficulty_gate_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.max_mask_rois_per_image = max_mask_rois_per_image
        self.use_occluder_decoder = use_occluder_decoder
        self.use_residual_guidance = use_residual_guidance
        self.use_prompt_adapter = use_prompt_adapter
        self.use_mask_booster = use_mask_booster
        self.mask_booster_strength = mask_booster_strength
        self.hard_aware_loss = hard_aware_loss
        self.dice_loss_weight = dice_loss_weight
        self.occlusion_loss_weight = occlusion_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.conditional_bisam = conditional_bisam
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_gate_threshold = difficulty_gate_threshold
        self.difficulty_loss_weight = difficulty_loss_weight
        super().__init__(*args, **kwargs)

        if self.use_prompt_adapter:
            self.prompt_adapter = nn.Sequential(
                nn.Conv2d(256, adapter_hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_hidden_channels, 256, 1),
            )
            nn.init.zeros_(self.prompt_adapter[-1].weight)
            nn.init.zeros_(self.prompt_adapter[-1].bias)
        if self.use_mask_booster:
            self.mask_booster = nn.Sequential(
                nn.Conv2d(2, mask_booster_hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mask_booster_hidden_channels, 1, 1),
            )
            nn.init.zeros_(self.mask_booster[-1].weight)
            nn.init.zeros_(self.mask_booster[-1].bias)
        if self.conditional_bisam:
            self.difficulty_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, difficulty_hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(difficulty_hidden_channels, 1),
            )

    def _limit_mask_sampling(
        self, sampling_results: list[SamplingResult]
    ) -> list[SamplingResult]:
        """Bound SAM memory while retaining the Cascade's random positives."""
        limited_results = []
        for result in sampling_results:
            count = len(result.pos_priors)
            if count <= self.max_mask_rois_per_image:
                limited_results.append(result)
                continue
            limited = copy.copy(result)
            selection = slice(0, self.max_mask_rois_per_image)
            for field in (
                "pos_inds",
                "pos_priors",
                "pos_is_gt",
                "pos_assigned_gt_inds",
                "pos_gt_labels",
                "pos_gt_bboxes",
            ):
                setattr(limited, field, getattr(result, field)[selection])
            limited.num_pos = self.max_mask_rois_per_image
            limited_results.append(limited)
        return limited_results

    def init_mask_head(self, mask_roi_extractor, mask_head) -> None:
        if isinstance(mask_head, (list, tuple)):
            if len(mask_head) != 1:
                raise ValueError("The bilayer branch uses exactly one mask head.")
            mask_head = mask_head[0]
        self.mask_head = MODELS.build(mask_head)
        if mask_roi_extractor is None:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor[-1]
        else:
            if isinstance(mask_roi_extractor, (list, tuple)):
                mask_roi_extractor = mask_roi_extractor[-1]
            self.share_roi_extractor = False
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)

    def _mask_forward(
        self,
        x: tuple[Tensor],
        rois: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        hard_routes: Tensor | None = None,
    ) -> dict:
        mask_feats = self.mask_roi_extractor(
            x[: self.mask_roi_extractor.num_inputs], rois
        )
        roi_img_ids = rois[:, 0]
        if self.use_prompt_adapter:
            mask_feats = mask_feats + self.prompt_adapter(mask_feats)

        gate_logits = (
            self.difficulty_gate(mask_feats).flatten()
            if self.conditional_bisam
            else None
        )
        if self.conditional_bisam:
            if hard_routes is None:
                hard_routes = gate_logits.sigmoid() >= self.difficulty_gate_threshold
            hard_routes = hard_routes.to(device=mask_feats.device, dtype=torch.bool)
            if len(hard_routes) != len(mask_feats):
                raise ValueError("hard_routes must contain one value per mask RoI.")
            mask_preds, overlap_preds, iou_predictions = self._conditional_decode(
                mask_feats,
                roi_img_ids,
                hard_routes,
                image_embeddings,
                image_positional_embeddings,
            )
        else:
            overlap_preds, residual_image, residual_prompt = self._occluder_decode(
                mask_feats,
                roi_img_ids,
                image_embeddings,
                image_positional_embeddings,
            )
            if not self.use_residual_guidance:
                residual_image = residual_prompt = None
            mask_preds, iou_predictions = self._target_decode(
                mask_feats,
                roi_img_ids,
                image_embeddings,
                image_positional_embeddings,
                residual_image,
                residual_prompt,
            )
            if self.use_mask_booster:
                mask_preds = self._boost_mask(mask_preds)
            if overlap_preds is None:
                overlap_preds = torch.zeros_like(mask_preds)
        return {
            "mask_preds": mask_preds,
            "mask_preds_bo": overlap_preds,
            "iou_predictions": iou_predictions,
            "mask_feats": mask_feats,
            "difficulty_gate_logits": gate_logits,
            "hard_routes": hard_routes,
        }

    def _decoder_inputs(
        self,
        roi_img_ids: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compact image embeddings and remap RoI ids for any RoI subset."""
        unique_ids, compact_ids = torch.unique(
            roi_img_ids.long(), sorted=True, return_inverse=True
        )
        return (
            compact_ids,
            image_embeddings[unique_ids],
            image_positional_embeddings[unique_ids],
        )

    def _decode(
        self,
        mask_feats: Tensor,
        roi_img_ids: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        *,
        if_bo: bool,
        residual_image: Tensor | None = None,
        residual_prompt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        compact_ids, compact_images, compact_positions = self._decoder_inputs(
            roi_img_ids, image_embeddings, image_positional_embeddings
        )
        return self.mask_head(
            mask_feats,
            image_embeddings=compact_images,
            image_positional_embeddings=compact_positions,
            roi_img_ids=compact_ids,
            if_bo=if_bo,
            res_img_embeddings=residual_image,
            res_point_embeddings=residual_prompt,
        )

    def _occluder_decode(
        self,
        mask_feats: Tensor,
        roi_img_ids: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if not self.use_occluder_decoder:
            return None, None, None
        overlap, _, residual_image, residual_prompt = self._decode(
            mask_feats,
            roi_img_ids,
            image_embeddings,
            image_positional_embeddings,
            if_bo=True,
        )
        return overlap, residual_image, residual_prompt

    def _target_decode(
        self,
        mask_feats: Tensor,
        roi_img_ids: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        residual_image: Tensor | None,
        residual_prompt: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        masks, iou, _, _ = self._decode(
            mask_feats,
            roi_img_ids,
            image_embeddings,
            image_positional_embeddings,
            if_bo=False,
            residual_image=residual_image,
            residual_prompt=residual_prompt,
        )
        return masks, iou

    def _boost_mask(self, mask_logits: Tensor) -> Tensor:
        probabilities = mask_logits.sigmoid()
        uncertainty = probabilities * (1.0 - probabilities)
        correction = self.mask_booster(torch.cat([mask_logits, uncertainty], dim=1))
        return mask_logits + self.mask_booster_strength * correction

    def _conditional_decode(
        self,
        mask_feats: Tensor,
        roi_img_ids: Tensor,
        hard_routes: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use one decoder for easy RoIs and guided BiSAM for hard RoIs."""
        outputs: dict[str, Tensor] = {}
        for route_name, route_mask in (
            ("easy", ~hard_routes),
            ("hard", hard_routes),
        ):
            indices = torch.where(route_mask)[0]
            if len(indices) == 0:
                continue
            route_feats = mask_feats[indices]
            route_ids = roi_img_ids[indices]
            if route_name == "hard":
                overlap, residual_image, residual_prompt = self._occluder_decode(
                    route_feats,
                    route_ids,
                    image_embeddings,
                    image_positional_embeddings,
                )
            else:
                overlap = residual_image = residual_prompt = None
            masks, iou = self._target_decode(
                route_feats,
                route_ids,
                image_embeddings,
                image_positional_embeddings,
                residual_image,
                residual_prompt,
            )
            if route_name == "hard" and self.use_mask_booster:
                masks = self._boost_mask(masks)
            outputs[f"{route_name}_indices"] = indices
            outputs[f"{route_name}_masks"] = masks
            outputs[f"{route_name}_iou"] = iou
            outputs[f"{route_name}_overlap"] = (
                overlap if overlap is not None else torch.zeros_like(masks)
            )

        first_route = "hard" if "hard_masks" in outputs else "easy"
        reference_masks = outputs[f"{first_route}_masks"]
        reference_iou = outputs[f"{first_route}_iou"]
        masks = reference_masks.new_zeros(
            (len(mask_feats),) + reference_masks.shape[1:]
        )
        overlaps = reference_masks.new_zeros(masks.shape)
        iou = reference_iou.new_zeros((len(mask_feats),) + reference_iou.shape[1:])
        for route_name in ("easy", "hard"):
            if f"{route_name}_masks" not in outputs:
                continue
            indices = outputs[f"{route_name}_indices"]
            masks = masks.index_copy(0, indices, outputs[f"{route_name}_masks"])
            overlaps = overlaps.index_copy(0, indices, outputs[f"{route_name}_overlap"])
            iou = iou.index_copy(0, indices, outputs[f"{route_name}_iou"])
        return masks, overlaps, iou

    def mask_loss(
        self,
        x: tuple[Tensor],
        sampling_results: list[SamplingResult],
        batch_gt_instances: InstanceList,
        image_embeddings: Tensor | None,
        image_positional_embeddings: Tensor | None,
    ) -> dict:
        sampling_results = self._limit_mask_sampling(sampling_results)
        pos_rois = bbox2roi([result.pos_priors for result in sampling_results])
        if len(pos_rois) == 0:
            zero = x[0].sum() * 0
            return {
                "loss_mask": {"loss_mask": zero},
                "loss_mask_bo": {"loss_mask_bo": zero},
            }
        if image_embeddings is None or image_positional_embeddings is None:
            raise ValueError("Positive mask RoIs require SAM image embeddings.")

        mask_targets, overlap_targets = self.mask_head.get_targets(
            sampling_results, batch_gt_instances, self.train_cfg[-1]
        )
        hard_routes = None
        if self.conditional_bisam:
            hard_routes = (
                self._occlusion_ratio(mask_targets, overlap_targets)
                >= self.difficulty_threshold
            )
        mask_results = self._mask_forward(
            x,
            pos_rois,
            image_embeddings,
            image_positional_embeddings,
            hard_routes=hard_routes,
        )
        targets = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            mask_bo_preds=mask_results["mask_preds_bo"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[-1],
        )
        if self.conditional_bisam:
            overlap_logits = F.interpolate(
                mask_results["mask_preds_bo"],
                size=self.mask_head.mask_size,
                mode="bilinear",
                align_corners=False,
            )
            if hard_routes.any():
                labels = torch.cat(
                    [result.pos_gt_labels for result in sampling_results]
                )[hard_routes]
                if self.mask_head.class_agnostic:
                    labels = torch.zeros_like(labels)
                overlap_loss = self.mask_head.loss_mask(
                    overlap_logits[hard_routes],
                    overlap_targets[hard_routes],
                    labels,
                )
            else:
                overlap_loss = overlap_logits.sum() * 0.0
            targets["loss_mask_bo"] = {
                "loss_mask_bo": self.mask_head.overlap_loss_weight * overlap_loss
            }
        if self.hard_aware_loss:
            targets["loss_mask"] = {
                "loss_mask": self._hard_aware_mask_loss(
                    mask_results["mask_preds"],
                    mask_targets,
                    overlap_targets,
                )
            }
        mask_results.update(
            loss_mask=targets["loss_mask"], loss_mask_bo=targets["loss_mask_bo"]
        )
        if self.conditional_bisam:
            gate_labels = hard_routes.to(
                dtype=mask_results["difficulty_gate_logits"].dtype
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                mask_results["difficulty_gate_logits"], gate_labels
            )
            mask_results["loss_difficulty_gate"] = {
                "loss_difficulty_gate": self.difficulty_loss_weight * gate_loss
            }
        return mask_results

    @staticmethod
    def _occlusion_ratio(mask_targets: Tensor, overlap_targets: Tensor) -> Tensor:
        """Fraction of the target instance that the occluder actually covers.

        ``overlap_targets`` holds the occluding object's own mask, so its raw
        area measures how large that object is, not how much of the target it
        hides. The two are intersected first: a large object lying beside a
        target no longer counts as maximally occluded, and the result is bounded
        by 1 by construction rather than by clamping.

        Both tensors are ``(N, H, W)`` float masks over the same grid, so the
        product is the per-instance overlap region.
        """
        target_area = mask_targets.flatten(1).sum(1).clamp_min(1.0)
        occluded_area = (mask_targets * overlap_targets).flatten(1).sum(1)
        return (occluded_area / target_area).clamp(0.0, 1.0)

    @staticmethod
    def _boundary_ratio(mask_targets: Tensor) -> Tensor:
        masks = mask_targets[:, None]
        dilated = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
        boundary_area = (dilated - eroded).clamp_min(0).flatten(1).sum(1)
        target_area = mask_targets.flatten(1).sum(1).clamp_min(1.0)
        return (boundary_area / target_area).clamp(0.0, 1.0)

    def _hard_aware_mask_loss(
        self,
        mask_preds: Tensor,
        mask_targets: Tensor,
        overlap_targets: Tensor,
    ) -> Tensor:
        logits = F.interpolate(
            mask_preds,
            size=self.mask_head.mask_size,
            mode="bilinear",
            align_corners=False,
        )
        targets = mask_targets[:, None]
        weights = (
            1.0
            + self.occlusion_loss_weight
            * self._occlusion_ratio(mask_targets, overlap_targets)
            + self.boundary_loss_weight * self._boundary_ratio(mask_targets)
        )
        weights = weights / weights.mean().clamp_min(1e-6)
        bce = (
            F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            .flatten(1)
            .mean(1)
        )
        probabilities = logits.sigmoid()
        intersection = (probabilities * targets).flatten(1).sum(1)
        denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return (weights * (bce + self.dice_loss_weight * dice)).mean()

    def loss(
        self,
        x: tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: list[DetDataSample],
        image_embeddings: Tensor | None = None,
        image_positional_embeddings: Tensor | None = None,
    ) -> dict:
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = (
            unpack_gt_instances(batch_data_samples)
        )
        results_list = copy.deepcopy(rpn_results_list)
        losses = {}
        final_sampling_results = None

        for stage in range(self.num_stages):
            self.current_stage = stage
            sampling_results = []
            for image_index, results in enumerate(results_list):
                results.priors = results.pop("bboxes")
                assignment = self.bbox_assigner[stage].assign(
                    results,
                    batch_gt_instances[image_index],
                    batch_gt_instances_ignore[image_index],
                )
                sampling_results.append(
                    self.bbox_sampler[stage].sample(
                        assignment,
                        results,
                        batch_gt_instances[image_index],
                        feats=[level[image_index][None] for level in x],
                    )
                )

            bbox_results = self.bbox_loss(stage, x, sampling_results)
            stage_weight = self.stage_loss_weights[stage]
            for name, value in bbox_results["loss_bbox"].items():
                losses[f"s{stage}.{name}"] = (
                    value * stage_weight if "loss" in name else value
                )

            final_sampling_results = sampling_results
            if stage < self.num_stages - 1:
                with torch.no_grad():
                    results_list = self.bbox_head[stage].refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas
                    )
                    if results_list is None:
                        break

        if final_sampling_results is not None:
            mask_results = self.mask_loss(
                x,
                final_sampling_results,
                batch_gt_instances,
                image_embeddings,
                image_positional_embeddings,
            )
            losses.update(mask_results["loss_mask"])
            losses.update(mask_results["loss_mask_bo"])
            if "loss_difficulty_gate" in mask_results:
                losses.update(mask_results["loss_difficulty_gate"])
        return losses

    def predict_mask(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        results_list: InstanceList,
        rescale: bool = False,
        image_embeddings: Tensor | None = None,
        image_positional_embeddings: Tensor | None = None,
    ) -> InstanceList:
        bboxes = [result.bboxes for result in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )
        if image_embeddings is None or image_positional_embeddings is None:
            raise ValueError("Retained proposals require SAM image embeddings.")

        mask_results = self._mask_forward(
            x, mask_rois, image_embeddings, image_positional_embeddings
        )
        counts = [len(result) for result in results_list]
        target_preds = mask_results["mask_preds"].split(counts, 0)
        overlap_preds = mask_results["mask_preds_bo"].split(counts, 0)

        overlap_results = self.mask_head.predict_by_feat(
            mask_preds=overlap_preds,
            results_list=copy.deepcopy(results_list),
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )
        results_list = self.mask_head.predict_by_feat(
            mask_preds=target_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )
        for result, overlap in zip(results_list, overlap_results):
            result.overlap_masks = overlap.masks
        if mask_results["hard_routes"] is not None:
            hard_routes = mask_results["hard_routes"].split(counts, 0)
            for result, routes in zip(results_list, hard_routes):
                result.mask_hard_routes = routes
        return results_list

    def predict(
        self,
        x: tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList,
        rescale: bool = False,
        image_embeddings: Tensor | None = None,
        image_positional_embeddings: Tensor | None = None,
    ) -> InstanceList:
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=False,
        )
        return self.predict_mask(
            x,
            batch_img_metas,
            results_list,
            rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
