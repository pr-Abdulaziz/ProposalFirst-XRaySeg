"""End-to-end proposal-first PIDray--Occ detector."""

import copy
import time
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS

from .pidray_cascade_rcnn import PIDrayCascadeRCNN


@MODELS.register_module()
class ProposalFirstPIDrayOcc(PIDrayCascadeRCNN):
    """PIDray R0/R1 detector with a proposal-conditioned bilayer SAM branch.

    PIDray and SAM retain separate encoders. PIDray features produce RPN
    proposals, Cascade boxes, classes, and learned RoI prompts. SAM supplies a
    dense image embedding only when mask processing is required at inference.
    During training, predicted gates are intentionally disabled so every
    matched positive RoI receives mask supervision.
    """

    def __init__(
        self,
        *args,
        sam_image_encoder: dict,
        sam_positional_embedding: dict,
        sam_encoder_frozen: bool = False,
        sam_positional_frozen: bool = True,
        sam_decoder_frozen: bool = False,
        det_score_threshold: float = 0.05,
        combined_score_threshold: float = 0.0,
        max_sam_proposals: int = 70,
        use_r0_screening: bool = True,
        use_proposal_screening: bool = True,
        training_gate: str = "none",
        profile_inference: bool = False,
        r0_rescue_enabled: bool = False,
        rescue_det_threshold: float = 0.20,
        rescue_topk: int = 5,
        **kwargs,
    ) -> None:
        if training_gate != "none":
            raise ValueError(
                'Hybrid training must use training_gate="none"; predicted R0 '
                "gates must not suppress positive mask supervision."
            )
        kwargs.pop("gate_during_inference", None)
        super().__init__(
            *args,
            training_gate="none",
            gate_during_inference=False,
            **kwargs,
        )
        if max_sam_proposals < 1:
            raise ValueError("max_sam_proposals must be positive.")
        if not 0.0 <= rescue_det_threshold <= 1.0:
            raise ValueError("rescue_det_threshold must be between 0 and 1.")
        if rescue_topk < 1:
            raise ValueError("rescue_topk must be positive.")

        self.sam_image_encoder = MODELS.build(sam_image_encoder)
        self.sam_positional_embedding = MODELS.build(sam_positional_embedding)
        self.sam_encoder_frozen = sam_encoder_frozen
        self.sam_positional_frozen = sam_positional_frozen
        self.sam_decoder_frozen = sam_decoder_frozen
        self.det_score_threshold = det_score_threshold
        self.combined_score_threshold = combined_score_threshold
        self.max_sam_proposals = max_sam_proposals
        self.use_r0_screening = use_r0_screening
        self.use_proposal_screening = use_proposal_screening
        self.profile_inference = profile_inference
        self.r0_rescue_enabled = r0_rescue_enabled
        self.rescue_det_threshold = rescue_det_threshold
        self.rescue_topk = rescue_topk

        if self.sam_encoder_frozen:
            self._freeze(self.sam_image_encoder)
        if self.sam_positional_frozen:
            self._freeze(self.sam_positional_embedding)
        if self.sam_decoder_frozen:
            self._freeze(self.roi_head.mask_head.mask_decoder)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.sam_encoder_frozen:
            self.sam_image_encoder.eval()
        if self.sam_positional_frozen:
            self.sam_positional_embedding.eval()
        if self.sam_decoder_frozen:
            self.roi_head.mask_head.mask_decoder.eval()
        return self

    def _sam_positional_encoding(self, height: int, width: int) -> Tensor:
        module = self.sam_positional_embedding.shared_image_embedding
        device = module.positional_embedding.device
        dtype = module.positional_embedding.dtype
        grid = torch.ones((height, width), device=device, dtype=dtype)
        y_embed = (grid.cumsum(dim=0) - 0.5) / height
        x_embed = (grid.cumsum(dim=1) - 0.5) / width
        encoding = self.sam_positional_embedding(
            torch.stack([x_embed, y_embed], dim=-1)
        )
        return encoding.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _synchronize(inputs: Tensor) -> None:
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)

    def _record_stage(
        self,
        timings: dict[str, float],
        name: str,
        started_at: float,
        inputs: Tensor,
    ) -> float:
        if not self.profile_inference:
            return started_at
        self._synchronize(inputs)
        finished_at = time.perf_counter()
        timings[name] = (finished_at - started_at) * 1000.0
        return finished_at

    def extract_sam_embeddings(self, batch_inputs: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.sam_image_encoder(batch_inputs)
        image_embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
        if image_embeddings.ndim != 4:
            raise ValueError(
                "SAM encoder must return a BCHW image-embedding tensor; got "
                f"{tuple(image_embeddings.shape)}."
            )
        positional = self._sam_positional_encoding(
            image_embeddings.shape[-2], image_embeddings.shape[-1]
        )
        positional = positional.repeat(image_embeddings.shape[0], 1, 1, 1)
        return image_embeddings, positional

    def _rpn_loss_and_predict(
        self, features: tuple, data_samples: Sequence
    ) -> tuple[dict, list]:
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_samples = copy.deepcopy(data_samples)
        for sample in rpn_samples:
            sample.gt_instances.labels = torch.zeros_like(sample.gt_instances.labels)
        rpn_losses, proposals = self.rpn_head.loss_and_predict(
            features, rpn_samples, proposal_cfg=proposal_cfg
        )
        for key in list(rpn_losses):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        return rpn_losses, proposals

    def loss(self, batch_inputs: Tensor, batch_data_samples: Sequence) -> dict:
        detector_features = self.extract_feat(batch_inputs)
        r0_logits = self.r0_head(detector_features, batch_data_samples)
        losses = self.r0_head.loss_by_logits(r0_logits, batch_data_samples)

        rpn_losses, proposals = self._rpn_loss_and_predict(
            detector_features, batch_data_samples
        )
        losses.update(rpn_losses)

        has_positive = any(
            len(sample.gt_instances) > 0 for sample in batch_data_samples
        )
        if has_positive:
            image_embeddings, positional_embeddings = self.extract_sam_embeddings(
                batch_inputs
            )
        else:
            image_embeddings = positional_embeddings = None
        roi_losses = self.roi_head.loss(
            detector_features,
            proposals,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=positional_embeddings,
        )
        losses.update(roi_losses)
        return losses

    def _screen_results(
        self, r0_scores: Tensor, results_list: list
    ) -> tuple[list, list[dict[str, object]]]:
        screened = []
        screening_stats = []
        for r0_score, result in zip(r0_scores, results_list):
            detections_before = len(result)
            combined = result.scores * r0_score
            r0_positive = bool((r0_score > self.r0_head.threshold).item())

            if self.use_r0_screening and not r0_positive:
                if self.r0_rescue_enabled:
                    indices = torch.where(result.scores >= self.rescue_det_threshold)[0]
                    if len(indices) > self.rescue_topk:
                        ranking = torch.argsort(result.scores[indices], descending=True)
                        indices = indices[ranking[: self.rescue_topk]]
                    route = "rescue" if len(indices) else "rejected"
                else:
                    indices = torch.empty(
                        0, dtype=torch.long, device=result.scores.device
                    )
                    route = "rejected"
            else:
                keep = torch.ones_like(result.scores, dtype=torch.bool)
                if self.use_proposal_screening:
                    keep &= result.scores >= self.det_score_threshold
                    keep &= combined >= self.combined_score_threshold
                indices = torch.where(keep)[0]
                if (
                    self.use_proposal_screening
                    and len(indices) > self.max_sam_proposals
                ):
                    ranking_scores = (
                        combined[indices]
                        if self.use_r0_screening
                        else result.scores[indices]
                    )
                    ranking = torch.argsort(ranking_scores, descending=True)
                    indices = indices[ranking[: self.max_sam_proposals]]
                route = "normal" if len(indices) else "rejected"

            result = result[indices]
            result.hybrid_scores = combined[indices]
            screened.append(result)
            screening_stats.append(
                {
                    "screening_route": route,
                    "rescue_executed": route == "rescue",
                    "detections_before_screening": detections_before,
                    "detections_after_screening": len(result),
                }
            )
        return screened, screening_stats

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: Sequence, rescale: bool = True
    ) -> Sequence:
        timings: dict[str, float] = {}
        if self.profile_inference:
            if batch_inputs.is_cuda:
                torch.cuda.reset_peak_memory_stats(batch_inputs.device)
            self._synchronize(batch_inputs)
        total_started_at = stage_started_at = time.perf_counter()

        detector_features = self.extract_feat(batch_inputs)
        stage_started_at = self._record_stage(
            timings,
            "pidray_backbone_attfpn",
            stage_started_at,
            batch_inputs,
        )
        r0_scores = self.r0_head.predict(detector_features, batch_data_samples)
        stage_started_at = self._record_stage(
            timings, "r0", stage_started_at, batch_inputs
        )

        if batch_data_samples[0].get("proposals", None) is None:
            proposals = self.rpn_head.predict(
                detector_features, batch_data_samples, rescale=False
            )
        else:
            proposals = [sample.proposals for sample in batch_data_samples]
        stage_started_at = self._record_stage(
            timings, "rpn", stage_started_at, batch_inputs
        )

        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        results = self.roi_head.predict_bbox(
            detector_features,
            batch_img_metas,
            proposals,
            rcnn_test_cfg=self.roi_head.test_cfg,
            rescale=False,
        )
        stage_started_at = self._record_stage(
            timings, "cascade_bbox", stage_started_at, batch_inputs
        )
        results, screening_stats = self._screen_results(r0_scores, results)
        stage_started_at = self._record_stage(
            timings, "proposal_screening", stage_started_at, batch_inputs
        )
        execution_flags = [len(result) > 0 for result in results]
        execute_sam = any(execution_flags)

        if execute_sam:
            image_embeddings, positional_embeddings = self.extract_sam_embeddings(
                batch_inputs
            )
        else:
            image_embeddings = positional_embeddings = None
        if self.profile_inference and not execute_sam:
            timings["sam_encoder"] = 0.0
        else:
            stage_started_at = self._record_stage(
                timings, "sam_encoder", stage_started_at, batch_inputs
            )

        results = self.roi_head.predict_mask(
            detector_features,
            batch_img_metas,
            results,
            rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=positional_embeddings,
        )
        stage_started_at = self._record_stage(
            timings, "bilayer_mask_branch", stage_started_at, batch_inputs
        )
        output = self.add_pred_to_datasample(batch_data_samples, results)
        self._record_stage(
            timings, "output_postprocess", stage_started_at, batch_inputs
        )
        if self.profile_inference:
            self._synchronize(batch_inputs)
            timings["model_total"] = (time.perf_counter() - total_started_at) * 1000.0
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(batch_inputs.device) / 1024**2
                if batch_inputs.is_cuda
                else None
            )
        else:
            peak_gpu_memory_mb = None
        for score, executed, stats, result, sample in zip(
            r0_scores, execution_flags, screening_stats, results, output
        ):
            hard_routes = result.get("mask_hard_routes", None)
            conditional_hard_count = (
                int(hard_routes.sum().item()) if hard_routes is not None else 0
            )
            conditional_total = len(hard_routes) if hard_routes is not None else 0
            sample.set_metainfo(
                {
                    "r0_score": float(score.detach().cpu()),
                    "sam_executed": executed,
                    **stats,
                    "stage_times_ms": timings,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                    "conditional_hard_count": conditional_hard_count,
                    "conditional_total_count": conditional_total,
                }
            )
        return output
