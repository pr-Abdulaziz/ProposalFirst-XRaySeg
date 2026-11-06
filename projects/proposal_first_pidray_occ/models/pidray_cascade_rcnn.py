"""MMDetection 3.2 Cascade Mask R-CNN with the PIDray R0 node."""

import copy
from collections.abc import Sequence

import torch
from torch import Tensor

from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.registry import MODELS


@MODELS.register_module()
class PIDrayCascadeRCNN(CascadeRCNN):
    """PIDray R0/R1 detector with an explicit training-gate policy.

    Args:
        r0_head: Config for :class:`PIDrayR0Head`.
        training_gate: ``'legacy'`` reproduces the released detached R0 gate;
            ``'none'`` keeps all training samples, which is the safe setting for
            the proposed PIDray--Occ hybrid.  Inference output gating remains
            enabled in both modes.
    """

    def __init__(self, *args, r0_head: dict, training_gate: str = 'none',
                 gate_during_inference: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if training_gate not in {'legacy', 'none'}:
            raise ValueError("training_gate must be either 'legacy' or 'none'.")
        self.r0_head = MODELS.build(r0_head)
        self.training_gate = training_gate
        self.gate_during_inference = gate_during_inference

    def _detection_loss(self, features: tuple, data_samples: Sequence) -> dict:
        losses = {}
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal', self.test_cfg.rpn
            )
            rpn_samples = copy.deepcopy(data_samples)
            for sample in rpn_samples:
                sample.gt_instances.labels = torch.zeros_like(
                    sample.gt_instances.labels
                )
            rpn_losses, proposals = self.rpn_head.loss_and_predict(
                features, rpn_samples, proposal_cfg=proposal_cfg
            )
            for key in list(rpn_losses):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            proposals = [sample.proposals for sample in data_samples]

        losses.update(self.roi_head.loss(features, proposals, data_samples))
        return losses

    @staticmethod
    def _legacy_gradient_scale(losses: dict, factor: float) -> dict:
        """Match PIDray's forward-value-preserving gradient scale."""
        scaled = {}
        for name, value in losses.items():
            if isinstance(value, Tensor):
                mean = value.mean()
                scaled[name] = (
                    mean * factor - (mean * factor).detach() + mean.detach()
                )
            elif isinstance(value, list):
                scaled[name] = sum(
                    item.mean() * factor
                    - (item.mean() * factor).detach()
                    + item.mean().detach()
                    for item in value
                )
            else:
                raise TypeError(f'{name} is not a tensor or list of tensors.')
        return scaled

    def loss(self, batch_inputs: Tensor, batch_data_samples: Sequence) -> dict:
        features = self.extract_feat(batch_inputs)
        r0_logits = self.r0_head(features, batch_data_samples)
        losses = self.r0_head.loss_by_logits(r0_logits, batch_data_samples)

        if self.training_gate == 'none':
            losses.update(self._detection_loss(features, batch_data_samples))
            return losses

        with torch.no_grad():
            r0_scores = torch.sigmoid(r0_logits)
            selected = torch.where(r0_scores > self.r0_head.threshold)[0]

        if len(selected) > 0:
            gated_features = tuple(feature[selected] for feature in features)
            gated_samples = [batch_data_samples[index] for index in selected.tolist()]
        else:
            # The released implementation still computes R1's forward values
            # on the full batch, but multiplies its gradients by zero.
            gated_features = features
            gated_samples = batch_data_samples

        detection_losses = self._detection_loss(gated_features, gated_samples)
        factor = float(len(selected)) / len(batch_data_samples)
        losses.update(self._legacy_gradient_scale(detection_losses, factor))
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: Sequence,
                rescale: bool = True) -> Sequence:
        features = self.extract_feat(batch_inputs)
        r0_scores = self.r0_head.predict(features, batch_data_samples)

        if batch_data_samples[0].get('proposals', None) is None:
            proposals = self.rpn_head.predict(
                features, batch_data_samples, rescale=False
            )
        else:
            proposals = [sample.proposals for sample in batch_data_samples]

        results = self.roi_head.predict(
            features, proposals, batch_data_samples, rescale=rescale
        )
        for score, result in zip(r0_scores, results):
            if self.gate_during_inference and score <= self.r0_head.threshold:
                result.scores.zero_()

        output = self.add_pred_to_datasample(batch_data_samples, results)
        for score, sample in zip(r0_scores, output):
            sample.set_metainfo({'r0_score': float(score.detach().cpu())})
        return output
