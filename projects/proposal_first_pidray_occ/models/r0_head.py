"""PIDray coarse normal/prohibited image node (R0)."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.models.layers import SinePositionalEncoding
from mmdet.registry import MODELS

from .r0_transformer import Transformer


@MODELS.register_module()
class PIDrayR0Head(nn.Module):
    """Image-level prohibited-item confidence head.

    The released code feeds the last AttFPN output directly into R0.  In the
    paper's notation this tensor is used as ``F^0`` even though the code obtains
    it as ``features[-1]``; no new aggregation operator is inserted here.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        threshold: float = 0.5,
        loss_weight: float = 0.1,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        if loss_type not in {"bce", "focal"}:
            raise ValueError('loss_type must be either "bce" or "focal".')
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be between 0 and 1.")
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.confidence_transformer = Transformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            num_decoder_layers=2,
        )
        self.confidence_fc = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(1, hidden_dim)
        self.positional_encoding = SinePositionalEncoding(
            num_feats=hidden_dim // 2, normalize=True
        )
        self.confidence_loss = nn.BCEWithLogitsLoss()

    def _classification_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.loss_type == "bce":
            return self.confidence_loss(logits, targets)

        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        target_probabilities = (
            probabilities * targets + (1 - probabilities) * (1 - targets)
        )
        alpha = (
            self.focal_alpha * targets
            + (1 - self.focal_alpha) * (1 - targets)
        )
        return (
            alpha
            * (1 - target_probabilities).pow(self.focal_gamma)
            * cross_entropy
        ).mean()

    @staticmethod
    def _padding_mask(feature: Tensor, data_samples: Sequence) -> Tensor:
        batch_size = feature.shape[0]
        batch_height, batch_width = data_samples[0].batch_input_shape
        mask = feature.new_ones((batch_size, batch_height, batch_width))
        for index, sample in enumerate(data_samples):
            image_height, image_width = sample.img_shape[:2]
            mask[index, :image_height, :image_width] = 0
        return F.interpolate(
            mask.unsqueeze(1), size=feature.shape[-2:]
        ).to(torch.bool).squeeze(1)

    def forward(self, features: Sequence[Tensor], data_samples: Sequence) -> Tensor:
        feature = features[-1]
        mask = self._padding_mask(feature, data_samples)
        position = self.positional_encoding(mask)
        hidden = self.confidence_transformer(
            feature, self.query_embed.weight, position
        )[0]
        return self.confidence_fc(hidden[-1].squeeze(1)).squeeze(1)

    def loss_by_logits(self, logits: Tensor, data_samples: Sequence) -> dict:
        targets = logits.new_tensor(
            [float(len(sample.gt_instances) > 0) for sample in data_samples]
        )
        return {
            "loss_r0": self.loss_weight
            * self._classification_loss(logits, targets)
        }

    def loss(self, features: Sequence[Tensor], data_samples: Sequence) -> dict:
        return self.loss_by_logits(self(features, data_samples), data_samples)

    def predict(self, features: Sequence[Tensor], data_samples: Sequence) -> Tensor:
        return torch.sigmoid(self(features, data_samples))
