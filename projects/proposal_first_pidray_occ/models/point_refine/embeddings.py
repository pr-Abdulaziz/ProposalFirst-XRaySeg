"""Token construction for the Method 09 point-refinement head.

One token per sampled point, formed by summing four contributions that are kept
as separate branches so their input scaling stays inspectable:

    image_i = Linear(256, 128)(f_i)
    mask_i  = Linear(64, 128)(GELU(Linear(C, 64)(m_i)))
    pos_i   = Linear(128, 128)(GELU(Linear(2, 128)(u_i)))
    token_i = LayerNorm(image_i + mask_i + pos_i + level_embedding[level])

``f_i`` is a sampled P2 feature, ``m_i`` the *original* coarse logits for that
cascade stage, and ``u_i`` the point's normalised coordinates inside its ROI.
The coarse logits are fed in unnormalised: the same tensor drives uncertainty
selection, and rescaling it would change which points get selected.

Class count is a constructor argument, never a literal, so the module transfers
to a dataset with a different ``C`` without edits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PointTokenEmbedding(nn.Module):
    """Build ``[N, K, D]`` tokens from per-point features, logits and coordinates.

    Args:
        num_classes: Foreground class count ``C``. 15 for PIXray.
        feature_channels: Channels of the sampled fine feature map (P2): 256.
        embed_dim: Token width ``D``: 128.
        mask_hidden: Width of the mask branch's hidden layer: 64.
        num_levels: Refinement output levels receiving their own embedding: 2,
            for the 56 and 112 grids.
        num_stages: Cascade stages. Stage conditioning is built only when
            ``use_stage_embedding`` is set, and is off by default.
        use_stage_embedding: Optional per-cascade-stage embedding. Off in the
            primary Method 09 configuration.
    """

    def __init__(
        self,
        num_classes: int,
        feature_channels: int = 256,
        embed_dim: int = 128,
        mask_hidden: int = 64,
        num_levels: int = 2,
        num_stages: int = 3,
        use_stage_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.use_stage_embedding = use_stage_embedding

        self.image_proj = nn.Linear(feature_channels, embed_dim)
        self.mask_proj = nn.Sequential(
            nn.Linear(num_classes, mask_hidden), nn.GELU(),
            nn.Linear(mask_hidden, embed_dim))
        self.pos_proj = nn.Sequential(
            nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.level_embedding = nn.Parameter(torch.empty(num_levels, embed_dim))
        self.stage_embedding = (
            nn.Parameter(torch.empty(num_stages, embed_dim))
            if use_stage_embedding else None)
        self.norm = nn.LayerNorm(embed_dim)
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        nn.init.normal_(self.level_embedding, std=0.02)
        if self.stage_embedding is not None:
            nn.init.normal_(self.stage_embedding, std=0.02)

    def branch_norms(
        self, features: Tensor, coarse_logits: Tensor, coords: Tensor
    ) -> dict:
        """Mean L2 norm of each branch, for pilot logging.

        Reported so a branch that dominates or collapses is visible rather than
        silently corrected later by a gate or a rescale.
        """
        with torch.no_grad():
            return {
                "image": self.image_proj(features).norm(dim=-1).mean().item(),
                "mask": self.mask_proj(coarse_logits).norm(dim=-1).mean().item(),
                "pos": self.pos_proj(coords).norm(dim=-1).mean().item(),
                "level": self.level_embedding.norm(dim=-1).mean().item(),
            }

    def forward(
        self,
        features: Tensor,
        coarse_logits: Tensor,
        coords: Tensor,
        level: int,
        stage: int | None = None,
    ) -> Tensor:
        """
        Args:
            features: ``[N, K, feature_channels]`` sampled P2 features.
            coarse_logits: ``[N, K, C]`` sampled original coarse logits.
            coords: ``[N, K, 2]`` normalised ROI coordinates, ``(x, y)`` in [0,1].
            level: Refinement level index, ``0`` for the 56 grid, ``1`` for 112.
            stage: Cascade stage index; used only with stage conditioning.

        Returns:
            ``[N, K, embed_dim]`` layer-normalised tokens.
        """
        if not 0 <= level < self.num_levels:
            raise ValueError(f"level {level} outside [0, {self.num_levels})")
        if coarse_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"coarse_logits has {coarse_logits.shape[-1]} channels, expected "
                f"{self.num_classes}; the head must be built with the dataset's "
                "class count rather than a literal")
        if coords.shape[-1] != 2:
            raise ValueError(f"coords last axis must be 2 (x, y), got {coords.shape[-1]}")

        token = (self.image_proj(features)
                 + self.mask_proj(coarse_logits)
                 + self.pos_proj(coords)
                 + self.level_embedding[level].view(1, 1, -1))
        if self.stage_embedding is not None:
            if stage is None:
                raise ValueError("stage conditioning is enabled but stage is None")
            token = token + self.stage_embedding[stage].view(1, 1, -1)
        return self.norm(token)
