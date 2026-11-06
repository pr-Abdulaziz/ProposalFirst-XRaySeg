"""Compact transformer that refines point logits inside a single ROI.

Two pre-LayerNorm encoder blocks over the sampled points of one ROI, then a
linear decoder to absolute class logits. Attention is bidirectional and is
confined to a single ROI: the ROI axis is the batch axis of the attention call,
so points of different ROIs, and of different images, never attend to each
other.

Positional information reaches the model only through the coordinate branch of
the token embedding. Nothing here depends on the order in which points were
sampled, which is arbitrary.

This is not a Mask Transfiner reproduction: there is no quadtree, no
incoherence detection and no residual-logit correction. It is a compact
transformer over uncertainty-selected points.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _EncoderBlock(nn.Module):
    """Pre-LayerNorm self-attention block with a residual around each sublayer."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        normed = self.norm_attn(tokens)
        attended, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + self.dropout(attended)
        return tokens + self.ffn(self.norm_ffn(tokens))


class PointRefineTransformer(nn.Module):
    """Refine per-point class logits within each ROI independently.

    Args:
        num_classes: Foreground class count ``C``; the decoder width.
        embed_dim: Token width ``D``: 128.
        num_heads: Attention heads: 4, giving head dimension 32.
        num_blocks: Encoder blocks: 2.
        ffn_dim: Feed-forward width: 512.
        dropout / attn_dropout: Both 0 in the initial method.
        output_std: Standard deviation for the decoder's normal init. Small but
            nonzero, so the head starts near neutral without the dead gradients
            a fully zeroed head would give.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        output_std: float = 0.001,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim {embed_dim} must divide by num_heads {num_heads}")
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.blocks = nn.ModuleList([
            _EncoderBlock(embed_dim, num_heads, ffn_dim, dropout, attn_dropout)
            for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, num_classes)
        self.output_std = output_std
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                nn.init.zeros_(module.out_proj.bias)
        # Small nonzero, not zero: a zeroed decoder starts with no gradient
        # signal through its own weights.
        nn.init.normal_(self.decoder.weight, std=self.output_std)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            tokens: ``[N, K, D]``; ``N`` is the ROI count, not the image batch.
            key_padding_mask: Optional ``[N, K]`` boolean, ``True`` marking
                positions to ignore, matching torch's convention. A row that is
                entirely ``True`` would make softmax undefined, so it is
                rejected rather than silently producing NaNs.

        Returns:
            ``[N, K, C]`` absolute point logits.
        """
        if tokens.dim() != 3:
            raise ValueError(f"expected [N, K, D] tokens, got {tuple(tokens.shape)}")
        if key_padding_mask is not None:
            if key_padding_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"key_padding_mask {tuple(key_padding_mask.shape)} does not "
                    f"match tokens {tuple(tokens.shape[:2])}")
            if bool(key_padding_mask.all(dim=1).any()):
                raise ValueError(
                    "a ROI has every point masked; softmax would be undefined")
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        return self.decoder(self.norm_out(tokens))
