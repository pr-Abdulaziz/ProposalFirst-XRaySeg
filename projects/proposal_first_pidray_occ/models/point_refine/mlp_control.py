"""Capacity-matched control for Method 09's point mixer.

The transformer's claim is that *points within an ROI should exchange
information*. The only way to test that claim is against a control that keeps
everything else fixed -- the same token embedding, the same depth, the same
pre-LayerNorm residual structure, the same feed-forward width, the same
decoder, the same point sampling, the same targets, the same schedule -- and
removes exactly one thing: the ability of one point to see another.

``PointRefineMLP`` therefore replaces each block's multi-head attention
sublayer with a position-wise two-layer MLP of the same shape budget. Every
point is processed independently: the module is equivariant to any permutation
of the point axis and its output at point *k* is a function of token *k* alone.

Parameter budget, embed_dim=128, ffn_dim=512, num_blocks=2, C=15
--------------------------------------------------------------
  attention sublayer   in_proj 128*384 + 384 + out_proj 128*128 + 128 = 66,048
  mixing MLP sublayer  128*256 + 256    + 256*128        + 128       = 65,920
                                                            per block  -128
  whole module         398,479 against the transformer's 398,735       -256

That is -0.064 percent. An exact match is not reachable with clean layer
widths, and padding the control with a parameter that exists only to equalise a
count would be a worse control than a 0.06 percent difference. The counts are
reported, not rounded away.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS

from .head import PointRefineHead


class _MixerBlock(nn.Module):
    """Pre-LayerNorm block whose first sublayer is position-wise, not attention.

    Structurally identical to ``_EncoderBlock``: same two norms, same two
    residual connections, same FFN. ``nn.MultiheadAttention`` is the only
    substitution.
    """

    def __init__(self, embed_dim: int, mixing_dim: int, ffn_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.mixing = nn.Sequential(
            nn.Linear(embed_dim, mixing_dim), nn.GELU(),
            nn.Linear(mixing_dim, embed_dim))
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        # key_padding_mask is accepted for interface parity and is genuinely
        # unnecessary here: a position-wise module cannot read a padded
        # neighbour, so masking one out would change nothing.
        tokens = tokens + self.dropout(self.mixing(self.norm_attn(tokens)))
        return tokens + self.ffn(self.norm_ffn(tokens))


class PointRefineMLP(nn.Module):
    """Drop-in replacement for :class:`PointRefineTransformer` with no attention.

    Same signature, same output shape ``[N, K, C]``, same initialisation policy
    including the small nonzero decoder init, so the two arms start from
    equivalent output distributions.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,          # accepted and unused: interface parity
        num_blocks: int = 2,
        ffn_dim: int = 512,
        mixing_dim: int = 256,
        dropout: float = 0.0,
        output_std: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mixing_dim = mixing_dim
        self.blocks = nn.ModuleList([
            _MixerBlock(embed_dim, mixing_dim, ffn_dim, dropout)
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
        nn.init.normal_(self.decoder.weight, std=self.output_std)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"expected [N, K, D] tokens, got {tuple(tokens.shape)}")
        if key_padding_mask is not None and key_padding_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"key_padding_mask {tuple(key_padding_mask.shape)} does not "
                f"match tokens {tuple(tokens.shape[:2])}")
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        return self.decoder(self.norm_out(tokens))


@MODELS.register_module()
class PointRefineMLPHead(PointRefineHead):
    """Method 09's head with the transformer replaced by the matched MLP.

    Only ``_build_mixer`` is overridden. Point selection, coordinate
    conversion, feature sampling, target sampling, the loss, the diagnostic
    keys and the empty-ROI path are inherited unchanged, so the two arms differ
    in the mixer and nothing else.

    The attribute is still named ``transformer`` so the parent's forward path
    and every existing test apply verbatim; the class name and the state-dict
    contents make the distinction, not the attribute name.
    """

    def _build_mixer(self, num_classes, embed_dim, num_heads, num_blocks,
                     ffn_dim) -> nn.Module:
        return PointRefineMLP(
            num_classes=num_classes, embed_dim=embed_dim, num_heads=num_heads,
            num_blocks=num_blocks, ffn_dim=ffn_dim)
