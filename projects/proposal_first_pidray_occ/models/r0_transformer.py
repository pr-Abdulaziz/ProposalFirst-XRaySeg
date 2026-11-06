"""Minimal checkpoint-compatible transformer used by PIDray R0.

PIDray configures zero encoder layers and two decoder layers with decoder
self-attention removed.  This module preserves that released computation and
its parameter hierarchy while avoiding legacy MMCV dependencies.
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                dropout: float = 0.1, activation: str = 'relu') -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, memory: Tensor,
                memory_mask: Tensor | None = None,
                memory_key_padding_mask: Tensor | None = None,
                pos: Tensor | None = None,
                query_pos: Tensor | None = None, **kwargs) -> Tensor:
        attended = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(feedforward))


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int,
                 norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                memory_key_padding_mask: Tensor | None = None,
                pos: Tensor | None = None,
                query_pos: Tensor | None = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output.unsqueeze(0)


class Transformer(nn.Module):
    """Released PIDray R0 cross-attention transformer."""

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_decoder_layers: int = 2, dim_feedforward: int = 1024,
                 dropout: float = 0.1, activation: str = 'relu') -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, nn.LayerNorm(d_model)
        )
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, src: Tensor, query_embed: Tensor, pos_embed: Tensor,
                mask: Tensor | None = None) -> tuple:
        batch_size, channels, height, width = src.shape
        memory = src.flatten(2).permute(2, 0, 1)
        position = pos_embed.flatten(2).permute(2, 0, 1)
        query = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        key_padding_mask = mask.flatten(1) if mask is not None else None
        target = torch.zeros_like(query)
        hidden = self.decoder(
            target,
            memory,
            memory_key_padding_mask=key_padding_mask,
            pos=position,
            query_pos=query,
        )
        memory_map = memory[:height * width].permute(1, 2, 0).view(
            batch_size, channels, height, width
        )
        return hidden.transpose(1, 2), memory_map


def _get_activation_fn(name: str):
    if name == 'relu':
        return F.relu
    if name == 'gelu':
        return F.gelu
    if name == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu/glu, not {name}.')
