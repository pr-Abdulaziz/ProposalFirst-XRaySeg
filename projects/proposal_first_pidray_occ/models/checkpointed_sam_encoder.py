"""SAM ViT image encoder with activation checkpointing (campaign amendment for Method 06, 2026-09-17).

Method 06 ran out of memory at batch 4 on a 24 GB GPU. By user decision, the only change is
activation checkpointing inside the frozen-weight SAM image encoder: each transformer layer's
activations are recomputed during the backward pass instead of being stored. Weights, LoRA
adaptation, inputs, losses and optimiser are unchanged. Recomputation replays the saved RNG state
(``preserve_rng_state``), so LoRA dropout draws the same mask, and replays the autocast state, so
the computation is the same as the uncheckpointed encoder. Checkpointing is active only in training
mode with gradients enabled; inference runs the layers directly.
"""
from __future__ import annotations

import types

from torch.utils.checkpoint import checkpoint

from mmdet.occlu.models import MMPretrainSamVisionEncoder
from mmdet.registry import MODELS


@MODELS.register_module()
class CheckpointedMMPretrainSamVisionEncoder(MMPretrainSamVisionEncoder):
    """``MMPretrainSamVisionEncoder`` whose ViT layers use non-reentrant activation checkpointing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mmpretrain.models.backbones.vit_sam import ViTSAM

        vits = [m for m in self.vision_encoder.modules() if isinstance(m, ViTSAM)]
        if len(vits) != 1:
            raise RuntimeError(f"expected exactly one ViTSAM inside the encoder, found {len(vits)}")
        self.checkpointed_layers = 0
        for layer in vits[0].layers:
            plain = layer.forward

            def forward(self, x, _plain=plain):
                import torch
                if self.training and torch.is_grad_enabled():
                    return checkpoint(_plain, x, use_reentrant=False, preserve_rng_state=True)
                return _plain(x)

            layer.forward = types.MethodType(forward, layer)
            self.checkpointed_layers += 1
