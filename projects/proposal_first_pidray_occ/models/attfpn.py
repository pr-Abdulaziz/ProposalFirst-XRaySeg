"""MMDetection 3.2 port of the released PIDray AttFPN.

The selective-attention equations and module names are intentionally kept
equivalent to ``references/legacy_pidray/mmdet/models/necks/attfpn.py``. Only the registry and
base FPN integration are adapted to MMDetection 3.2/MMEngine.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import ContextBlock
from torch import nn

from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS


class BatchOneSafeBatchNorm2d(nn.BatchNorm2d):
    """Use stored statistics for a singleton pooled training tensor.

    PIDray's channel-attention block applies BatchNorm after global pooling.
    The original detector trained with multiple images per GPU, but the SAM
    hybrid uses one image per GPU.  Falling back to running statistics only
    for a single value per channel preserves all BatchNorm parameter/state
    names and leaves the original behavior unchanged for larger batches.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values_per_channel = inputs.numel() // inputs.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(inputs)


class ChannelAttention(nn.Module):
    """Channel-wise selection across the five PIDray pyramid levels."""

    def __init__(self, channels: int = 256, num_levels: int = 5) -> None:
        super().__init__()
        hidden_channels = 128
        self.channels = channels
        self.num_levels = num_levels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            BatchOneSafeBatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
        )
        self.fcs = nn.ModuleList(
            [nn.Conv2d(hidden_channels, channels, 1) for _ in range(num_levels)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.sum(dim=1)
        descriptor = self.fc(self.gap(features))
        attention = torch.cat([layer(descriptor) for layer in self.fcs], dim=1)
        attention = attention.view(
            -1, self.num_levels, self.channels, 1, 1
        )
        return self.softmax(attention)


class SpatialAttention(nn.Module):
    """Spatial selection across the five PIDray pyramid levels."""

    def __init__(self, num_levels: int = 5, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        # Keep the released attribute name for checkpoint compatibility.
        self.conv = nn.ModuleList(
            [nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            for _ in range(num_levels)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.sum(dim=1)
        average = features.mean(dim=1, keepdim=True)
        maximum = features.max(dim=1, keepdim=True).values
        descriptor = torch.cat([average, maximum], dim=1)
        attention = torch.cat([layer(descriptor) for layer in self.conv], dim=1)
        return self.softmax(attention).unsqueeze(2)


class SelectiveAttention(nn.Module):
    """Gather, attend, refine, and residually restore one pyramid level."""

    def __init__(self, refine_level: int, channels: int = 256,
                num_levels: int = 5) -> None:
        super().__init__()
        self.refine_level = refine_level
        self.channels = channels
        self.num_levels = num_levels
        self.channel_att = ChannelAttention(channels, num_levels)
        self.spatial_att = SpatialAttention(num_levels)
        self.refine = ContextBlock(channels, 1.0 / 16)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        gather_size = inputs[self.refine_level].shape[2:]
        gathered = []
        for level, feature in enumerate(inputs):
            if level < self.refine_level:
                feature = F.adaptive_max_pool2d(feature, gather_size)
            else:
                feature = F.interpolate(feature, size=gather_size, mode='nearest')
            gathered.append(feature)

        features = torch.stack(gathered, dim=1)
        channel_features = (features * self.channel_att(features)).sum(dim=1)
        spatial_features = (features * self.spatial_att(features)).sum(dim=1)
        refined = self.refine(channel_features + spatial_features)
        return refined + inputs[self.refine_level]


@MODELS.register_module()
class AttFPN(FPN):
    """Exact five-level, 256-channel PIDray attention pyramid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.out_channels != 256 or self.num_outs != 5:
            raise ValueError(
                'PIDray AttFPN requires out_channels=256 and num_outs=5; '
                f'got {self.out_channels} and {self.num_outs}.'
            )
        self.selective_attention_0 = SelectiveAttention(0)
        self.selective_attention_1 = SelectiveAttention(1)
        self.selective_attention_2 = SelectiveAttention(2)
        self.selective_attention_3 = SelectiveAttention(3)
        self.selective_attention_4 = SelectiveAttention(4)

    def enhance_feature(self, inputs: Sequence[torch.Tensor]) -> tuple:
        return tuple(
            getattr(self, f'selective_attention_{level}')(inputs)
            for level in range(5)
        )

    def forward(self, inputs: tuple) -> tuple:
        return self.enhance_feature(super().forward(inputs))
