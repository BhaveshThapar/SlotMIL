"""Trainable instance encoders for the end-to-end (uncached) path.

Only used by the W1 fast path, where a small CNN encodes MedMNIST3D slices
directly so the slot module can be validated in minutes without waiting on a
feature cache. The real CT pipeline uses frozen DINOv2 features from
``slotmil.features`` instead, and passes ``encoder=None``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SliceCNNEncoder(nn.Module):
    """Small 2D CNN applied per slice, shared across instances in a bag.

    Input  ``(B, N, H*W)`` flattened slices, output ``(B, N, out_dim)``.
    """

    def __init__(self, slice_hw: int, out_dim: int = 256, width: int = 32, in_ch: int = 1):
        super().__init__()
        self.slice_hw = slice_hw
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(width * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = w = self.slice_hw
        z = x.reshape(b * n, 1, h, w)
        z = self.net(z).flatten(1)
        z = self.proj(z)
        z = z.view(b, n, self.out_dim)
        if pad_mask is not None:
            # BatchNorm has already seen the padded slices; zeroing here keeps
            # them from reaching the pooling module as spurious instances.
            z = z.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        return z
