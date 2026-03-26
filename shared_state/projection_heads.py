"""Projection head architectures for the multimodal shared space.

These are the canonical definitions used by both training
(scripts/vision/train_projection.py) and inference
(shared_state/multimodal_encoder.py).

Architecture:
  ProjectionText: 384 -> 512 -> 384 (identity-initialized final layer)
  ProjectionImg:  768 -> 512 -> 384 (Xavier-initialized)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionText(nn.Module):
    """384 -> 512 -> 384 with identity initialization on final layer."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 384),
        )
        self._identity_init()

    def _identity_init(self):
        """Initialize final layer close to identity."""
        with torch.no_grad():
            final = self.layers[-1]
            final.weight.zero_()
            final.weight[:384, :384].copy_(torch.eye(384))
            final.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ProjectionImg(nn.Module):
    """768 -> 512 -> 384 with Xavier initialization."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 384),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
