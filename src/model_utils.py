"""
Accuracy utilities: Fourier coordinate encoding, EMA weight averaging.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class FourierCoordinateEncoding(nn.Module):
    """Multi-scale sin/cos features on (S_norm, t_norm) for sharper price surfaces."""

    def __init__(self, n_freq: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freq = n_freq
        self.include_input = include_input
        self.out_dim = (2 if include_input else 0) + 4 * n_freq

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.include_input:
            feats.append(coords)
        for k in range(self.n_freq):
            freq = (2.0**k) * torch.pi
            feats.append(torch.sin(freq * coords))
            feats.append(torch.cos(freq * coords))
        return torch.cat(feats, dim=-1)


class ModelEMA:
    """Exponential moving average of model weights for stabler, more accurate inference."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
