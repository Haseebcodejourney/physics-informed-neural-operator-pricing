"""
Fractional calculus utilities for physics-informed losses.

Re-exports Caputo L1 from the model module and adds Grünwald-Letnikov weights
for finite-difference residual assembly.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .cf_hpino_model import CaputoL1FractionalDerivative


def grunwald_letnikov_weights(alpha: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """
    GL coefficients w_j^{(α)} for j = 0, ..., n_steps-1 via recurrence:
        w_0 = 1,  w_j = w_{j-1} * (1 - (α+1)/j)
    """
    w = torch.zeros(n_steps, device=device, dtype=torch.float64)
    w[0] = 1.0
    for j in range(1, n_steps):
        w[j] = w[j - 1] * (1.0 - (alpha + 1.0) / j)
    return w.flip(0).to(torch.float32)


def caputo_l1_time_derivative(
    u: torch.Tensor,
    dt: float,
    alpha: float,
    n_steps: int = 20,
) -> torch.Tensor:
    """Functional wrapper around CaputoL1FractionalDerivative (no nn.Parameter)."""
    op = CaputoL1FractionalDerivative(alpha=alpha, n_steps=n_steps)
    return op(u, dt)


def gl_time_derivative(
    u: torch.Tensor,
    dt: float,
    alpha: float,
    n_steps: int = 20,
) -> torch.Tensor:
    """
    Grünwald-Letnikov fractional derivative along last dimension of u.

    Args:
        u: (..., T)
    """
    *prefix, t_len = u.shape
    w = grunwald_letnikov_weights(alpha, min(n_steps, t_len), u.device)
    n = w.shape[0]
    u_pad = F.pad(u, (n, 0), value=0.0)
    kernel = (w / dt**alpha).view(1, 1, -1)
    flat = u_pad.reshape(-1, 1, u_pad.shape[-1])
    out = F.conv1d(flat, kernel, padding=0).reshape(*prefix, -1)
    pad_len = t_len - out.shape[-1]
    if pad_len > 0:
        out = F.pad(out, (pad_len, 0), value=0.0)
    return out[..., :t_len]
