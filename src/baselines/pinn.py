"""
Standard Physics-Informed Neural Network baseline (no neural operator).

MLP maps (S, t, θ) -> V with same physics loss as CF-HPINO (see cf_hpino_loss.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    n_params: int = 8
    n_coords: int = 2
    hidden: List[int] = None  # type: ignore
    activation: str = "tanh"

    def __post_init__(self):
        if self.hidden is None:
            self.hidden = [256, 256, 256, 256]


class StandardPINN(nn.Module):
    """Vanilla PINN: concat(params, coords) -> price."""

    def __init__(self, cfg: Optional[PINNConfig] = None):
        super().__init__()
        self.cfg = cfg or PINNConfig()
        in_dim = self.cfg.n_params + self.cfg.n_coords
        dims = [in_dim] + self.cfg.hidden + [1]
        layers: List[nn.Module] = []
        act = nn.Tanh() if self.cfg.activation == "tanh" else nn.GELU()
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)
        self.out_act = nn.Softplus()

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        return_features: bool = False,
    ):
        p = params.unsqueeze(1).expand(-1, coords.shape[1], -1)
        x = torch.cat([coords, p], dim=-1)
        v = self.out_act(self.net(x.reshape(-1, x.shape[-1])).view(
            coords.shape[0], coords.shape[1]
        ))
        if return_features:
            return {"prices": v, "operator_values": v}
        return v

    def greeks(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        compute: Tuple[str, ...] = ("delta", "gamma", "vega"),
    ):
        coords = coords.detach().requires_grad_(True)
        prices = self.forward(params, coords)
        out = {"price": prices}
        if "delta" in compute or "gamma" in compute:
            g = torch.autograd.grad(
                prices.sum(), coords, create_graph=True, retain_graph=True
            )[0][..., 0]
        if "delta" in compute:
            out["delta"] = g
        if "gamma" in compute:
            out["gamma"] = torch.autograd.grad(
                g.sum(), coords, retain_graph=True
            )[0][..., 0]
        if "vega" in compute:
            pv = params.detach().clone().requires_grad_(True)
            out["vega"] = torch.autograd.grad(
                self.forward(pv, coords).sum(), pv, retain_graph=True
            )[0][:, 1]
        return out


def build_pinn(**kwargs) -> StandardPINN:
    return StandardPINN(PINNConfig(**kwargs))
