"""
Pure FNO / DeepONet baseline without physics branch or hybrid fusion.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..cf_hpino_model import CFHPINOConfig, DeepONetBackbone, FNOBackbone, OperatorBackbone


class PureFNO(nn.Module):
    """Operator-only model for ablation vs CF-HPINO."""

    def __init__(self, cfg: Optional[CFHPINOConfig] = None, backbone: str = "fno"):
        super().__init__()
        self.cfg = cfg or CFHPINOConfig()
        if backbone == "deeponet":
            self.cfg.backbone = OperatorBackbone.DEEPONET
            self.operator = DeepONetBackbone(self.cfg)
        else:
            self.cfg.backbone = OperatorBackbone.FNO
            self.operator = FNOBackbone(self.cfg)
        self.act = nn.Softplus()

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, dict]:
        if self.cfg.backbone == OperatorBackbone.FNO:
            b, n, _ = coords.shape
            h, w = self.cfg.n_spatial, self.cfg.n_temporal
            grid = coords.view(b, h, w, 2) if n == h * w else coords
            values, _ = self.operator(params, grid)
            prices = self.act(values.reshape(b, -1))
        else:
            values, _ = self.operator(params, coords)
            prices = self.act(values)
        if return_features:
            return {"prices": prices, "operator_values": prices}
        return prices


def build_pure_fno(backbone: str = "fno", **kwargs) -> PureFNO:
    cfg = CFHPINOConfig(**{k: v for k, v in kwargs.items() if hasattr(CFHPINOConfig, k)})
    return PureFNO(cfg, backbone=backbone)
