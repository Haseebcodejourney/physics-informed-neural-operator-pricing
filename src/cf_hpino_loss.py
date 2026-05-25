"""
CF-HPINO hybrid loss: data + physics + operator + boundary/terminal.

Supports:
  - Black-Scholes PDE residual
  - Fractional BS with Caputo time derivative
  - Merton jump-diffusion PIDE (diffusion + integral jump term, trapezoid quad)
  - American early-exercise penalty (V >= intrinsic)

Adaptive weighting via learnable log-variance (Kendall et al.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cf_hpino_model import CF_HPINO
from .fractional_ops import caputo_l1_time_derivative


class LossPDEType(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    FRACTIONAL_BS = "fractional_bs"
    MERTON = "merton"


@dataclass
class LossConfig:
    pde_type: LossPDEType = LossPDEType.BLACK_SCHOLES
    # Fixed weights (used if adaptive=False)
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lambda_operator: float = 0.05
    lambda_boundary: float = 0.5
    lambda_american: float = 0.2
    adaptive: bool = True
    # Collocation
    n_collocation: int = 4096
    # Scaling for normalized coordinates
    S_min: float = 20.0
    S_max: float = 300.0
    # Merton quadrature
    jump_quad_points: int = 32
    jump_std_mult: float = 5.0
    # Accuracy-oriented data loss
    relative_data_weight: float = 0.5
    huber_delta: float = 1.0
    payoff_weight: float = 2.0
    use_log_spatial: bool = True
    lambda_market: float = 2.0


class AdaptiveLossWeights(nn.Module):
    """Homoscedastic uncertainty weighting: L = Σ exp(-s_i)*L_i + s_i."""

    def __init__(self, n_terms: int = 8):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_terms))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        keys = list(losses.keys())
        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = {}
        for i, k in enumerate(keys):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * losses[k] + self.log_vars[i]
            weights[k] = precision.item()
        return total, weights


class CFHPINOLoss(nn.Module):
    """
    Composite loss for CF-HPINO training.

    Expects batch dict with:
        params, coords, prices (optional), grid (optional)
    """

    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.adaptive = AdaptiveLossWeights(n_terms=8) if self.cfg.adaptive else None
        self._geom_meta: Dict[str, float] = {}

    def set_geometry_meta(self, meta: Dict[str, float]) -> None:
        """Call once per dataset so S denorm matches log-spaced training grid."""
        self._geom_meta = dict(meta)
        if meta.get("log_spatial"):
            self.cfg.use_log_spatial = True
            self.cfg.S_min = meta.get("S_min", self.cfg.S_min)
            self.cfg.S_max = meta.get("S_max", self.cfg.S_max)

    # ----- coordinate transforms -----

    def _denorm_coords(
        self,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map normalized coords to physical (S, t). Supports log-spaced S grids."""
        T = params[:, 3:4]
        t = coords[..., 1:2] * T.unsqueeze(1)

        if self.cfg.use_log_spatial and self._geom_meta.get("log_spatial"):
            log_min = self._geom_meta.get("log_S_min", math.log(self.cfg.S_min))
            log_max = self._geom_meta.get("log_S_max", math.log(self.cfg.S_max))
            log_S = coords[..., 0:1] * (log_max - log_min) + log_min
            S = torch.exp(log_S)
        else:
            S = coords[..., 0:1] * (self.cfg.S_max - self.cfg.S_min) + self.cfg.S_min

        if coords.dim() == 2:
            S = S.squeeze(-1) if S.shape[-1] == 1 else S
            t = t.squeeze(-1) if t.shape[-1] == 1 else t
        else:
            S = S.squeeze(-1)
            t = t.squeeze(-1)
        return S, t

    # ----- individual losses -----

    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Absolute Huber + relative error — matches true prices across scales."""
        err = pred - target
        abs_loss = F.huber_loss(pred, target, delta=self.cfg.huber_delta)
        rel = (err**2) / (target**2 + 1e-2)
        w = self.cfg.relative_data_weight
        return (1.0 - w) * abs_loss + w * rel.mean()

    def market_quote_loss(
        self,
        pred_grid_flat: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Match predicted surface at t≈0 to listed mids (nearest strike on grid).

        Uses `pred` from the main forward on the full lattice — avoids FNO on
        scattered quote coordinates.
        """
        grid = batch["grid"]
        quote_coords = batch["quote_coords"]
        quote_prices = batch["quote_prices"]
        quote_mask = batch.get("quote_mask")
        B, H, W, _ = grid.shape
        pred_g = pred_grid_flat.view(B, H, W)
        s_grid = grid[:, :, 0, 0]

        loss_sum = torch.tensor(0.0, device=pred_grid_flat.device)
        count = 0
        for b in range(B):
            nq = quote_coords.shape[1]
            for q in range(nq):
                if quote_mask is not None and quote_mask[b, q] < 0.5:
                    continue
                sq = quote_coords[b, q, 0]
                i = torch.argmin((s_grid[b] - sq).abs())
                pred_q = pred_g[b, i, 0]
                tgt = quote_prices[b, q]
                err = pred_q - tgt
                # Relative error stabilizes training across OTM/ITM dollar scales
                loss_sum = loss_sum + (err**2) / (tgt**2 + 1.0)
                count += 1
        return loss_sum / max(count, 1)

    def operator_loss(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encourage operator branch to match fused target (consistency)."""
        if "operator_values" not in features or "prices" not in features:
            return torch.tensor(0.0, device=features.get("prices", torch.zeros(1)).device)
        return F.mse_loss(features["operator_values"], features["prices"].detach())

    def _fno_grid(self, model: CF_HPINO, params: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Build normalized (S, t) lattice matching FNO (H, W)."""
        B = params.shape[0]
        h, w = model.cfg.n_spatial, model.cfg.n_temporal
        s = torch.linspace(0.0, 1.0, h, device=device)
        t = torch.linspace(0.0, 1.0, w, device=device)
        Sg, Tg = torch.meshgrid(s, t, indexing="ij")
        grid = torch.stack([Sg, Tg], dim=-1)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    def boundary_loss(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Terminal payoff + S=0 Dirichlet for calls (FNO-compatible grid).
        """
        B = params.shape[0]
        K = params[:, 2]
        grid = self._fno_grid(model, params, device)

        # Terminal slice t_norm = 1
        coords_T = grid.clone()
        coords_T[..., 1] = 1.0
        if self.cfg.use_log_spatial and self._geom_meta.get("log_spatial"):
            log_min = self._geom_meta.get("log_S_min", math.log(self.cfg.S_min))
            log_max = self._geom_meta.get("log_S_max", math.log(self.cfg.S_max))
            log_S = coords_T[:, :, 0, 0] * (log_max - log_min) + log_min
            S_T = torch.exp(log_S)
        else:
            S_T = coords_T[:, :, 0, 0] * (self.cfg.S_max - self.cfg.S_min) + self.cfg.S_min
        pred_T = model.forward_grid(params, coords_T)[:, :, 0]
        payoff = torch.relu(S_T - K.unsqueeze(1))
        loss_terminal = F.mse_loss(pred_T, payoff) * self.cfg.payoff_weight
        # S = 0 boundary (first spatial row)
        coords_0 = grid.clone()
        coords_0[..., 0] = 0.0
        pred_0 = model.forward_grid(params, coords_0)[:, 0, :]
        loss_s0 = F.mse_loss(pred_0, torch.zeros_like(pred_0))

        return loss_terminal + loss_s0

    def american_penalty(
        self,
        pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        S, _ = self._denorm_coords(coords, params)
        K = params[:, 2:3]
        intrinsic = torch.relu(S - K)
        return F.relu(intrinsic - pred).pow(2).mean()

    # ----- physics residuals -----

    def _derivatives(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = coords.detach().requires_grad_(True)
        V = model(params, coords)
        grad = torch.autograd.grad(
            V.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        V_s = grad[..., 0]
        V_t = grad[..., 1]

        grad2 = torch.autograd.grad(
            V_s.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        V_uu = grad2[..., 0]
        return V, V_s, V_t, V_uu, coords

    def black_scholes_residual(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        V, V_u, V_tn, V_uu, coords_g = self._derivatives(model, params, coords)
        r, sigma, _, T, _, _, _, q = params.unbind(dim=1)
        V_t = V_tn / (T.unsqueeze(1) + 1e-8)

        if self.cfg.use_log_spatial and self._geom_meta.get("log_spatial"):
            log_min = self._geom_meta.get("log_S_min", math.log(self.cfg.S_min))
            log_max = self._geom_meta.get("log_S_max", math.log(self.cfg.S_max))
            dx_du = log_max - log_min + 1e-8
            V_x = V_u / dx_du
            V_xx = V_uu / (dx_du**2)
            residual = (
                V_t
                + (r.unsqueeze(1) - q.unsqueeze(1) - 0.5 * sigma.unsqueeze(1) ** 2) * V_x
                + 0.5 * sigma.unsqueeze(1) ** 2 * V_xx
                - r.unsqueeze(1) * V
            )
        else:
            S, _ = self._denorm_coords(coords_g, params)
            dS = self.cfg.S_max - self.cfg.S_min
            V_S = V_u / dS
            V_SS = V_uu / (dS**2)
            residual = (
                V_t
                + 0.5 * sigma.unsqueeze(1) ** 2 * S**2 * V_SS
                + (r.unsqueeze(1) - q.unsqueeze(1)) * S * V_S
                - r.unsqueeze(1) * V
            )
        return (residual**2).mean()

    def fractional_bs_residual(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Caputo D_t^α V + spatial BS operator = 0."""
        V, V_u, V_tn, V_uu, coords_g = self._derivatives(model, params, coords)
        S, t = self._denorm_coords(coords_g, params)
        r, sigma, _, T, alpha, _, _, q = params.unbind(dim=1)

        if self.cfg.use_log_spatial and self._geom_meta.get("log_spatial"):
            dx = self._geom_meta.get("log_S_max", 0) - self._geom_meta.get("log_S_min", 0) + 1e-8
            V_S = V_u / dx
            V_SS = V_uu / (dx**2)
        else:
            dS = self.cfg.S_max - self.cfg.S_min
            V_S = V_u / dS
            V_SS = V_uu / (dS**2)
        spatial = (
            0.5 * sigma.unsqueeze(1) ** 2 * S**2 * V_SS
            + (r.unsqueeze(1) - q.unsqueeze(1)) * S * V_S
            - r.unsqueeze(1) * V
        )

        # Group by parameter sample, apply Caputo along temporal index
        B, N = V.shape
        n_t = model.cfg.n_temporal
        n_s = model.cfg.n_spatial

        if N != n_s * n_t:
            # Scattered collocation: use GL on sorted-by-t slices (approximate)
            dt = (T / n_t).mean().item()
            V_flat = V.unsqueeze(1)
            D_alpha_V = caputo_l1_time_derivative(
                V_flat, dt, alpha.mean().item(), model.cfg.caputo_l1_steps
            ).squeeze(1)
            residual = D_alpha_V + spatial
            return (residual**2).mean()

        V_grid = V.view(B, n_s, n_t)
        dt = (T / (n_t - 1)).view(B, 1, 1)
        D_alpha = model.fractional_time_derivative(V_grid, dt.mean().item())
        D_alpha_flat = D_alpha.reshape(B, N)
        residual = D_alpha_flat + spatial
        return (residual**2).mean()

    def merton_pide_residual(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merton PIDE (European):
            V_t + 0.5 σ² S² V_SS + (r-q) S V_S - r V
            + λ ∫ [V(Se^y) - V(S)] ν(dy) = 0

        ν = lognormal jump density with mean μ_J, vol σ (same as diffusive σ here).
        """
        V, V_u, V_tn, V_uu, coords_g = self._derivatives(model, params, coords)
        S, _ = self._denorm_coords(coords_g, params)
        r, sigma, _, T, _, lam, mu_j, q = params.unbind(dim=1)

        if self.cfg.use_log_spatial and self._geom_meta.get("log_spatial"):
            dx = self._geom_meta.get("log_S_max", 0) - self._geom_meta.get("log_S_min", 0) + 1e-8
            V_S = V_u / dx
            V_SS = V_uu / (dx**2)
        else:
            dS = self.cfg.S_max - self.cfg.S_min
            V_S = V_u / dS
            V_SS = V_uu / (dS**2)
        V_t = V_tn / (T.unsqueeze(1) + 1e-8)

        diffusion = (
            V_t
            + 0.5 * sigma.unsqueeze(1) ** 2 * S**2 * V_SS
            + (r.unsqueeze(1) - q.unsqueeze(1)) * S * V_S
            - r.unsqueeze(1) * V
        )

        # Jump integral quadrature (subsample collocation for tractability)
        n_q = self.cfg.jump_quad_points
        n_sub = min(coords.shape[1], 256)
        idx = torch.randperm(coords.shape[1], device=S.device)[:n_sub]
        S_sub = S[:, idx]
        V_sub = V[:, idx]
        coords_sub = coords_g[:, idx]

        y = torch.linspace(
            -self.cfg.jump_std_mult * sigma.mean().item(),
            self.cfg.jump_std_mult * sigma.mean().item(),
            n_q,
            device=S.device,
        )
        dy = y[1] - y[0]
        S_jump = S_sub.unsqueeze(-1) * torch.exp(y)  # (B, n_sub, n_q)
        s_jump_norm = (S_jump - self.cfg.S_min) / (self.cfg.S_max - self.cfg.S_min)
        t_norm = coords_sub[..., 1].unsqueeze(-1).expand_as(s_jump_norm)
        B, ns, nq = s_jump_norm.shape
        coords_flat = torch.stack(
            [s_jump_norm.reshape(B, ns * nq), t_norm.reshape(B, ns * nq)], dim=-1
        )
        params_flat = params.unsqueeze(1).expand(B, ns * nq, -1).reshape(B * ns * nq, -1)
        coords_flat = coords_flat.reshape(B * ns * nq, 2).unsqueeze(0)
        # Per-batch forward (memory-safe)
        V_jump_list = []
        for b in range(B):
            V_jump_list.append(
                model(params[b : b + 1], coords_flat[b : b + 1].reshape(1, ns * nq, 2))
                .view(ns, nq)
            )
        V_jump = torch.stack(V_jump_list, dim=0)
        integrand = V_jump - V_sub.unsqueeze(-1)
        sig_j = sigma.view(B, 1, 1)
        mu = mu_j.view(B, 1, 1)
        density = torch.exp(-0.5 * ((y - mu) / (sig_j + 1e-8)) ** 2) / (
            math.sqrt(2 * math.pi) * (sig_j + 1e-8)
        )
        jump_term = lam.view(B, 1) * (integrand * density).sum(dim=-1) * dy
        # Pad jump_term back to full N for residual combine — use mean jump on subsample
        residual = diffusion[:, idx] + jump_term
        return (residual**2).mean()

    def physics_loss(
        self,
        model: CF_HPINO,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        pde = self.cfg.pde_type
        if pde == LossPDEType.BLACK_SCHOLES:
            return self.black_scholes_residual(model, params, coords)
        if pde == LossPDEType.FRACTIONAL_BS:
            return self.fractional_bs_residual(model, params, coords)
        if pde == LossPDEType.MERTON:
            return self.merton_pide_residual(model, params, coords)
        raise ValueError(pde)

    # ----- forward -----

    def forward(
        self,
        model: CF_HPINO,
        batch: Dict[str, torch.Tensor],
        return_breakdown: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        params = batch["params"]
        coords = batch["coords"]
        device = params.device

        out = model(params, coords, return_features=True)
        pred = out["prices"]

        losses: Dict[str, torch.Tensor] = {}

        if "prices" in batch and batch["prices"] is not None:
            losses["data"] = self.data_loss(pred, batch["prices"])

        if self.cfg.lambda_physics > 0:
            losses["physics"] = self.physics_loss(model, params, coords)
        else:
            losses["physics"] = torch.tensor(0.0, device=device)

        losses["operator"] = self.operator_loss(out)

        if self.cfg.lambda_boundary > 0:
            losses["boundary"] = self.boundary_loss(model, params, device)
        else:
            losses["boundary"] = torch.tensor(0.0, device=device)

        if model.cfg.option_style.value == "american":
            losses["american"] = self.american_penalty(pred, coords, params)

        if "quote_coords" in batch and "quote_prices" in batch:
            losses["market"] = self.market_quote_loss(pred, batch)

        if self.cfg.adaptive and self.adaptive is not None:
            total, weights = self.adaptive(losses)
        else:
            total = (
                self.cfg.lambda_data * losses.get("data", 0.0)
                + self.cfg.lambda_physics * losses["physics"]
                + self.cfg.lambda_operator * losses["operator"]
                + self.cfg.lambda_boundary * losses["boundary"]
                + (
                    self.cfg.lambda_american * losses.get("american", 0.0)
                    if "american" in losses
                    else 0.0
                )
                + self.cfg.lambda_market * losses.get("market", 0.0)
            )
            weights = {k: getattr(self.cfg, f"lambda_{k}", 1.0) for k in losses}

        if return_breakdown:
            breakdown = {k: v.detach().item() for k, v in losses.items()}
            breakdown["total"] = total.detach().item()
            breakdown.update({f"weight_{k}": w for k, w in weights.items()})
            return total, breakdown
        return total
