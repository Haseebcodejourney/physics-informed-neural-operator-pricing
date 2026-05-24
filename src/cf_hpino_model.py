"""
CF-HPINO: Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator.

Hybrid architecture combining:
  - Neural operator backbone (FNO or DeepONet) for parametric generalization
  - Physics-informed feature branch for PDE-consistent representations
  - Attention-based fusion of operator and physics pathways
  - DRDD (Dynamic Randomized Domain Decomposition) weight initialization
  - Autodiff-ready outputs for Greeks (Delta, Gamma, Vega)

Designed for European/American options under:
  - Black-Scholes
  - Fractional Black-Scholes (Caputo variable-order)
  - Merton jump-diffusion

DDP note: wrap with DistributedDataParallel; call set_static_graph(True)
after first forward if using gradient checkpointing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class OperatorBackbone(str, Enum):
    FNO = "fno"
    DEEPONET = "deeponet"


class OptionStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class CFHPINOConfig:
    """Model hyperparameters and feature flags."""

    # Spatial / temporal resolution (for FNO grid-based path)
    n_spatial: int = 64
    n_temporal: int = 32

    # Backbone
    backbone: OperatorBackbone = OperatorBackbone.FNO
    hidden_dim: int = 128
    fno_modes: int = 16
    fno_depth: int = 4
    deeponet_branch_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    deeponet_trunk_layers: List[int] = field(default_factory=lambda: [128, 128, 128])

    # Parametric input dimension: [r, sigma, K, T, alpha, lambda_j, mu_j, ...]
    n_params: int = 8
    # Coordinate dim: (S, t) or (log S, tau) with tau = T - t
    n_coords: int = 2

    # Physics branch
    physics_hidden: List[int] = field(default_factory=lambda: [256, 256, 256])
    use_physics_branch: bool = True

    # Fusion
    fusion_heads: int = 4
    fusion_dropout: float = 0.0

    # Fractional derivative (Caputo L1 scheme)
    fractional_order: float = 0.8  # default alpha in (0, 1)
    caputo_l1_steps: int = 20

    # DRDD initialization
    drdd_n_subspaces: int = 8
    drdd_subspace_ratio: float = 0.25
    drdd_rescale: float = 1.0

    # Output
    option_style: OptionStyle = OptionStyle.EUROPEAN
    output_activation: str = "softplus"  # ensures non-negative prices

    # Training helpers
    use_spectral_norm: bool = False
    layer_norm: bool = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _make_mlp(
    dims: List[int],
    activation: nn.Module = nn.GELU(),
    layer_norm: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> mode truncation -> linear mix -> IFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.w(x))


class FNOBackbone(nn.Module):
    """
    Parametric FNO: lifts (params + coordinate grid) to a 2D field, applies
    spectral convolutions, projects to scalar price per (S, t) node.
    """

    def __init__(self, cfg: CFHPINOConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.hidden_dim
        modes = cfg.fno_modes

        # Input channels: param broadcast (n_params) + 2 coord channels
        in_ch = cfg.n_params + cfg.n_coords
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes, modes) for _ in range(cfg.fno_depth)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, 1),
        )

    def forward(
        self,
        params: torch.Tensor,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            params: (B, P) parameter vectors
            grid:   (B, N, 2) or (B, H, W, 2) normalized (S, t) coordinates

        Returns:
            field:  (B, H, W) operator branch prediction on grid
            latent: (B, C, H, W) last hidden feature map for fusion
        """
        if grid.dim() == 3:
            b, n, _ = grid.shape
            h = self.cfg.n_spatial
            w = self.cfg.n_temporal
            # Reshape scattered points to regular grid if N = H*W
            if n == h * w:
                grid = grid.view(b, h, w, 2)
            else:
                raise ValueError(
                    f"FNO expects grid with H*W={h*w} points or (B,H,W,2); got N={n}"
                )

        b, h, w, _ = grid.shape
        p = params.unsqueeze(-1).unsqueeze(-1).expand(b, self.cfg.n_params, h, w)
        g = grid.permute(0, 3, 1, 2)  # (B, 2, H, W)
        x = torch.cat([p, g], dim=1)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        latent = x
        field = self.proj(x).squeeze(1)  # (B, H, W)
        return field, latent


class DeepONetBackbone(nn.Module):
    """Standard DeepONet: branch(params) ⊗ trunk(coords)."""

    def __init__(self, cfg: CFHPINOConfig):
        super().__init__()
        self.cfg = cfg
        branch_dims = [cfg.n_params] + cfg.deeponet_branch_layers
        trunk_dims = [cfg.n_coords] + cfg.deeponet_trunk_layers
        self.branch = _make_mlp(branch_dims, layer_norm=cfg.layer_norm)
        self.trunk = _make_mlp(trunk_dims, layer_norm=cfg.layer_norm)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            params: (B, P)
            coords: (B, N, 2)

        Returns:
            values: (B, N) DeepONet output per coordinate
            latent: (B, N, D) concatenated branch/trunk features for fusion
        """
        b_coef = self.branch(params)  # (B, D_b)
        t_feat = self.trunk(coords)  # (B, N, D_t)
        d_b = b_coef.shape[-1]
        d_t = t_feat.shape[-1]
        if d_b != d_t:
            raise ValueError(f"Branch dim {d_b} must match trunk dim {d_t}")
        values = torch.einsum("bd,bnd->bn", b_coef, t_feat) + self.bias
        latent = torch.stack(
            [b_coef.unsqueeze(1).expand(-1, coords.shape[1], -1), t_feat], dim=-1
        ).mean(dim=-1)
        return values, latent


class PhysicsFeatureBranch(nn.Module):
    """
    Encodes (S, t, θ) into physics-consistent features used by fusion and
    optional residual correction. Does not replace PDE loss (see losses.py).
    """

    def __init__(self, cfg: CFHPINOConfig):
        super().__init__()
        in_dim = cfg.n_coords + cfg.n_params
        dims = [in_dim] + cfg.physics_hidden
        self.net = _make_mlp(dims, layer_norm=cfg.layer_norm)

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            params: (B, P)
            coords: (B, N, 2)

        Returns:
            (B, N, H_phys)
        """
        p = params.unsqueeze(1).expand(-1, coords.shape[1], -1)
        x = torch.cat([coords, p], dim=-1)
        return self.net(x.reshape(-1, x.shape[-1])).view(
            coords.shape[0], coords.shape[1], -1
        )


class HybridFusionModule(nn.Module):
    """
    Cross-attention fusion: operator queries, physics keys/values.

    Uses explicit softmax attention (not nn.MultiheadAttention) so physics
    loss can take second derivatives w.r.t. (S, t) on CPU/GPU without flash-SDP
    backward limitations.
    """

    def __init__(self, op_dim: int, phys_dim: int, out_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(op_dim, out_dim)
        self.k_proj = nn.Linear(phys_dim, out_dim)
        self.v_proj = nn.Linear(phys_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        op_feat: torch.Tensor,
        phys_feat: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(op_feat)
        k = self.k_proj(phys_feat)
        v = self.v_proj(phys_feat)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_out = torch.matmul(weights, v)
        return self.norm(q + self.out_proj(attn_out))


class CaputoL1FractionalDerivative(nn.Module):
    """
    L1 approximation of Caputo fractional derivative D^α u(t) along the
    temporal axis of a price field. Used inside physics loss assembly.

    For α ∈ (0, 1), on uniform grid t_j with spacing Δt:
        D^α u(t_n) ≈ (Δt^{-α} / Γ(2-α)) * Σ_{k=0}^{n-1} b_k * (u_{n-k} - u_{n-k-1})
    with L1 weights b_k = (k+1)^{1-α} - k^{1-α}.

    Reference: Li & Zeng (2015); used in fractional Black-Scholes PINO literature.
    """

    def __init__(self, alpha: float, n_steps: int = 20):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Caputo L1 expects alpha in (0,1); got {alpha}")
        self.alpha = alpha
        self.n_steps = n_steps
        self.register_buffer("_weights", self._build_l1_weights(alpha, n_steps))

    @staticmethod
    def _build_l1_weights(alpha: float, n_steps: int) -> torch.Tensor:
        k = torch.arange(n_steps, dtype=torch.float64)
        b = (k + 1).pow(1 - alpha) - k.pow(1 - alpha)
        return b.flip(0)  # convolutional form: newest first

    def forward(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Args:
            u:  (..., T) values along last dimension (time)
            dt: scalar grid spacing

        Returns:
            D_alpha u with same shape as u (truncated at boundaries)
        """
        *prefix, t_len = u.shape
        w = self._weights.to(device=u.device, dtype=u.dtype)
        n = min(self.n_steps, t_len)
        w = w[-n:]

        # Causal convolution along time
        u_pad = F.pad(u, (n, 0), mode="constant", value=0.0)
        diff = u_pad[..., 1:] - u_pad[..., :-1]
        kernel = w.view(1, 1, -1)
        diff_flat = diff.reshape(-1, 1, diff.shape[-1])
        conv = F.conv1d(diff_flat, kernel, padding=0)
        scale = dt ** (-self.alpha) / math.gamma(2.0 - self.alpha)
        out = scale * conv.reshape(*prefix, -1)
        # Align length with input
        pad_len = t_len - out.shape[-1]
        if pad_len > 0:
            out = F.pad(out, (pad_len, 0), value=0.0)
        return out[..., :t_len]


class DRDDInitializer:
    """
    Dynamic Randomized Domain Decomposition (DRDD) weight init.

    Partitions parameter subspaces, applies Xavier/Kaiming on each subset,
    optionally rescales for multi-swarm diversity (ABC can replace sampling).
    """

    def __init__(
        self,
        n_subspaces: int = 8,
        subspace_ratio: float = 0.25,
        rescale: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_subspaces = n_subspaces
        self.subspace_ratio = subspace_ratio
        self.rescale = rescale
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def initialize_module(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if not param.requires_grad or param.dim() < 2:
                if param.dim() == 1:
                    nn.init.zeros_(param)
                continue

            out_f, in_f = param.shape[0], param.shape[1]
            n_active = max(1, int(in_f * self.subspace_ratio))

            for s in range(self.n_subspaces):
                cols = torch.randperm(in_f, generator=self.rng)[:n_active]
                sub_w = param.data[:, cols]
                fan_in, fan_out = sub_w.shape[1], sub_w.shape[0]
                std = self.rescale * math.sqrt(2.0 / (fan_in + fan_out))
                sub_w.normal_(0.0, std, generator=self.rng)
                param.data[:, cols] = sub_w

            if param.dim() > 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CF_HPINO(nn.Module):
    """
    Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator.

    Forward modes
    -------------
    1) Pointwise pricing: forward(params, coords) -> V(S, t; θ)
    2) Grid pricing (FNO): forward(params, grid) with grid (B, H, W, 2)
    3) Greeks: autograd on forward with coords requiring grad

    Physics residuals (Black-Scholes, fractional BS, Merton) are computed in
    cf_hpino_loss.py using derivatives of the scalar output w.r.t. (S, t).
    This module exposes `fractional_time_derivative` for fractional BS.
    """

    def __init__(self, cfg: Optional[CFHPINOConfig] = None):
        super().__init__()
        self.cfg = cfg or CFHPINOConfig()

        if self.cfg.backbone == OperatorBackbone.FNO:
            self.operator = FNOBackbone(self.cfg)
            op_feat_dim = self.cfg.hidden_dim
        else:
            self.operator = DeepONetBackbone(self.cfg)
            op_feat_dim = self.cfg.deeponet_branch_layers[-1]

        self.physics_branch = (
            PhysicsFeatureBranch(self.cfg) if self.cfg.use_physics_branch else None
        )
        phys_dim = self.cfg.physics_hidden[-1] if self.cfg.use_physics_branch else op_feat_dim

        self.fusion = HybridFusionModule(
            op_dim=op_feat_dim,
            phys_dim=phys_dim,
            out_dim=self.cfg.hidden_dim,
            n_heads=self.cfg.fusion_heads,
            dropout=self.cfg.fusion_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim // 2, 1),
        )

        self.fractional_derivative = CaputoL1FractionalDerivative(
            alpha=self.cfg.fractional_order,
            n_steps=self.cfg.caputo_l1_steps,
        )

        self._init_output_activation()

    def _init_output_activation(self) -> None:
        act = self.cfg.output_activation.lower()
        if act == "softplus":
            self.output_act: Optional[nn.Module] = nn.Softplus()
        elif act == "relu":
            self.output_act = nn.ReLU()
        elif act == "none":
            self.output_act = None
        else:
            raise ValueError(f"Unknown output_activation: {act}")

    def apply_drdd_init(self, seed: Optional[int] = None) -> None:
        """Apply DRDD initialization to all learnable layers."""
        init = DRDDInitializer(
            n_subspaces=self.cfg.drdd_n_subspaces,
            subspace_ratio=self.cfg.drdd_subspace_ratio,
            rescale=self.cfg.drdd_rescale,
            seed=seed,
        )
        init.initialize_module(self)
        # Bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ----- feature extraction -----

    def _operator_forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.backbone == OperatorBackbone.FNO:
            values, latent = self.operator(params, coords)
            # Flatten grid latent to (B, N, C)
            b, c, h, w = latent.shape
            op_feat = latent.permute(0, 2, 3, 1).reshape(b, h * w, c)
            values_flat = values.reshape(b, h * w)
            return values_flat, op_feat

        values, latent = self.operator(params, coords)
        return values, latent.unsqueeze(-1) if latent.dim() == 2 else latent

    def _apply_head(self, fused: torch.Tensor) -> torch.Tensor:
        v = self.head(fused).squeeze(-1)
        if self.output_act is not None:
            v = self.output_act(v)
        return v

    # ----- public API -----

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict option prices at coordinate points.

        Args:
            params: (B, P) — [r, sigma, K, T, alpha, lambda_j, mu_j, q, ...]
            coords: (B, N, 2) — normalized (S, t) or (log S, tau)
            return_features: if True, return dict with intermediates

        Returns:
            prices: (B, N) or dict
        """
        op_values, op_feat = self._operator_forward(params, coords)

        if self.physics_branch is not None:
            phys_feat = self.physics_branch(params, coords)
        else:
            phys_feat = op_feat

        fused = self.fusion(op_feat, phys_feat)
        correction = self._apply_head(fused)

        # Residual connection to operator branch (hybrid fusion)
        prices = op_values + correction

        if self.cfg.option_style == OptionStyle.AMERICAN:
            # LSTM-free early exercise proxy: enforce V >= intrinsic at query points
            # Full American constraint is enforced in loss (cf_hpino_loss.py)
            s = coords[..., 0]
            k = params[:, 2:3]
            intrinsic = torch.relu(s - k) if s.shape == k.shape else torch.relu(
                coords[..., 0:1] - params[:, 2:3]
            ).squeeze(-1)
            prices = torch.maximum(prices, intrinsic.squeeze(-1))

        if return_features:
            return {
                "prices": prices,
                "operator_values": op_values,
                "operator_features": op_feat,
                "physics_features": phys_feat,
                "fused_features": fused,
                "correction": correction,
            }
        return prices

    def forward_grid(
        self,
        params: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper when coords are structured as (B, H, W, 2)."""
        b, h, w, _ = grid.shape
        coords = grid.reshape(b, h * w, 2)
        return self.forward(params, coords).view(b, h, w)

    # ----- Greeks (autodiff) -----

    def greeks(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        compute: Tuple[str, ...] = ("delta", "gamma", "vega"),
    ) -> Dict[str, torch.Tensor]:
        """
        Automatic differentiation for sensitivities.

        Assumes coords[..., 0] = S (or log S) and params[..., 1] = sigma for Vega.
        """
        coords = coords.detach().requires_grad_(True)
        sigma_idx = 1

        prices = self.forward(params, coords)

        out: Dict[str, torch.Tensor] = {"price": prices}

        if "delta" in compute or "gamma" in compute:
            grad_s = torch.autograd.grad(
                prices.sum(),
                coords,
                create_graph=True,
                retain_graph=True,
            )[0][..., 0]

        if "delta" in compute:
            out["delta"] = grad_s

        if "gamma" in compute:
            grad2 = torch.autograd.grad(
                grad_s.sum(),
                coords,
                retain_graph=True,
                create_graph=True,
            )[0][..., 0]
            out["gamma"] = grad2

        if "vega" in compute:
            params_v = params.detach().clone()
            params_v.requires_grad_(True)
            p_vega = self.forward(params_v, coords)
            out["vega"] = torch.autograd.grad(
                p_vega.sum(),
                params_v,
                retain_graph=True,
            )[0][:, sigma_idx]

        return out

    # ----- fractional helper for loss module -----

    def fractional_time_derivative(
        self,
        price_field: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Apply Caputo L1 derivative along temporal dimension of a price surface.

        Args:
            price_field: (B, H, W) or (B, T) — time on last axis after reshape
            dt: uniform time step
        """
        if price_field.dim() == 3:
            # Apply along W (time) for each spatial slice
            b, h, w = price_field.shape
            flat = price_field.reshape(b * h, w)
            d = self.fractional_derivative(flat, dt)
            return d.view(b, h, w)
        return self.fractional_derivative(price_field, dt)

    # ----- calibration / inverse hook -----

    def calibrate_parameters(
        self,
        market_prices: torch.Tensor,
        coords: torch.Tensor,
        param_mask: torch.Tensor,
        initial_params: torch.Tensor,
        n_steps: int = 200,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        Simple inverse problem: optimize selected entries of θ to match market.

        Args:
            market_prices: (B, N) observed quotes
            coords:        (B, N, 2)
            param_mask:    (P,) bool — which parameters are learnable
            initial_params:(B, P)

        Returns:
            optimized params (B, P)
        """
        theta = initial_params.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            pred = self.forward(theta, coords)
            loss = F.mse_loss(pred, market_prices)
            loss.backward()
            with torch.no_grad():
                if theta.grad is not None:
                    theta.grad[:, ~param_mask] = 0.0
            opt.step()
        return theta.detach()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_cf_hpino(
    backbone: str = "fno",
    fractional_order: float = 0.8,
    option_style: str = "european",
    **kwargs,
) -> CF_HPINO:
    """Convenience factory for experiments and DDP workers."""
    cfg = CFHPINOConfig(
        backbone=OperatorBackbone(backbone.lower()),
        fractional_order=fractional_order,
        option_style=OptionStyle(option_style.lower()),
        **{k: v for k, v in kwargs.items() if hasattr(CFHPINOConfig, k)},
    )
    model = CF_HPINO(cfg)
    model.apply_drdd_init()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
