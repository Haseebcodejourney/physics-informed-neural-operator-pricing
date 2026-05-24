"""
Parametric sampling for option pricing experiments.

Samples market-style parameter vectors θ = [r, σ, K, T, α, λ_J, μ_J, q]
with ranges suitable for equity index options (SPX-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class ParamRanges:
    """Uniform / log-uniform ranges for synthetic data generation."""

    r: Tuple[float, float] = (0.01, 0.06)
    sigma: Tuple[float, float] = (0.10, 0.45)
    K: Tuple[float, float] = (80.0, 120.0)
    T: Tuple[float, float] = (0.1, 2.0)
    alpha: Tuple[float, float] = (0.55, 0.95)  # fractional order
    lambda_j: Tuple[float, float] = (0.0, 2.0)  # jump intensity (per year)
    mu_j: Tuple[float, float] = (-0.15, 0.05)  # log-jump mean
    q: Tuple[float, float] = (0.0, 0.03)  # dividend yield

    S0: float = 100.0
    S_min_ratio: float = 0.2
    S_max_ratio: float = 3.0


@dataclass
class SamplingConfig:
    n_samples: int = 256
    seed: Optional[int] = 42
    log_uniform_sigma: bool = True
    log_uniform_K: bool = False
    fixed_S0: bool = True


def _uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    return rng.uniform(low, high, size=n)


def _log_uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=n))


def sample_parameters(
    cfg: SamplingConfig,
    ranges: Optional[ParamRanges] = None,
) -> torch.Tensor:
    """
    Draw parameter batch θ.

    Returns:
        (N, 8) tensor [r, sigma, K, T, alpha, lambda_j, mu_j, q]
    """
    ranges = ranges or ParamRanges()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    r = _uniform(rng, *ranges.r, n)
    sigma = (
        _log_uniform(rng, *ranges.sigma, n)
        if cfg.log_uniform_sigma
        else _uniform(rng, *ranges.sigma, n)
    )
    K = (
        _log_uniform(rng, *ranges.K, n)
        if cfg.log_uniform_K
        else _uniform(rng, *ranges.K, n)
    )
    T = _uniform(rng, *ranges.T, n)
    alpha = _uniform(rng, *ranges.alpha, n)
    lam = _uniform(rng, *ranges.lambda_j, n)
    mu_j = _uniform(rng, *ranges.mu_j, n)
    q = _uniform(rng, *ranges.q, n)

    params = np.stack([r, sigma, K, T, alpha, lam, mu_j, q], axis=1)
    return torch.from_numpy(params.astype(np.float32))


def build_coordinate_grid(
    n_spatial: int,
    n_temporal: int,
    params: torch.Tensor,
    ranges: Optional[ParamRanges] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Build (S, t) grid per parameter sample for FNO-style training.

    Args:
        params: (B, 8)
        normalize: map to [0,1]^2 for network input

    Returns:
        grid: (B, n_spatial, n_temporal, 2)
        meta: scaling constants for denormalization in loss
    """
    ranges = ranges or ParamRanges()
    B = params.shape[0]
    S0 = ranges.S0

    S_min = S0 * ranges.S_min_ratio
    S_max = S0 * ranges.S_max_ratio

    S_lin = torch.linspace(S_min, S_max, n_spatial)
    device = params.device
    S_lin = S_lin.to(device)

    grids = []
    for b in range(B):
        T_mat = params[b, 3]
        t_lin = torch.linspace(0.0, T_mat.item(), n_temporal, device=device)
        S_grid, t_grid = torch.meshgrid(S_lin, t_lin, indexing="ij")
        coords = torch.stack([S_grid, t_grid], dim=-1)
        if normalize:
            coords = coords.clone()
            coords[..., 0] = (coords[..., 0] - S_min) / (S_max - S_min)
            coords[..., 1] = coords[..., 1] / (T_mat + 1e-8)
        grids.append(coords)

    grid = torch.stack(grids, dim=0)
    meta = {"S_min": S_min, "S_max": S_max, "S0": S0}
    return grid, meta


def collate_option_batch(batch: list) -> Dict[str, torch.Tensor]:
    """Default collate for OptionPricingDataset."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if batch[0][k] is None:
            out[k] = None
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
