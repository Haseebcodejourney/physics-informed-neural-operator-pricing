"""
Synthetic option price surfaces and reference solvers.

- Black-Scholes: closed-form European call
- Merton jump-diffusion: Monte Carlo + control variate (BS)
- Fractional Black-Scholes: finite-difference with Grünwald-Letnikov Caputo operator
- American (BS): Brennan-Schwartz PSOR on log-price grid
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampling import ParamRanges, SamplingConfig, build_coordinate_grid, sample_parameters


class PricingModel(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    FRACTIONAL_BS = "fractional_bs"
    MERTON = "merton"


# ---------------------------------------------------------------------------
# Analytic / numerical pricers
# ---------------------------------------------------------------------------


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def black_scholes_call(
    S: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    European call under Black-Scholes with continuous dividend yield q.

    All inputs broadcastable; t is current time in [0, T], tau = T - t.
    """
    tau = torch.clamp(T - t, min=1e-8)
    sig_sqrt = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sig_sqrt + 1e-10)
    d2 = d1 - sig_sqrt
    return S * torch.exp(-q * tau) * _norm_cdf(d1) - K * torch.exp(-r * tau) * _norm_cdf(d2)


def merton_call_mc(
    S: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    q: torch.Tensor,
    lam: torch.Tensor,
    mu_j: torch.Tensor,
    n_paths: int = 50_000,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    European call under Merton (log-normal jumps, compound Poisson).

    Uses log-asset-price recursion with exact number of jumps per path.
    """
    device = S.device
    dtype = S.dtype
    tau = torch.clamp(T - t, min=1e-8)

    # Merton compensator
    k_m = torch.exp(mu_j + 0.5 * sigma**2) - 1.0
    r_adj = r - q - lam * k_m

    rng = np.random.default_rng(seed)
    # Flatten for MC then reshape
    shape = S.shape
    n_elem = S.numel()
    S_flat = S.reshape(n_elem).cpu().numpy()
    tau_flat = tau.reshape(n_elem).cpu().numpy()
    r_flat = r_adj.reshape(n_elem).cpu().numpy()
    sig_flat = sigma.reshape(n_elem).cpu().numpy()
    K_flat = K.reshape(n_elem).cpu().numpy()
    lam_flat = lam.reshape(n_elem).cpu().numpy()
    mu_flat = mu_j.reshape(n_elem).cpu().numpy()

    prices = np.zeros(n_elem, dtype=np.float64)
    half = n_paths // 2

    for i in range(n_elem):
        Z = rng.standard_normal((half,))
        Z = np.concatenate([Z, -Z])  # antithetic
        N_j = rng.poisson(lam_flat[i] * tau_flat[i], size=Z.shape[0])
        jump_sum = np.zeros_like(Z)
        for p in range(Z.shape[0]):
            if N_j[p] > 0:
                jump_sum[p] = rng.normal(mu_flat[i], sig_flat[i], size=N_j[p]).sum()
        log_ST = (
            np.log(S_flat[i])
            + r_flat[i] * tau_flat[i]
            + sig_flat[i] * np.sqrt(tau_flat[i]) * Z
            + jump_sum
        )
        payoff = np.maximum(np.exp(log_ST) - K_flat[i], 0.0)
        prices[i] = np.exp(-r_flat[i] * tau_flat[i]) * payoff.mean()

    return torch.from_numpy(prices.astype(np.float32)).reshape(shape).to(device)


def fractional_bs_fdm(
    r: float,
    sigma: float,
    K: float,
    T: float,
    alpha: float,
    q: float = 0.0,
    n_s: int = 80,
    n_t: int = 60,
    s_max_mult: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fractional Black-Scholes (European call) via implicit-explicit FDM.

    PDE (risk-neutral, Caputo in time):
        D_t^α V + 0.5 σ² S² V_SS + (r-q) S V_S - r V = 0,  0 < α < 1

    Discretization: L1 Caputo weights on uniform time grid, central space.
    Boundary: V(0,t)=0, V(S_max,t) ~ S_max - K*exp(-r(T-t)).

    Returns:
        S_grid (n_s,), t_grid (n_t,), V (n_s, n_t)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    S_max = s_max_mult * K
    S = np.linspace(0.0, S_max, n_s)
    t = np.linspace(0.0, T, n_t)
    dt = t[1] - t[0]
    dS = S[1] - S[0] if n_s > 1 else 1.0

    V = np.zeros((n_s, n_t), dtype=np.float64)
    # Terminal payoff
    V[:, -1] = np.maximum(S - K, 0.0)

    # L1 Caputo weights b_k = (k+1)^{1-α} - k^{1-α}
    n_cap = min(n_t - 1, 30)
    k_idx = np.arange(n_cap)
    b = (k_idx + 1) ** (1 - alpha) - k_idx ** (1 - alpha)
    scale = (dt ** (-alpha)) / float(np.math.gamma(2.0 - alpha))

    def spatial_operator(V_slice: np.ndarray) -> np.ndarray:
        L = np.zeros_like(V_slice)
        for j in range(1, n_s - 1):
            Sj = S[j]
            V_ss = (V_slice[j + 1] - 2 * V_slice[j] + V_slice[j - 1]) / dS**2
            V_s = (V_slice[j + 1] - V_slice[j - 1]) / (2 * dS)
            L[j] = 0.5 * sigma**2 * Sj**2 * V_ss + (r - q) * Sj * V_s - r * V_slice[j]
        return L

    # Backward march in physical time index (from maturity to t=0)
    for n in range(n_t - 2, -1, -1):
        V_next = V[:, n + 1].copy()
        Lv = spatial_operator(V_next)

        # Caputo history sum on past time levels
        caputo = np.zeros(n_s)
        for m in range(1, min(n_cap, n_t - 1 - n) + 1):
            idx_new = n + m
            idx_old = n + m - 1
            if idx_new < n_t and idx_old < n_t:
                caputo += b[m - 1] * (V[:, idx_new] - V[:, idx_old])

        # D^α V ≈ scale * caputo ≈ -Lv  =>  V^n ≈ V^{n+1} - dt^α * Lv / scale (explicit step)
        V[:, n] = V_next - (dt**alpha) * Lv / (scale + 1e-12)
        V[0, n] = 0.0
        V[-1, n] = S_max - K * np.exp(-r * (T - t[n]))

    return S, t, V.astype(np.float32)


def american_bs_psor(
    r: float,
    sigma: float,
    K: float,
    T: float,
    q: float = 0.0,
    n_x: int = 120,
    n_t: int = 80,
    omega: float = 1.2,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brennan-Schwartz PSOR for American call on log-price grid (x = log S)."""
    x_max = np.log(5.0 * K)
    x_min = np.log(1e-4 * K)
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(0.0, T, n_t)
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    S = np.exp(x)
    V = np.zeros((n_x, n_t), dtype=np.float64)
    V[:, -1] = np.maximum(S - K, 0.0)

    sig2 = sigma**2
    for n in range(n_t - 2, -1, -1):
        V_old = V[:, n + 1].copy()
        V_new = V_old.copy()
        for _ in range(200):
            V_prev = V_new.copy()
            for i in range(1, n_x - 1):
                Vx = (V_new[i + 1] - V_new[i - 1]) / (2 * dx)
                Vxx = (V_new[i + 1] - 2 * V_new[i] + V_new[i - 1]) / dx**2
                Si = S[i]
                rhs = V_old[i] + dt * (
                    0.5 * sig2 * Si**2 * Vxx + (r - q) * Si * Vx - r * V_new[i]
                )
                intrinsic = max(Si - K, 0.0)
                V_new[i] = max((1 - omega) * V_new[i] + omega * rhs, intrinsic)
            if np.max(np.abs(V_new - V_prev)) < tol:
                break
        V[:, n] = V_new

    return S, t, V.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    model: PricingModel = PricingModel.BLACK_SCHOLES
    n_param_samples: int = 512
    n_spatial: int = 64
    n_temporal: int = 32
    normalize_coords: bool = True
    option_style: str = "european"
    merton_paths: int = 20_000
    seed: int = 42
    param_ranges: Optional[ParamRanges] = None


class OptionPricingDataset(Dataset):
    """
    PyTorch dataset yielding (params, coords, prices, grid) tuples.

    Each __getitem__ is one parameter draw with full (S,t) lattice.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.ranges = cfg.param_ranges or ParamRanges()
        self.params = sample_parameters(
            SamplingConfig(n_samples=cfg.n_param_samples, seed=cfg.seed),
            self.ranges,
        )
        self.grid, self.meta = build_coordinate_grid(
            cfg.n_spatial,
            cfg.n_temporal,
            self.params,
            self.ranges,
            normalize=cfg.normalize_coords,
        )
        self.prices = self._generate_all_prices()

    def _price_single(self, idx: int) -> torch.Tensor:
        p = self.params[idx]
        r, sigma, K, T, alpha, lam, mu_j, q = p
        grid = self.grid[idx]
        S_phys = grid[..., 0] * (self.meta["S_max"] - self.meta["S_min"]) + self.meta["S_min"]
        t_phys = grid[..., 1] * T

        if self.cfg.model == PricingModel.BLACK_SCHOLES:
            if self.cfg.option_style == "american":
                S_np, t_np, V_np = american_bs_psor(
                    r.item(), sigma.item(), K.item(), T.item(), q.item(),
                    n_s=self.cfg.n_spatial, n_t=self.cfg.n_temporal,
                )
                # Interpolate PSOR grid onto our lattice
                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator((S_np, t_np), V_np, bounds_error=False, fill_value=0.0)
                pts = np.stack([S_phys.numpy().ravel(), t_phys.numpy().ravel()], axis=1)
                v = interp(pts).reshape(self.cfg.n_spatial, self.cfg.n_temporal)
                return torch.from_numpy(v.astype(np.float32))
            return black_scholes_call(
                S_phys, t_phys, r, sigma, K, T, q
            )

        if self.cfg.model == PricingModel.MERTON:
            return merton_call_mc(
                S_phys, t_phys, r, sigma, K, T, q, lam, mu_j,
                n_paths=self.cfg.merton_paths,
                seed=self.cfg.seed + idx,
            )

        if self.cfg.model == PricingModel.FRACTIONAL_BS:
            S_np, t_np, V_np = fractional_bs_fdm(
                r.item(), sigma.item(), K.item(), T.item(), alpha.item(), q.item(),
                n_s=self.cfg.n_spatial, n_t=self.cfg.n_temporal,
            )
            return torch.from_numpy(V_np)

        raise ValueError(f"Unknown model {self.cfg.model}")

    def _generate_all_prices(self) -> torch.Tensor:
        surfaces = []
        for i in range(len(self.params)):
            surfaces.append(self._price_single(i))
        return torch.stack(surfaces, dim=0)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        coords = self.grid[idx].reshape(-1, 2)
        prices = self.prices[idx].reshape(-1)
        return {
            "params": self.params[idx],
            "coords": coords,
            "prices": prices,
            "grid": self.grid[idx],
            "price_surface": self.prices[idx],
        }
