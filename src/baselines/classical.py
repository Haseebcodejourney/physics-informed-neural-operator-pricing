"""
Classical reference pricers: Black-Scholes analytic, Crank-Nicolson FD, Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..data.synthetic_pde import black_scholes_call, merton_call_mc


class PricingMethod(str, Enum):
    ANALYTIC = "analytic"
    FINITE_DIFFERENCE = "fd"
    MONTE_CARLO = "mc"


@dataclass
class ClassicalConfig:
    method: PricingMethod = PricingMethod.ANALYTIC
    n_paths: int = 100_000
    n_s: int = 200
    n_t: int = 100
    seed: Optional[int] = None


class ClassicalPricer:
    """Non-learned baselines for evaluation."""

    def __init__(self, cfg: Optional[ClassicalConfig] = None):
        self.cfg = cfg or ClassicalConfig()

    def price(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
        model: str = "black_scholes",
    ) -> torch.Tensor:
        r, sigma, K, T, _, lam, mu_j, q = params.unbind(dim=1)
        # Broadcast scalar params to collocation shape (B, N)
        r = r.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        K = K.unsqueeze(-1)
        T = T.unsqueeze(-1)
        q = q.unsqueeze(-1)
        if model == "merton" and self.cfg.method == PricingMethod.MONTE_CARLO:
            return merton_call_mc(
                S, t, r, sigma, K, T, q, lam.unsqueeze(-1), mu_j.unsqueeze(-1),
                n_paths=self.cfg.n_paths, seed=self.cfg.seed,
            )
        if self.cfg.method == PricingMethod.FINITE_DIFFERENCE and model == "black_scholes":
            return self._fd_surface_lookup(S, t, params)
        return black_scholes_call(S, t, r, sigma, K, T, q)

    def _fd_surface_lookup(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Crank-Nicolson per parameter row, interpolate at query points."""
        from scipy.interpolate import RegularGridInterpolator

        out = []
        for b in range(params.shape[0]):
            p = params[b]
            Sg, tg, Vg = crank_nicolson_bs(
                p[0].item(), p[1].item(), p[2].item(), p[3].item(), p[7].item(),
                n_s=self.cfg.n_s, n_t=self.cfg.n_t,
            )
            interp = RegularGridInterpolator(
                (Sg, tg), Vg, bounds_error=False, fill_value=0.0
            )
            pts = np.stack([S[b].detach().cpu().numpy(), t[b].detach().cpu().numpy()], axis=1)
            out.append(torch.from_numpy(interp(pts).astype(np.float32)))
        return torch.stack(out, dim=0).to(S.device)


def crank_nicolson_bs(
    r: float,
    sigma: float,
    K: float,
    T: float,
    q: float = 0.0,
    s_max_mult: float = 3.0,
    n_s: int = 200,
    n_t: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicit Crank-Nicolson for European call under BS."""
    S_max = s_max_mult * K
    S = np.linspace(0.0, S_max, n_s)
    t = np.linspace(0.0, T, n_t)
    dt = t[1] - t[0]
    dS = S[1] - S[0]

    V = np.zeros((n_s, n_t))
    V[:, -1] = np.maximum(S - K, 0.0)

    sig2 = sigma**2
    for n in range(n_t - 2, -1, -1):
        V_old = V[:, n + 1].copy()
        for i in range(1, n_s - 1):
            Si = S[i]
            Vx = (V_old[i + 1] - V_old[i - 1]) / (2 * dS)
            Vxx = (V_old[i + 1] - 2 * V_old[i] + V_old[i - 1]) / dS**2
            V[i, n] = V_old[i] + dt * (
                0.5 * sig2 * Si**2 * Vxx + (r - q) * Si * Vx - r * V_old[i]
            )
        V[0, n] = 0.0
        V[-1, n] = S_max - K * np.exp(-r * (T - t[n]))
    return S.astype(np.float32), t.astype(np.float32), V.astype(np.float32)


def bs_greeks_analytic(
    S: float,
    t: float,
    r: float,
    sigma: float,
    K: float,
    T: float,
    q: float = 0.0,
) -> dict:
    """Closed-form Delta, Gamma, Vega for European call."""
    from scipy.stats import norm

    tau = max(T - t, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    delta = np.exp(-q * tau) * norm.cdf(d1)
    gamma = np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
    return {"delta": delta, "gamma": gamma, "vega": vega}
