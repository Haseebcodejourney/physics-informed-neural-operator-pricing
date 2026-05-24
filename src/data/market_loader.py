"""
Market option chain loader (SPX / CSV).

Expected CSV columns (flexible aliases):
  strike, expiry or maturity, mid or close, spot, rate, div_yield, vol (optional)

If vol missing, implied vol is inverted via Newton on Black-Scholes.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampling import ParamRanges, build_coordinate_grid
from .synthetic_pde import _norm_cdf


_COLUMN_ALIASES = {
    "strike": ("strike", "k", "K"),
    "expiry": ("expiry", "maturity", "T", "days_to_expiry", "dte"),
    "price": ("mid", "close", "last", "price", "premium"),
    "spot": ("spot", "S0", "underlying", "spx"),
    "rate": ("rate", "r", "risk_free"),
    "div": ("div_yield", "q", "dividend"),
    "vol": ("iv", "implied_vol", "sigma", "vol"),
}


@dataclass
class MarketLoaderConfig:
    csv_path: str
    n_spatial: int = 32
    n_temporal: int = 16
    normalize_coords: bool = True
    option_type: str = "call"
    day_count: float = 365.0
    param_ranges: Optional[ParamRanges] = None


def _resolve_column(header: List[str], key: str) -> Optional[str]:
    lower = {h.lower(): h for h in header}
    for alias in _COLUMN_ALIASES[key]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _parse_expiry(val: str, day_count: float) -> float:
    """Return maturity in years."""
    try:
        x = float(val)
        if x > 3.0:  # likely days
            return x / day_count
        return x
    except ValueError:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                d = datetime.strptime(val.strip(), fmt)
                days = (d - datetime.today()).days
                return max(days / day_count, 1 / day_count)
            except ValueError:
                continue
    return 0.25


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    max_iter: int = 50,
) -> float:
    """Invert BS call price for sigma."""
    sigma = 0.25
    for _ in range(max_iter):
        St = torch.tensor(S)
        Kt = torch.tensor(K)
        Tt = torch.tensor(T)
        rt = torch.tensor(r)
        qt = torch.tensor(q)
        sig = torch.tensor(sigma)
        tau = T
        d1 = (torch.log(St / Kt) + (rt - qt + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau) + 1e-10)
        price = (
            St * torch.exp(-qt * tau) * _norm_cdf(d1)
            - Kt * torch.exp(-rt * tau) * _norm_cdf(d1 - sig * np.sqrt(tau))
        ).item()
        vega = (
            St.item()
            * np.exp(-q * tau)
            * float(np.exp(-0.5 * d1.item() ** 2) / np.sqrt(2 * np.pi))
            * np.sqrt(tau)
        )
        diff = price - market_price
        if abs(diff) < 1e-6:
            break
        sigma = max(sigma - diff / (vega + 1e-8), 1e-4)
    return sigma


class MarketOptionDataset(Dataset):
    """
    Each row in CSV becomes one training item with interpolated lattice.

    params: [r, sigma, K, T, alpha, lambda_j, mu_j, q]
    alpha/lambda/mu set to 0.8 / 0 / 0 for pure BS market fit unless provided.
    """

    def __init__(self, cfg: MarketLoaderConfig):
        self.cfg = cfg
        self.ranges = cfg.param_ranges or ParamRanges()
        self.rows = self._load_csv(Path(cfg.csv_path))
        self._build_tensors()

    def _load_csv(self, path: Path) -> List[Dict[str, float]]:
        if not path.exists():
            raise FileNotFoundError(f"Market CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            cols = {k: _resolve_column(header, k) for k in _COLUMN_ALIASES}
            if cols["strike"] is None or cols["price"] is None:
                raise ValueError(f"CSV must include strike and price columns. Got: {header}")

            rows = []
            for row in reader:
                strike = float(row[cols["strike"]])
                price = float(row[cols["price"]])
                spot = float(row[cols["spot"]]) if cols["spot"] else self.ranges.S0
                rate = float(row[cols["rate"]]) if cols["rate"] else 0.03
                div = float(row[cols["div"]]) if cols["div"] else 0.0
                T = _parse_expiry(row[cols["expiry"]], self.cfg.day_count) if cols["expiry"] else 0.5
                if cols["vol"] and row.get(cols["vol"]):
                    vol = float(row[cols["vol"]])
                else:
                    vol = implied_vol_newton(price, spot, strike, T, rate, div)
                rows.append(
                    {
                        "strike": strike,
                        "price": price,
                        "spot": spot,
                        "rate": rate,
                        "div": div,
                        "T": T,
                        "vol": vol,
                    }
                )
        return rows

    def _build_tensors(self) -> None:
        n = len(self.rows)
        params = []
        grids = []
        prices_list = []

        for row in self.rows:
            p = torch.tensor(
                [
                    row["rate"],
                    row["vol"],
                    row["strike"],
                    row["T"],
                    0.8,
                    0.0,
                    0.0,
                    row["div"],
                ],
                dtype=torch.float32,
            )
            params.append(p)
            pg = p.unsqueeze(0)
            grid, meta = build_coordinate_grid(
                self.cfg.n_spatial,
                self.cfg.n_temporal,
                pg,
                self.ranges,
                normalize=self.cfg.normalize_coords,
            )
            grids.append(grid.squeeze(0))
            # Target surface from market quote (flat IV slice — extend with smile later)
            S_phys = grid[..., 0] * (meta["S_max"] - meta["S_min"]) + meta["S_min"]
            t_phys = grid[..., 1] * row["T"]
            from .synthetic_pde import black_scholes_call

            surf = black_scholes_call(
                S_phys,
                t_phys,
                p[0],
                p[1],
                p[2],
                p[3],
                p[7],
            )
            prices_list.append(surf)

        self.params = torch.stack(params, dim=0)
        self.grids = torch.stack(grids, dim=0)
        self.surfaces = torch.stack(prices_list, dim=0)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "params": self.params[idx],
            "coords": self.grids[idx].reshape(-1, 2),
            "prices": self.surfaces[idx].reshape(-1),
            "grid": self.grids[idx],
            "price_surface": self.surfaces[idx],
            "market_mid": torch.tensor(self.rows[idx]["price"]),
        }


def generate_demo_spx_csv(path: Path, n_quotes: int = 50, seed: int = 0) -> None:
    """Write synthetic SPX-style quotes for testing without external data."""
    rng = np.random.default_rng(seed)
    S0 = 4500.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strike", "expiry", "mid", "spot", "rate", "div_yield"])
        for _ in range(n_quotes):
            K = rng.uniform(4200, 4800)
            dte = rng.integers(7, 180)
            T = dte / 365.0
            sigma = rng.uniform(0.12, 0.28)
            tau = T
            d1 = (np.log(S0 / K) + (0.04 + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            mid = S0 * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * np.exp(-0.04 * tau) * 0.5 * (
                1 + math.erf((d1 - sigma * math.sqrt(tau)) / math.sqrt(2))
            )
            w.writerow([f"{K:.2f}", f"{dte}", f"{max(mid, 0.01):.4f}", f"{S0:.2f}", "0.04", "0.013"])
