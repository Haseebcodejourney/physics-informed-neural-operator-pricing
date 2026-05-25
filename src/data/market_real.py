"""
Real market option data for CF-HPINO training and testing.

Pipeline:
  1. Load CSV from fetch_market_data / Yahoo / your broker export
  2. Group quotes by expiry (one surface per maturity slice)
  3. Build (S,t) grid with log-spaced spot
  4. Target surface: interpolate real mids at t≈0, extend in time via BS with smile IV
  5. Supervise quoted strikes explicitly (market_quote loss in cf_hpino_loss.py)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from .market_loader import _COLUMN_ALIASES, _resolve_column, implied_vol_newton
from .sampling import ParamRanges, build_coordinate_grid
from .synthetic_pde import black_scholes_call


@dataclass
class RealMarketConfig:
    csv_path: str
    n_spatial: int = 64
    n_temporal: int = 32
    normalize_coords: bool = True
    log_spatial: bool = True
    min_quotes_per_expiry: int = 8
    min_moneyness: float = 0.7
    max_moneyness: float = 1.3
    day_count: float = 365.0
    # S grid uses spot * [min_ratio, max_ratio]
    spot_range: Tuple[float, float] = (0.75, 1.25)
    param_ranges: Optional[ParamRanges] = None
    # Split indices filled after load
    train_indices: Optional[List[int]] = None
    val_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None


def load_market_csv(path: Path) -> List[Dict]:
    """Load standard market CSV (from market_fetch or custom)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        cols = {k: _resolve_column(header, k) for k in _COLUMN_ALIASES}
        extra = {}
        for h in header:
            hl = h.lower()
            if hl in ("dte", "trade_date", "ticker", "iv", "impliedvolatility"):
                extra[hl] = h

        if cols["strike"] is None or cols["price"] is None:
            raise ValueError(f"CSV needs strike and mid/price. Header: {header}")

        rows = []
        for row in reader:
            try:
                strike = float(row[cols["strike"]])
                mid = float(row[cols["price"]])
            except (ValueError, KeyError):
                continue
            if mid <= 0:
                continue

            spot = float(row[cols["spot"]]) if cols["spot"] else 0.0
            rate = float(row[cols["rate"]]) if cols["rate"] else 0.04
            div = float(row[cols["div"]]) if cols["div"] else 0.0

            if cols["expiry"]:
                exp_val = row[cols["expiry"]]
            elif "dte" in extra:
                exp_val = row[extra["dte"]]
            else:
                exp_val = "30"

            if "dte" in extra and row.get(extra["dte"]):
                dte = float(row[extra["dte"]])
                T = dte / 365.0
                expiry_key = f"dte_{int(dte)}"
            else:
                from .market_loader import _parse_expiry

                T = _parse_expiry(str(exp_val), 365.0)
                expiry_key = str(exp_val)

            iv = None
            iv_col = cols["vol"] or extra.get("iv")
            if iv_col and row.get(iv_col) not in (None, ""):
                try:
                    iv = float(row[iv_col])
                    if iv > 3.0:
                        iv = iv / 100.0
                except ValueError:
                    iv = None

            if iv is None or iv <= 0:
                if spot <= 0:
                    continue
                iv = implied_vol_newton(mid, spot, strike, T, rate, div)

            if spot <= 0:
                spot = 450.0

            moneyness = strike / spot
            if moneyness < 0.5 or moneyness > 2.0:
                continue

            opt_type = "call"
            for h in header:
                if h.lower() == "option_type":
                    opt_type = str(row.get(h, "call")).lower()
                    break
            if opt_type not in ("call", ""):
                continue

            rows.append(
                {
                    "strike": strike,
                    "mid": mid,
                    "spot": spot,
                    "rate": rate,
                    "div": div,
                    "T": T,
                    "iv": iv,
                    "expiry_key": expiry_key,
                    "moneyness": moneyness,
                }
            )
    return rows


def _group_by_expiry(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["expiry_key"], []).append(r)
    return groups


def _build_surface_from_quotes(
    group: List[Dict],
    n_spatial: int,
    n_temporal: int,
    ranges: ParamRanges,
    log_spatial: bool,
    spot_range: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
    """
    Returns:
        params (8,), grid (H,W,2), surface (H,W), meta,
        quote_coords (Nq,2), quote_prices (Nq,)
    """
    spot = float(np.median([g["spot"] for g in group]))
    T = float(np.median([g["T"] for g in group]))
    rate = float(np.median([g["rate"] for g in group]))
    div = float(np.median([g["div"] for g in group]))
    iv_atm = float(np.median([g["iv"] for g in group]))

    strikes = np.array([g["strike"] for g in group])
    mids = np.array([g["mid"] for g in group])
    ivs = np.array([g["iv"] for g in group])
    order = np.argsort(strikes)
    strikes, mids, ivs = strikes[order], mids[order], ivs[order]

    K_ref = spot
    params = torch.tensor(
        [rate, iv_atm, K_ref, T, 0.8, 0.0, 0.0, div], dtype=torch.float32
    )

    S_min = spot * spot_range[0]
    S_max = spot * spot_range[1]
    ranges_local = ParamRanges(
        S0=spot, S_min_ratio=spot_range[0], S_max_ratio=spot_range[1]
    )

    grid, meta = build_coordinate_grid(
        n_spatial,
        n_temporal,
        params.unsqueeze(0),
        ranges_local,
        normalize=True,
        log_spatial=log_spatial,
    )
    grid = grid.squeeze(0)
    meta["spot"] = spot

    if log_spatial:
        log_min, log_max = meta["log_S_min"], meta["log_S_max"]
        S_phys = torch.exp(
            grid[..., 0] * (log_max - log_min) + log_min
        )
    else:
        S_phys = grid[..., 0] * (S_max - S_min) + S_min

    t_phys = grid[..., 1] * T

    # Smile IV(K) interpolator
    iv_interp = interp1d(
        strikes, ivs, kind="linear", fill_value="extrapolate", bounds_error=False
    )
    iv_grid = torch.tensor(iv_interp(S_phys.numpy().ravel()), dtype=torch.float32).view(
        n_spatial, n_temporal
    )

    # Surface via BS with local IV at each (S, t)
    surface = torch.zeros(n_spatial, n_temporal)
    for i in range(n_spatial):
        for j in range(n_temporal):
            sig = iv_grid[i, j]
            surface[i, j] = black_scholes_call(
                S_phys[i, j : j + 1],
                t_phys[i, j : j + 1],
                params[0],
                sig,
                params[2],
                params[3],
                params[7],
            ).squeeze()

    # Anchor t=0 slice to interpolated market mids
    iv_strikes = interp1d(strikes, mids, kind="linear", fill_value="extrapolate")
    mids_on_grid = torch.tensor(
        iv_strikes(S_phys[:, 0].numpy()), dtype=torch.float32
    )
    surface[:, 0] = mids_on_grid

    # Quoted points at t=0 (normalized coords)
    quote_coords = torch.stack(
        [
            grid[:, 0, 0],
            torch.zeros(n_spatial),
        ],
        dim=-1,
    )
    # Also add raw market (K,T) points
    q_list_c, q_list_p = [], []
    for g in group:
        s_norm = (g["strike"] - S_min) / (S_max - S_min + 1e-8)
        if log_spatial:
            s_norm = (math.log(g["strike"]) - log_min) / (log_max - log_min + 1e-8)
        q_list_c.append([s_norm, 0.0])
        q_list_p.append(g["mid"])

    quote_coords = torch.tensor(q_list_c, dtype=torch.float32)
    quote_prices = torch.tensor(q_list_p, dtype=torch.float32)

    return params, grid, surface, meta, quote_coords, quote_prices


class RealMarketDataset(Dataset):
    """
    One item per expiry bucket = one training surface from real quotes.

    Includes `quote_coords` / `quote_prices` for direct market supervision.
    """

    def __init__(self, cfg: RealMarketConfig, indices: Optional[List[int]] = None):
        self.cfg = cfg
        self.ranges = cfg.param_ranges or ParamRanges()
        path = Path(cfg.csv_path)
        rows = load_market_csv(path)
        groups = _group_by_expiry(rows)

        self.samples: List[Dict] = []
        for key, grp in sorted(groups.items()):
            if len(grp) < cfg.min_quotes_per_expiry:
                continue
            m = [g["moneyness"] for g in grp]
            if min(m) > cfg.max_moneyness or max(m) < cfg.min_moneyness:
                continue
            try:
                sample = _build_surface_from_quotes(
                    grp,
                    cfg.n_spatial,
                    cfg.n_temporal,
                    self.ranges,
                    cfg.log_spatial,
                    cfg.spot_range,
                )
                self.samples.append(
                    {
                        "expiry_key": key,
                        "n_quotes": len(grp),
                        "data": sample,
                    }
                )
            except Exception:
                continue

        if not self.samples:
            raise RuntimeError(
                f"No valid expiry groups in {path}. Need >= {cfg.min_quotes_per_expiry} quotes per expiry."
            )

        self._indices = indices if indices is not None else list(range(len(self.samples)))
        self.meta = self.samples[0]["data"][3] if self.samples else {}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[self._indices[idx]]
        params, grid, surface, meta, quote_c, quote_p = s["data"]
        return {
            "params": params,
            "coords": grid.reshape(-1, 2),
            "prices": surface.reshape(-1),
            "grid": grid,
            "price_surface": surface,
            "quote_coords": quote_c,
            "quote_prices": quote_p,
            "market_mid": quote_p.mean(),
            "expiry_key": s["expiry_key"],
        }


def split_real_market_dataset(
    cfg: RealMarketConfig,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[RealMarketDataset, RealMarketDataset, RealMarketDataset]:
    """Split by expiry groups (not random rows) for honest generalization."""
    full = RealMarketDataset(cfg)
    n = len(full.samples)
    if n < 3:
        return full, full, full

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
    n_val = max(1, int(n * val_ratio))
    n_train = max(1, n - n_test - n_val)

    test_idx = perm[:n_test].tolist()
    val_idx = perm[n_test : n_test + n_val].tolist()
    train_idx = perm[n_test + n_val : n_test + n_val + n_train].tolist()

    def _subset(indices: List[int]) -> RealMarketDataset:
        ds = RealMarketDataset.__new__(RealMarketDataset)
        ds.cfg = cfg
        ds.ranges = full.ranges
        ds.samples = [full.samples[i] for i in indices]
        ds._indices = list(range(len(ds.samples)))
        ds.meta = full.meta
        return ds

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def collate_market_batch(batch: list) -> Dict[str, torch.Tensor]:
    """Pad variable-length quote lists; stack grid-sized fields."""
    base = {
        "params": torch.stack([b["params"] for b in batch], 0),
        "coords": torch.stack([b["coords"] for b in batch], 0),
        "prices": torch.stack([b["prices"] for b in batch], 0),
        "grid": torch.stack([b["grid"] for b in batch], 0),
        "price_surface": torch.stack([b["price_surface"] for b in batch], 0),
    }
    max_q = max(b["quote_coords"].shape[0] for b in batch)
    qc, qp, mask = [], [], []
    for b in batch:
        nq = b["quote_coords"].shape[0]
        pad = max_q - nq
        c = b["quote_coords"]
        p = b["quote_prices"]
        if pad > 0:
            c = torch.cat([c, torch.zeros(pad, 2)], 0)
            p = torch.cat([p, torch.zeros(pad)], 0)
        m = torch.zeros(max_q)
        m[:nq] = 1.0
        qc.append(c)
        qp.append(p)
        mask.append(m)
    base["quote_coords"] = torch.stack(qc, 0)
    base["quote_prices"] = torch.stack(qp, 0)
    base["quote_mask"] = torch.stack(mask, 0)
    return base
