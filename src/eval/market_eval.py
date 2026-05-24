"""Evaluation helpers for real market datasets."""

from __future__ import annotations

import torch

from ..eval.metrics import relative_l2


@torch.no_grad()
def validate_market_surface(trainer, loader, device) -> float:
    trainer._eval_model().eval()
    rel, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = trainer._eval_model()(batch["params"], batch["coords"])
        rel += relative_l2(pred, batch["prices"])
        n += 1
    return rel / max(n, 1)


@torch.no_grad()
def test_market_quotes(trainer, loader, device) -> tuple:
    """Returns (mean surface rel-L2, RMSE on quoted mids)."""
    surf_rel, mse_q, n_s, n_q = 0.0, 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = trainer._eval_model()(batch["params"], batch["coords"])
        surf_rel += relative_l2(pred, batch["prices"])
        n_s += 1
        B = batch["params"].shape[0]
        H, W = batch["grid"].shape[1], batch["grid"].shape[2]
        pred_g = pred.reshape(B, H, W)
        s_grid = batch["grid"][:, :, 0, 0]
        for b in range(B):
            nq = batch["quote_coords"].shape[1]
            for q in range(nq):
                if batch["quote_mask"][b, q] < 0.5:
                    continue
                sq = batch["quote_coords"][b, q, 0]
                i = torch.argmin((s_grid[b] - sq).abs())
                pq = pred_g[b, i, 0]
                mse_q += ((pq - batch["quote_prices"][b, q]) ** 2).item()
                n_q += 1
    return surf_rel / max(n_s, 1), (mse_q / max(n_q, 1)) ** 0.5
