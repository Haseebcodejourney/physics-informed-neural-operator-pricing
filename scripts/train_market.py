#!/usr/bin/env python3
"""
Train CF-HPINO on REAL market option data.

Step 1 — Fetch data (requires network):
    python scripts/fetch_market_data.py --ticker SPY --out data/raw/spy_options.csv

Step 2 — Train:
    python scripts/train_market.py --csv data/raw/spy_options.csv --device cuda

Step 3 — Test:
    python scripts/test_market.py --checkpoint checkpoints/market/best.pt --csv data/raw/spy_options.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.cf_hpino_loss import CFHPINOLoss, LossConfig, LossPDEType
from src.cf_hpino_model import build_cf_hpino  # noqa: F401 — fallback path
from src.data.market_real import (
    RealMarketConfig,
    collate_market_batch,
    split_real_market_dataset,
)
from src.eval.market_eval import test_market_quotes, validate_market_surface
from src.train.trainer import CFHPINOTrainer
from src.utils.config_loader import build_trainer_from_experiment, load_experiment, merge_cli_overrides


def main():
    parser = argparse.ArgumentParser(description="Train CF-HPINO on real market CSV")
    parser.add_argument("--csv", default="data/raw/spy_options.csv")
    parser.add_argument("--config", default="configs/market_spy.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fetch", action="store_true", help="Download SPY chain first")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if args.fetch or not csv_path.exists():
        from src.data.market_fetch import fetch_option_chain, save_chain_csv

        print(f"Fetching {args.ticker} options from Yahoo Finance...")
        df = fetch_option_chain(ticker=args.ticker)
        save_chain_csv(df, csv_path)
        print(f"Saved {len(df)} rows to {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No CSV at {csv_path}. Run: python scripts/fetch_market_data.py"
        )

    cfg_path = ROOT / args.config
    exp = load_experiment(cfg_path) if cfg_path.exists() else None

    mcfg = RealMarketConfig(
        csv_path=str(csv_path),
        n_spatial=64,
        n_temporal=32,
        log_spatial=True,
        min_quotes_per_expiry=8,
    )
    if exp:
        mcfg.n_spatial = exp["data"].n_spatial
        mcfg.n_temporal = exp["data"].n_temporal

    train_ds, val_ds, test_ds = split_real_market_dataset(mcfg, seed=42)
    print(
        f"Real market surfaces: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    if len(train_ds) < 2:
        raise RuntimeError(
            "Too few expiry groups. Fetch more expiries (--max-expiries 20) or lower min_quotes_per_expiry."
        )

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    if exp:
        exp = merge_cli_overrides(exp, device=str(device))
        if args.epochs:
            exp["train"].epochs_per_stage = args.epochs
        model, loss_fn, trainer = build_trainer_from_experiment(exp, device=str(device))
    else:
        from src.train.trainer import TrainConfig

        model = build_cf_hpino(n_spatial=64, n_temporal=32)
        loss_fn = CFHPINOLoss(
            LossConfig(
                pde_type=LossPDEType.BLACK_SCHOLES,
                adaptive=True,
                lambda_market=3.0,
                lambda_physics=0.05,
                lambda_data=0.5,
            )
        )
        train_cfg = TrainConfig(
            device=str(device),
            checkpoint_dir="checkpoints/market",
            epochs_per_stage=args.epochs or 80,
            batch_size=4,
            lr=0.0008,
            use_ema=True,
        )
        trainer = CFHPINOTrainer(model, loss_fn, train_cfg, device=device)

    trainer.loss_fn.set_geometry_meta(train_ds.meta)
    train_cfg = trainer.cfg

    batch_size = train_cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_market_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_market_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_market_batch
    )

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_rel = float("inf")

    for ep in range(train_cfg.epochs_per_stage):
        t0 = time.time()
        trainer.model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer.optimizer.zero_grad(set_to_none=True)
            loss, bd = trainer.loss_fn(trainer._unwrap(), batch, return_breakdown=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            trainer.optimizer.step()
            if trainer.ema:
                trainer.ema.update(trainer._unwrap())
            train_loss += bd["total"]
            n += 1
        train_loss /= max(n, 1)

        val_rel = validate_market_surface(trainer, val_loader, device)
        if trainer.scheduler:
            trainer.scheduler.step()

        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_rel_l2": val_rel,
            "market_loss": bd.get("market", 0) if n else 0,
        }
        trainer.history.append(row)
        print(
            f"epoch {ep+1}/{train_cfg.epochs_per_stage} "
            f"train={train_loss:.4e} val_relL2={val_rel:.4e} ({time.time()-t0:.1f}s)"
        )

        if val_rel < best_rel:
            best_rel = val_rel
            trainer.save_checkpoint(ckpt_dir / "best.pt", "market", ep)

    test_rel, test_mkt = test_market_quotes(trainer, test_loader, device)
    results = {
        "test_surface_rel_l2": test_rel,
        "test_quote_rmse": test_mkt,
        "best_val_rel_l2": best_rel,
        "csv": str(csv_path),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
    }
    with (ckpt_dir / "logs" / "market_test.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTest surface rel-L2: {test_rel:.4e}")
    print(f"Test quote RMSE (actual mids): {test_mkt:.4e}")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
