#!/usr/bin/env python3
"""
Train CF-HPINO on REAL market option data.

    python scripts/fetch_market_data.py --ticker SPY --max-expiries 0 --out data/raw/spy_options_full.csv
    python scripts/train_market.py --csv data/raw/spy_options_full.csv --device cuda --epochs 100
    python scripts/test_market.py --checkpoint checkpoints/market_spy/best.pt --csv data/raw/spy_options_full.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.cf_hpino_loss import CFHPINOLoss, LossConfig, LossPDEType
from src.cf_hpino_model import build_cf_hpino  # noqa: F401
from src.data.market_real import (
    RealMarketConfig,
    collate_market_batch,
    split_real_market_dataset,
)
from src.eval.market_eval import test_market_quotes, validate_market_surface
from src.train.trainer import CFHPINOTrainer, TrainConfig
from src.utils.config_loader import build_trainer_from_experiment, load_experiment, merge_cli_overrides


def _append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train CF-HPINO on real market CSV")
    parser.add_argument("--csv", default="data/raw/spy_options_full.csv")
    parser.add_argument("--config", default="configs/market_spy.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fetch", action="store_true", help="Download SPY chain first")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--max-expiries", type=int, default=0)
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if args.fetch or not csv_path.exists():
        from src.data.market_fetch import fetch_option_chain, save_chain_csv

        print(f"Fetching {args.ticker} options from Yahoo Finance...")
        df = fetch_option_chain(ticker=args.ticker, max_expiries=args.max_expiries)
        save_chain_csv(df, csv_path)
        print(f"Saved {len(df)} rows to {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No CSV at {csv_path}. Run: python scripts/fetch_market_data.py --max-expiries 0"
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
        f"Real market surfaces: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    if len(train_ds) < 2:
        raise RuntimeError(
            "Too few expiry groups. Fetch with --max-expiries 0 or lower min_quotes_per_expiry."
        )

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if exp:
        exp = merge_cli_overrides(exp, device=str(device))
        if args.epochs is not None:
            exp["train"].epochs_per_stage = args.epochs
        if args.patience is not None:
            exp["train"].patience = args.patience
        model, loss_fn, trainer = build_trainer_from_experiment(exp, device=str(device))
    else:
        model = build_cf_hpino(n_spatial=64, n_temporal=32)
        loss_fn = CFHPINOLoss(
            LossConfig(
                pde_type=LossPDEType.BLACK_SCHOLES,
                adaptive=True,
                lambda_market=4.0,
                lambda_physics=0.03,
                lambda_data=0.4,
            )
        )
        train_cfg = TrainConfig(
            device=str(device),
            checkpoint_dir="checkpoints/market_spy",
            epochs_per_stage=args.epochs or 80,
            patience=args.patience or 20,
            min_epochs=15,
            batch_size=4,
            lr=0.0006,
            use_ema=True,
            warmup_epochs=8,
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
    log_csv = ckpt_dir / "logs" / "metrics.csv"

    best_rel = float("inf")
    stale = 0
    max_epochs = train_cfg.epochs_per_stage
    patience = train_cfg.patience
    min_epochs = train_cfg.min_epochs

    print(
        f"Training up to {max_epochs} epochs (patience={patience}, min_epochs={min_epochs})"
    )

    for ep in range(max_epochs):
        t0 = time.time()
        trainer.model.train()
        train_loss = 0.0
        mkt_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer.optimizer.zero_grad(set_to_none=True)
            loss, bd = trainer.loss_fn(trainer._unwrap(), batch, return_breakdown=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), train_cfg.grad_clip)
            trainer.optimizer.step()
            if trainer.ema:
                trainer.ema.update(trainer._unwrap())
            train_loss += float(bd["total"])
            mkt_loss += float(bd.get("market", 0.0))
            n += 1
        train_loss /= max(n, 1)
        mkt_loss /= max(n, 1)

        val_rel = validate_market_surface(trainer, val_loader, device)
        if trainer.scheduler:
            if train_cfg.scheduler == "plateau":
                trainer.scheduler.step(val_rel)
            else:
                trainer.scheduler.step()

        lr = trainer.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_rel_l2": val_rel,
            "market_loss": mkt_loss,
            "lr": lr,
            "time_s": round(elapsed, 1),
        }
        trainer.history.append(row)
        _append_metrics_csv(log_csv, row)
        print(
            f"epoch {ep+1}/{max_epochs} train={train_loss:.4e} "
            f"val_relL2={val_rel:.4e} market={mkt_loss:.4e} lr={lr:.2e} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_rel < best_rel:
            best_rel = val_rel
            stale = 0
            trainer.best_metric = best_rel
            trainer.save_checkpoint(ckpt_dir / "best.pt", "market", ep)
            print(f"  -> new best val_rel_l2={best_rel:.4e}")
        else:
            stale += 1

        if ep + 1 >= min_epochs and stale >= patience:
            print(f"Early stop at epoch {ep+1} (no val improvement for {patience} epochs)")
            break

    test_rel, test_mkt = test_market_quotes(trainer, test_loader, device)
    spot = float(train_ds.meta.get("spot", 500.0))
    results = {
        "test_surface_rel_l2": test_rel,
        "test_quote_rmse": test_mkt,
        "test_quote_rmse_pct_spot": 100.0 * test_mkt / spot,
        "best_val_rel_l2": best_rel,
        "epochs_ran": ep + 1,
        "csv": str(csv_path),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "device": str(device),
    }
    with (ckpt_dir / "logs" / "market_test.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Best val rel-L2: {best_rel:.4e}")
    print(f"Test surface rel-L2: {test_rel:.4e}")
    print(f"Test quote RMSE: ${test_mkt:.2f} ({results['test_quote_rmse_pct_spot']:.2f}% of spot)")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
