#!/usr/bin/env python3
"""Evaluate CF-HPINO on held-out real market expiries."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.cf_hpino_model import CF_HPINO
from src.data.market_real import RealMarketConfig, collate_market_batch, split_real_market_dataset
from src.utils.config_loader import model_config_from_dict

from src.cf_hpino_loss import CFHPINOLoss, LossConfig
from src.eval.market_eval import test_market_quotes, validate_market_surface
from src.train.trainer import CFHPINOTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default="data/raw/spy_options.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    mcfg = RealMarketConfig(csv_path=str(csv_path))
    _, _, test_ds = split_real_market_dataset(mcfg, seed=42)
    loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_market_batch)

    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = CF_HPINO(model_config_from_dict(ckpt["model_config"]))
    if ckpt.get("ema"):
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])

    trainer = CFHPINOTrainer(
        model, CFHPINOLoss(LossConfig()), TrainConfig(device=str(device)), device=device
    )
    if ckpt.get("ema") and trainer.ema:
        trainer.ema.load_state_dict(ckpt["ema"])

    surf_rel = validate_market_surface(trainer, loader, device)
    _, quote_rmse = test_market_quotes(trainer, loader, device)

    out = {"test_surface_rel_l2": surf_rel, "test_quote_rmse": quote_rmse}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
