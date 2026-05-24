#!/usr/bin/env python3
"""Train CF-HPINO from YAML config and/or CLI overrides."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import (  # noqa: E402
    build_trainer_from_experiment,
    load_experiment,
    merge_cli_overrides,
)


def main():
    parser = argparse.ArgumentParser(description="Train CF-HPINO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/black_scholes.yaml",
        help="Path to YAML experiment config",
    )
    parser.add_argument("--backbone", choices=["fno", "deeponet"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs-per-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    exp = load_experiment(config_path)
    exp = merge_cli_overrides(
        exp,
        backbone=args.backbone,
        device=args.device,
        epochs_per_stage=args.epochs_per_stage,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
    )

    model, loss_fn, trainer = build_trainer_from_experiment(exp)
    print(f"Config: {config_path}")
    print(f"Device: {trainer.device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Curriculum: {trainer.cfg.curriculum}")

    trainer.train()
    ckpt = Path(trainer.cfg.checkpoint_dir) / "best.pt"
    print(f"Training complete. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
