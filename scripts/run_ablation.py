#!/usr/bin/env python3
"""
Ablation study: train and evaluate CF-HPINO variants vs baselines.

Variants:
  - cf_hpino_full      Full model (Fourier + log grid + hybrid loss)
  - no_fourier         n_fourier_freq = 0
  - linear_grid        Linear spot grid (no log spacing)
  - pure_fno           Operator only (no physics fusion)
  - pinn               Standard PINN baseline

Results written to results/ablation/summary.csv and summary.md

Usage:
  python scripts/run_ablation.py --device cuda
  python scripts/run_ablation.py --device cuda --epochs-per-stage 20 --skip-train  # eval only
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
from torch.utils.data import DataLoader, Subset

from src.baselines.fno_baseline import build_pure_fno
from src.baselines.pinn import build_pinn
from src.cf_hpino_loss import CFHPINOLoss
from src.cf_hpino_model import CF_HPINO
from src.data import OptionPricingDataset
from src.data.sampling import collate_option_batch
from src.eval.metrics import evaluate_model, results_to_table
from src.train.trainer import CFHPINOTrainer, TrainConfig, _split_indices
from src.utils.config_loader import build_trainer_from_experiment, load_experiment


ABLATION_CONFIGS = {
    "cf_hpino_full": "configs/ablation_full.yaml",
    "no_fourier": "configs/ablation_no_fourier.yaml",
    "linear_grid": "configs/ablation_linear_grid.yaml",
}


def _build_test_loader(exp: dict, seed: int = 42) -> DataLoader:
    """Shared test split for fair comparison across variants trained on same data config."""
    ds = OptionPricingDataset(exp["data"])
    _, _, test_idx = _split_indices(
        len(ds),
        exp["train"].val_fraction,
        exp["train"].test_fraction,
        seed,
    )
    return DataLoader(
        Subset(ds, test_idx),
        batch_size=exp["train"].batch_size,
        shuffle=False,
        collate_fn=collate_option_batch,
    )


def train_variant(name: str, config_path: Path, device: str, epochs_override: int | None) -> Path:
    exp = load_experiment(config_path)
    if epochs_override:
        exp["train"].epochs_per_stage = epochs_override
    exp["train"].device = device
    model, loss_fn, trainer = build_trainer_from_experiment(exp, device=device)
    print(f"\n{'='*60}\nTraining ablation: {name}\n{'='*60}")
    trainer.train(config_path=str(config_path))
    return Path(exp["train"].checkpoint_dir) / "best.pt"


def eval_checkpoint(
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    batch = next(iter(loader))
    params = batch["params"].to(device)
    coords = batch["coords"].to(device)
    target = batch["prices"].to(device)
    # Full loader pass for more stable metrics
    model.eval()
    mse_sum, rel_sum, n = 0.0, 0.0, 0
    inf_ms = 0.0
    with torch.no_grad():
        for b in loader:
            p = b["params"].to(device)
            c = b["coords"].to(device)
            t = b["prices"].to(device)
            r = evaluate_model(model, p, c, t, device=device)
            mse_sum += r.mse
            rel_sum += r.relative_l2
            inf_ms += r.inference_ms
            n += 1
    return {
        "variant": name,
        "mse": mse_sum / max(n, 1),
        "relative_l2": rel_sum / max(n, 1),
        "inference_ms": inf_ms / max(n, 1),
    }


def load_cf_hpino_from_ckpt(ckpt_path: Path, config_path: Path, device: torch.device) -> CF_HPINO:
    from src.utils.config_loader import model_config_from_dict

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CF_HPINO(model_config_from_dict(ckpt["model_config"]))
    if ckpt.get("ema") is not None:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def run_baselines(exp: dict, device: torch.device) -> dict:
    """Train-free baselines on same architecture sizes as ablation config."""
    mcfg = exp["model"]
    fno_kw = dict(
        n_spatial=mcfg.n_spatial,
        n_temporal=mcfg.n_temporal,
        hidden_dim=mcfg.hidden_dim,
        fno_depth=mcfg.fno_depth,
        fno_modes=mcfg.fno_modes,
    )
    loader = _build_test_loader(exp)
    results = {}

    # Untrained baselines (architecture only) — optional quick reference
    pure = build_pure_fno(backbone="fno", **fno_kw).to(device)
    results["pure_fno_untrained"] = eval_checkpoint("pure_fno_untrained", pure, loader, device)

    pinn = build_pinn().to(device)
    results["pinn_untrained"] = eval_checkpoint("pinn_untrained", pinn, loader, device)

    return results


def main():
    parser = argparse.ArgumentParser(description="CF-HPINO ablation study")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs-per-stage", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing checkpoints")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(ABLATION_CONFIGS.keys()),
        help="Subset of: cf_hpino_full no_fourier linear_grid",
    )
    parser.add_argument("--out-dir", default="results/ablation")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()

    # Reference experiment for shared test loader
    ref_exp = load_experiment(ROOT / "configs" / "ablation_full.yaml")
    test_loader = _build_test_loader(ref_exp)

    for name in args.variants:
        if name not in ABLATION_CONFIGS:
            print(f"Unknown variant {name}, skip.")
            continue
        cfg_path = ROOT / ABLATION_CONFIGS[name]
        ckpt_path = Path(load_experiment(cfg_path)["train"].checkpoint_dir) / "best.pt"

        if not args.skip_train:
            ckpt_path = train_variant(
                name, cfg_path, str(device), args.epochs_per_stage
            )

        if not ckpt_path.exists():
            print(f"Missing checkpoint for {name}: {ckpt_path}")
            continue

        model = load_cf_hpino_from_ckpt(ckpt_path, cfg_path, device)
        row = eval_checkpoint(name, model, test_loader, device)
        row["checkpoint"] = str(ckpt_path)
        rows.append(row)
        print(f"  {name}: rel-L2={row['relative_l2']:.4e} MSE={row['mse']:.4e}")

    # Optional: untrained baseline reference on same test set
    if not args.skip_train:
        print("\nEvaluating untrained architecture baselines (reference only)...")
        baseline_rows = run_baselines(ref_exp, device)
        for k, v in baseline_rows.items():
            v["checkpoint"] = "N/A"
            rows.append(v)

    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Ablation study results\n\n")
        f.write(f"Device: `{device}` | Time: {time.time() - t_start:.0f}s\n\n")
        f.write("| Variant | Rel-L2 | MSE | Inf (ms) |\n")
        f.write("|---------|--------|-----|----------|\n")
        for r in sorted(rows, key=lambda x: x["relative_l2"]):
            f.write(
                f"| {r['variant']} | {r['relative_l2']:.4e} | {r['mse']:.4e} | {r['inference_ms']:.1f} |\n"
            )
        f.write("\nSee [LIMITATIONS.md](../LIMITATIONS.md) for interpretation.\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nAblation complete. Results: {csv_path}")
    if rows:
        print(
            "\n".join(
                f"{r['variant']:20} rel-L2={r['relative_l2']:.4e}"
                for r in sorted(rows, key=lambda x: x["relative_l2"])
            )
        )


if __name__ == "__main__":
    main()
