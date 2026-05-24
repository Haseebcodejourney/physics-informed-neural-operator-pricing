#!/usr/bin/env python3
"""
Evaluate CF-HPINO vs PINN, Pure FNO, and classical pricers.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --out-dir results
"""

from __future__ import annotations

import argparse
from typing import Optional
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.baselines.classical import ClassicalConfig, ClassicalPricer, PricingMethod
from src.baselines.fno_baseline import build_pure_fno
from src.baselines.pinn import build_pinn
from src.cf_hpino_loss import CFHPINOLoss, LossConfig
from src.cf_hpino_model import CF_HPINO, build_cf_hpino
from src.utils.config_loader import model_config_from_dict
from src.data import DatasetConfig, OptionPricingDataset, PricingModel
from src.data.market_loader import MarketLoaderConfig, MarketOptionDataset, generate_demo_spx_csv
from src.data.sampling import collate_option_batch
from src.eval.metrics import compare_methods, results_to_table
from src.eval.plots import plot_comparison_bar, plot_convergence, plot_error_heatmap


def load_checkpoint(path: Path, device: torch.device, config_path: Optional[Path] = None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_config" in ckpt:
        model = CF_HPINO(model_config_from_dict(ckpt["model_config"]))
    elif config_path and config_path.exists():
        from src.utils.config_loader import load_experiment

        model = CF_HPINO(load_experiment(config_path)["model"])
    else:
        model = build_cf_hpino()
    model.load_state_dict(ckpt["model"])
    history = ckpt.get("history", [])
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="", help="YAML config if checkpoint lacks model_config")
    parser.add_argument("--model", default="black_scholes")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--market-csv", type=str, default="")
    parser.add_argument("--demo-spx", action="store_true", help="Generate demo SPX CSV")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.demo_spx:
        csv_path = out_dir / "demo_spx.csv"
        generate_demo_spx_csv(csv_path)
        print(f"Wrote {csv_path}")

    cfg_path = Path(args.config) if args.config else ROOT / "configs" / "black_scholes.yaml"
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    from src.utils.config_loader import load_experiment

    exp = load_experiment(cfg_path) if cfg_path.exists() else None
    data_cfg = (
        exp["data"]
        if exp
        else DatasetConfig(model=PricingModel(args.model), n_param_samples=args.n_samples)
    )
    data_cfg.n_param_samples = args.n_samples

    if args.market_csv:
        ds = MarketOptionDataset(MarketLoaderConfig(csv_path=args.market_csv))
    else:
        ds = OptionPricingDataset(data_cfg)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_option_batch)
    batch = next(iter(loader))
    params, coords, target = batch["params"], batch["coords"], batch["prices"]

    mcfg = exp["model"] if exp else None
    fno_kw = {}
    if mcfg:
        fno_kw = dict(
            n_spatial=mcfg.n_spatial,
            n_temporal=mcfg.n_temporal,
            hidden_dim=mcfg.hidden_dim,
            fno_depth=mcfg.fno_depth,
            fno_modes=mcfg.fno_modes,
        )
    models = {
        "PureFNO": build_pure_fno(
            backbone=mcfg.backbone.value if mcfg else "fno", **fno_kw
        ),
        "PINN": build_pinn(),
    }
    if args.checkpoint and Path(args.checkpoint).exists():
        cf, history = load_checkpoint(Path(args.checkpoint), device, cfg_path)
        models["CF-HPINO"] = cf
        if history:
            plot_convergence(history, save_path=out_dir / "convergence.png")
            print(f"Saved convergence plot to {out_dir / 'convergence.png'}")
    else:
        models["CF-HPINO"] = build_cf_hpino()
        print("No checkpoint found — evaluating untrained CF-HPINO.")

    # Classical reference on first batch element
    classical = ClassicalPricer(ClassicalConfig(method=PricingMethod.ANALYTIC))
    S = coords[..., 0] * (300.0 - 20.0) + 20.0  # default loss scaling
    t = coords[..., 1] * params[:, 3:4]
    classical_pred = classical.price(S, t, params, model=args.model)

    results = compare_methods(
        {k: v.to(device) for k, v in models.items()},
        params.to(device),
        coords.to(device),
        target.to(device),
        device=args.device,
    )

    table = results_to_table(results)
    print(table)

    plot_comparison_bar(
        {k: v.relative_l2 for k, v in results.items()},
        save_path=out_dir / "rel_error_comparison.png",
    )

    cf_model = models["CF-HPINO"].to(device)
    with torch.no_grad():
        pred = cf_model(params[:1].to(device), coords[:1].to(device))
    plot_error_heatmap(
        batch["grid"][:1],
        pred.view(-1),
        target[:1].view(-1),
        save_path=out_dir / "cf_hpino_error_heatmap.png",
    )

    metrics_json = {
        name: {
            "mse": r.mse,
            "relative_l2": r.relative_l2,
            "max_abs_error": r.max_abs_error,
            "inference_ms": r.inference_ms,
            "greeks": r.greeks,
        }
        for name, r in results.items()
    }
    metrics_json["classical_mse"] = torch.mean((classical_pred - target) ** 2).item()

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
