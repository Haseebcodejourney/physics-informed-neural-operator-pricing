# Research-Grade Training Guide

This document explains how to train CF-HPINO for **publishable** scientific results (not the quick smoke test).

## Smoke test vs research training

| | Smoke (`smoke_test.yaml`) | Research (`research_*.yaml`) |
|---|---------------------------|------------------------------|
| Purpose | Verify code runs | Minimize error on held-out parameters |
| Samples | 12 | 200–800 |
| Grid | 16×8 | 48×24 to 64×32 |
| Epochs | 3 | 60–120 per stage |
| Validation | No | Yes (15% held out) |
| Test set | No | Yes (10% held out, never trained on) |
| Best checkpoint | Training loss | **Validation relative L2** |

## Recommended training order

### Step 1 — Black-Scholes foundation (required)

```bash
python scripts/train.py --config configs/research_bs.yaml --device cuda
```

**Target metrics** (test set, after convergence):

| Test rel-L2 | Quality |
|-------------|---------|
| < 0.05 | Strong (paper-ready BS results) |
| 0.05 – 0.10 | Good |
| 0.10 – 0.20 | Needs more epochs or tuning |
| > 0.20 | Not ready for publication |

Check: `checkpoints/research_bs/logs/all_test_results.json`

### Step 2 — Medium multi-grid run (optional bridge)

```bash
python scripts/train.py --config configs/research_medium.yaml --device cuda
```

### Step 3 — Full curriculum (BS → fractional → Merton)

```bash
python scripts/train.py --config configs/research_full.yaml --device cuda
```

Expect **several hours on GPU** (Merton MC data generation is slow).

### Resume interrupted training

```bash
python scripts/train.py --config configs/research_bs.yaml --resume checkpoints/research_bs/best.pt --device cuda
```

## What happens each epoch

1. **Train** on 75% of parameter samples (hybrid loss: data + PDE + boundary).
2. **Validate** on 15% (no gradient) — compute val loss and **val rel-L2**.
3. If val rel-L2 improves → save `best.pt`.
4. Early stop if no improvement for `patience` epochs (after `min_epochs`).
5. After each curriculum stage → evaluate on **test** 10% → `logs/test_<stage>.json`.

## Logs and artifacts

```
checkpoints/research_bs/
├── best.pt              # Best model (lowest val rel-L2)
├── last.pt              # Final epoch
├── epoch_*.pt           # Periodic snapshots
└── logs/
    ├── metrics.csv      # Per-epoch train/val metrics
    ├── manifest.json    # Experiment metadata
    ├── test_black_scholes.json
    └── all_test_results.json
```

## Final evaluation for the paper

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/research_bs/best.pt \
  --config configs/research_bs.yaml \
  --device cuda \
  --n-samples 100 \
  --out-dir results/research_bs
```

Compare CF-HPINO vs PINN vs Pure FNO vs classical BS in `results/research_bs/metrics.json`.

## Hyperparameter tips

| Issue | Try |
|-------|-----|
| Val rel-L2 stuck > 0.15 | More `epochs_per_stage`, lower `lr` (5e-4), increase `n_param_samples` |
| Physics unstable | Lower `lambda_physics` in YAML or keep `adaptive: true` |
| OOM on GPU | Reduce `batch_size`, `n_spatial`, `n_temporal`, or `hidden_dim` |
| Merton too slow | Lower `merton_paths` in data section (e.g. 10000) for training |

## Reporting in a paper

Report at minimum:

- Train / val / test **relative L2** and **MSE** on option prices
- Comparison table vs PINN, Pure FNO, FD/MC
- Grid resolution (`n_spatial` × `n_temporal`)
- Number of parameter samples and split fractions
- Total training time and hardware (GPU model)

## CPU-only fallback

```bash
python scripts/train.py --config configs/research_medium.yaml --device cpu
```

Use GPU for `research_bs` and `research_full` when possible.
