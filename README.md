# Physics-Informed Neural Operator Pricing (CF-HPINO)

[![Repository](https://img.shields.io/badge/GitHub-physics--informed--neural--operator--pricing-blue)](https://github.com/Haseebcodejourney/physics-informed-neural-operator-pricing)

**Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator (CF-HPINO)** for financial derivative pricing.

> **Repository:** https://github.com/Haseebcodejourney/physics-informed-neural-operator-pricing

Research codebase for European and American options under:

- Black–Scholes (BS)
- Fractional Black–Scholes (Caputo time derivative)
- Merton jump-diffusion

The model combines a **neural operator** (FNO or DeepONet) for parametric generalization with a **physics-informed** branch and hybrid fusion, trained with data + PDE residual + boundary losses and adaptive weighting.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [End-to-end workflow](#end-to-end-workflow)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Demo notebook](#demo-notebook)
9. [Configuration (YAML)](#configuration-yaml)
10. [Multi-GPU and cloud](#multi-gpu-and-cloud)
11. [Market data (SPX / CSV)](#market-data-spx--csv)
12. [Model and parameters](#model-and-parameters)
13. [Baselines](#baselines)
14. [Outputs and checkpoints](#outputs-and-checkpoints)
15. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.10+
- PyTorch 2.1+
- NumPy, SciPy, Matplotlib, PyYAML, tqdm
- Optional: CUDA GPU for full training (CPU works for `smoke_test.yaml`)
- Optional: Jupyter for the demo notebook

---

## Installation

Open a terminal in the project folder:

```bash
cd cf_hpino
pip install -r requirements.txt
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"
```

All commands below assume your working directory is **`cf_hpino`** (the folder that contains `src/`, `scripts/`, and `configs/`).

---

## Project structure

```
cf_hpino/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/                  # Experiment YAML files
│   ├── black_scholes.yaml    # Full BS training
│   ├── fractional_bs.yaml    # Fractional BS stage
│   ├── merton_jump.yaml      # Merton jump-diffusion
│   └── smoke_test.yaml       # Fast CPU smoke test (~20 s)
├── notebooks/
│   └── cf_hpino_demo.ipynb   # Interactive walkthrough
├── scripts/
│   ├── train.py              # Main training (single GPU / CPU)
│   ├── train_ddp.py          # Multi-GPU training
│   ├── evaluate.py           # Metrics + plots vs baselines
│   └── export_cloud.py       # SageMaker / GCP bundle
├── src/
│   ├── cf_hpino_model.py     # CF-HPINO architecture
│   ├── cf_hpino_loss.py      # Hybrid loss (data + physics + BC)
│   ├── fractional_ops.py     # Caputo / Grünwald-Letnikov helpers
│   ├── data/
│   │   ├── synthetic_pde.py  # BS / fractional BS / Merton data
│   │   ├── sampling.py         # Parameter & grid sampling
│   │   └── market_loader.py    # Real option CSV loader
│   ├── baselines/
│   │   ├── pinn.py             # Standard PINN
│   │   ├── fno_baseline.py     # Pure FNO (no physics hybrid)
│   │   └── classical.py        # Analytic BS / FD / MC
│   ├── train/
│   │   ├── trainer.py          # Adam, L-BFGS, curriculum, checkpoints
│   │   └── ddp_utils.py        # Distributed training helpers
│   ├── eval/
│   │   ├── metrics.py          # MSE, rel-L2, Greeks, latency
│   │   └── plots.py            # Surfaces, convergence, heatmaps
│   └── utils/
│       └── config_loader.py    # YAML → model / loss / trainer
├── checkpoints/              # Created when training (not in git)
└── results/                  # Created when evaluating
```

---

## Quick start

**1. Smoke test (CPU, ~20 seconds)**

```bash
python scripts/train.py --config configs/smoke_test.yaml --device cpu
python scripts/evaluate.py --checkpoint checkpoints/smoke/best.pt --config configs/smoke_test.yaml --device cpu --out-dir results/smoke
```

**2. Full Black–Scholes training (GPU recommended)**

```bash
python scripts/train.py --config configs/black_scholes.yaml --device cuda
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/black_scholes.yaml --device cuda --out-dir results
```

---

## End-to-end workflow

```mermaid
flowchart LR
    A[Install deps] --> B[Pick YAML config]
    B --> C[train.py]
    C --> D[checkpoints/best.pt]
    D --> E[evaluate.py]
    E --> F[results/ metrics + plots]
```

| Step | Action | Command |
|------|--------|---------|
| 1 | Install | `pip install -r requirements.txt` |
| 2 | Train | `python scripts/train.py --config configs/<name>.yaml` |
| 3 | Evaluate | `python scripts/evaluate.py --checkpoint checkpoints/.../best.pt --config configs/<name>.yaml` |
| 4 | (Optional) Notebook | `jupyter notebook notebooks/cf_hpino_demo.ipynb` |

**Curriculum training (default in full configs):**

1. Black–Scholes  
2. Fractional Black–Scholes  
3. Merton jump-diffusion  

Each stage uses matching synthetic data and PDE loss. Edit `train.curriculum` in YAML to change stages.

---

## Training

### Main script: `scripts/train.py`

```bash
python scripts/train.py --config configs/black_scholes.yaml
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML experiment file | `configs/black_scholes.yaml` |
| `--device` | `cuda` or `cpu` | From YAML |
| `--backbone` | `fno` or `deeponet` | From YAML |
| `--epochs-per-stage` | Epochs per curriculum stage | From YAML |
| `--batch-size` | Batch size | From YAML |
| `--lr` | Learning rate | From YAML |
| `--checkpoint-dir` | Where to save `best.pt` | From YAML |

**Examples**

```bash
# Fast debug on CPU
python scripts/train.py --config configs/smoke_test.yaml --device cpu

# Override backbone and epochs
python scripts/train.py --config configs/black_scholes.yaml --backbone deeponet --epochs-per-stage 50 --device cuda

# Fractional BS preset
python scripts/train.py --config configs/fractional_bs.yaml --device cuda
```

### What happens during training

1. Load YAML → build `CF_HPINO`, `CFHPINOLoss`, `CFHPINOTrainer`
2. For each curriculum stage, build synthetic `OptionPricingDataset`
3. Each epoch: forward pass, hybrid loss, AdamW step, cosine LR schedule
4. Optional L-BFGS polish on last epoch of a stage (if enabled in trainer)
5. Early stopping on validation loss plateau
6. Save `best.pt` with `model`, `model_config`, `loss_fn`, `history`

### Programmatic training

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("cf_hpino").resolve()))

from src.utils.config_loader import load_experiment, build_trainer_from_experiment

exp = load_experiment("configs/black_scholes.yaml")
model, loss_fn, trainer = build_trainer_from_experiment(exp, device="cuda")
trainer.train()
```

---

## Evaluation

### Script: `scripts/evaluate.py`

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/smoke/best.pt \
  --config configs/smoke_test.yaml \
  --device cpu \
  --n-samples 32 \
  --out-dir results/smoke
```

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to `best.pt` (optional; untrained CF-HPINO if omitted) |
| `--config` | YAML for grid size / model (must match checkpoint) |
| `--model` | Pricing model name if no config: `black_scholes`, `fractional_bs`, `merton` |
| `--n-samples` | Number of parameter samples in test set |
| `--market-csv` | Use real quotes from CSV instead of synthetic data |
| `--demo-spx` | Generate `demo_spx.csv` in `--out-dir` |
| `--out-dir` | Folder for plots and `metrics.json` |
| `--device` | `cuda` or `cpu` |

**Compared models**

- **CF-HPINO** (from checkpoint)
- **Pure FNO** (operator only)
- **PINN** (MLP baseline)
- **Classical** analytic Black–Scholes (reference MSE in JSON)

**Generated files in `--out-dir`**

- `metrics.json` — MSE, relative L2, max error, inference ms per model  
- `convergence.png` — training loss history (if checkpoint has it)  
- `rel_error_comparison.png` — bar chart  
- `cf_hpino_error_heatmap.png` — |pred − target| on grid  

---

## Demo notebook

```bash
pip install jupyter
jupyter notebook notebooks/cf_hpino_demo.ipynb
```

The notebook:

1. Loads `configs/smoke_test.yaml`
2. Visualizes a synthetic price surface
3. Computes initial hybrid loss
4. Runs a short training loop
5. Compares CF-HPINO vs PINN vs Pure FNO

Run the notebook with kernel cwd = `cf_hpino` (or parent; the first cell adjusts `sys.path`).

---

## Configuration (YAML)

Each config has four sections:

```yaml
model:    # CF-HPINO architecture
data:     # Synthetic dataset / grid size
loss:     # PDE type and loss weights
train:    # Optimizer, curriculum, checkpoints
```

### Available configs

| File | Purpose |
|------|---------|
| `configs/smoke_test.yaml` | 16×8 grid, 3 epochs, CPU-friendly |
| `configs/black_scholes.yaml` | Full BS, 64×32 grid, 50 epochs/stage |
| `configs/fractional_bs.yaml` | Fractional BS curriculum |
| `configs/merton_jump.yaml` | Merton with DeepONet backbone |

### Example: `configs/black_scholes.yaml`

```yaml
model:
  backbone: fno
  hidden_dim: 128
  n_spatial: 64
  n_temporal: 32
  fractional_order: 0.8

data:
  model: black_scholes
  n_param_samples: 512
  option_style: european

loss:
  pde_type: black_scholes
  adaptive: true
  lambda_physics: 0.1

train:
  epochs_per_stage: 50
  batch_size: 8
  lr: 0.001
  curriculum: [black_scholes]
  checkpoint_dir: checkpoints
```

Create a custom YAML by copying one of these files and editing paths under `train.checkpoint_dir`.

---

## Multi-GPU and cloud

### Local multi-GPU (PyTorch DDP)

```bash
torchrun --nproc_per_node=2 scripts/train_ddp.py
```

Uses `src/train/ddp_utils.py` for process group setup.

### AWS SageMaker / container bundle

```bash
# Package source for upload
python scripts/export_cloud.py --mode package --out cloud_bundle

# Inside training container (SM_MODEL_DIR set by SageMaker)
python scripts/export_cloud.py --mode train
```

Environment variables used:

- `SM_MODEL_DIR` — output directory  
- `SM_NUM_GPUS` — GPU count  
- `SM_HOSTS` / `SM_CURRENT_HOST` — multi-node (via `ddp_utils.py`)  

For GCP Vertex AI, use the same `cloud_bundle` as a custom training image entrypoint with `scripts/train.py` or `export_cloud.py --mode train`.

---

## Market data (SPX / CSV)

### CSV format

Required columns (names are flexible; aliases supported):

| Field | Aliases |
|-------|---------|
| Strike | `strike`, `K` |
| Price | `mid`, `close`, `price` |
| Expiry | `expiry`, `maturity`, `T`, `dte` (days) |
| Spot | `spot`, `S0`, `underlying` |
| Rate | `rate`, `r` |
| Dividend | `div_yield`, `q` |
| Vol (optional) | `iv`, `sigma` — if missing, implied vol is estimated |

### Generate demo file

```bash
python scripts/evaluate.py --demo-spx --out-dir results
# Creates results/demo_spx.csv
```

### Train / evaluate on market CSV

```python
from src.data import MarketOptionDataset, MarketLoaderConfig

ds = MarketOptionDataset(MarketLoaderConfig(csv_path="path/to/spx_options.csv"))
```

Or evaluation:

```bash
python scripts/evaluate.py --market-csv results/demo_spx.csv --config configs/black_scholes.yaml --out-dir results/market
```

---

## Model and parameters

### Parameter vector θ (8 dimensions)

| Index | Symbol | Meaning |
|-------|--------|---------|
| 0 | r | Risk-free rate |
| 1 | σ | Volatility |
| 2 | K | Strike |
| 3 | T | Maturity (years) |
| 4 | α | Fractional order (fractional BS) |
| 5 | λ_J | Jump intensity (Merton) |
| 6 | μ_J | Log-jump mean (Merton) |
| 7 | q | Dividend yield |

Coordinates passed to the network are **normalized** `(S, t) ∈ [0, 1]²` on a structured grid for FNO.

### Loss function

```
L = λ_data·L_data + λ_physics·L_PDE + λ_operator·L_op + λ_boundary·L_BC + λ_am·L_american
```

With `adaptive: true` in YAML, weights are learned via uncertainty weighting (see `cf_hpino_loss.py`).

### Greeks

```python
greeks = model.greeks(params, coords, compute=("delta", "gamma", "vega"))
```

### Inverse problem (calibration)

```python
theta_fit = model.calibrate_parameters(market_prices, coords, param_mask, initial_params)
```

---

## Baselines

| Module | Class | Description |
|--------|-------|-------------|
| `src/baselines/pinn.py` | `StandardPINN` | MLP + physics loss |
| `src/baselines/fno_baseline.py` | `PureFNO` | FNO/DeepONet without hybrid fusion |
| `src/baselines/classical.py` | `ClassicalPricer` | Closed-form BS, FD, Merton MC |

Evaluation script loads these automatically for comparison.

---

## Outputs and checkpoints

### Checkpoint format (`best.pt`)

```python
{
  "model": state_dict,
  "model_config": { ... },   # required to reload correct architecture
  "loss_fn": state_dict,
  "optimizer": state_dict,
  "stage": str,
  "epoch": int,
  "history": [ {"stage", "epoch", "loss", "time_s"}, ... ],
}
```

### Load checkpoint in Python

```python
from src.cf_hpino_model import CF_HPINO
from src.utils.config_loader import model_config_from_dict
import torch

ckpt = torch.load("checkpoints/smoke/best.pt", map_location="cpu", weights_only=False)
model = CF_HPINO(model_config_from_dict(ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
```

Always use the **same** `configs/*.yaml` (or `model_config` in the checkpoint) for evaluation grid sizes.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: numpy/torch` | Run `pip install -r requirements.txt` from `cf_hpino` |
| Checkpoint size mismatch on evaluate | Pass `--config` matching training (e.g. `smoke_test.yaml` for smoke checkpoint) |
| CUDA OOM | Reduce `n_spatial`, `n_temporal`, or `batch_size` in YAML |
| Slow on CPU | Use `configs/smoke_test.yaml` first |
| `FNO expects grid H*W=...` | Data grid must match `model.n_spatial` × `model.n_temporal` |
| Training loss NaN | Lower `lr` or `lambda_physics` in YAML |

---

## Citation and extensions

If you use this code in a publication, cite your CF-HPINO paper and acknowledge:

- Fourier Neural Operator (FNO) backbone  
- Physics-informed neural network (PINN) loss formulation  
- Caputo fractional derivative (L1 scheme) for fractional BS  

**Suggested extensions:** American LCP solver in loss, real SPX chains, AWS SageMaker hyperparameter sweeps, inverse calibration experiments on listed options.

---

## License

Use and modify for research. Add your institution’s license file if distributing publicly.
