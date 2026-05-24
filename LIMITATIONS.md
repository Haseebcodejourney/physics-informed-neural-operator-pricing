# CF-HPINO: Known Limitations

This document lists **current limitations** of the implementation as of the research codebase in this repository. Use it for honest positioning in papers, reviews, and deployment decisions.

---

## 1. Scope of instruments and models

| Topic | Limitation |
|-------|------------|
| Option type | Primary focus on **European** calls. American pricing uses approximate penalties / PSOR references, not a full LCP neural solver. |
| Underlying models | **Black–Scholes**, **fractional BS** (Caputo time), and **Merton jump-diffusion** only. No Heston, SABR, local/stochastic vol surfaces, baskets, or rates exotics. |
| Payoff | Mostly **call** payoffs. Put–call parity is not enforced in the loss. |
| Market realism | **Real data pipeline** added (`fetch_market_data`, `train_market`) via Yahoo SPY chains; SPX/index and full smile calibration remain limited. |

---

## 2. Ground truth and physics fidelity

| Topic | Limitation |
|-------|------------|
| Black–Scholes labels | Analytic formula — **high quality** within model assumptions. |
| Fractional BS labels | Generated with an **explicit FDM** scheme and truncated Caputo history — not a published high-order reference solver. |
| Merton labels | **Monte Carlo** with finite paths → noise and bias in training targets. |
| Merton PIDE loss | Jump integral uses **subsampled** collocation and a simplified Gaussian jump density in log-y. |
| Fractional operator | **L1 Caputo** approximation on a uniform time grid — not full memory on irregular times. |
| Dividends / rates | Constant \(r, q\) per sample — no term structure. |

---

## 3. Neural architecture

| Topic | Limitation |
|-------|------------|
| FNO grid | Predictions are defined on a **fixed regular** \((S,t)\) lattice. Off-grid market quotes require interpolation. |
| Domain | Spatial domain uses **global** `S_min` / `S_max` (or log bounds), not per-deal adaptive boxes. |
| Fusion | Cross-attention fusion is **not** a provably conservative or arbitrage-free structural prior. |
| Size | Very large surfaces (e.g. full SPX strike × expiry panels) may require memory optimizations not included here. |
| DeepONet path | Less tested than FNO in the provided configs. |

---

## 4. Training protocol

| Topic | Limitation |
|-------|------------|
| Generalization metric | Val/test splits are over **parameter vectors** \(\theta\), not over calendar time or market regimes. |
| Convergence | Short runs (smoke / few epochs) yield **high relative error** (\(\gg 10\%\)) — not representative of final capability. |
| Full accuracy | `research_accuracy.yaml` needs **GPU + long runtime**; results are not guaranteed without hyperparameter tuning. |
| Adaptive loss weights | Learned \(\lambda\) terms improve stability but complicate **interpretability** and reproducibility across seeds. |
| L-BFGS polish | Optional and expensive; applied on limited batches, not a full production optimization pass. |
| Reproducibility | GPU nondeterminism and MC noise can cause **small run-to-run variation**. |

---

## 5. Evaluation and baselines

| Topic | Limitation |
|-------|------------|
| Baseline tuning | PINN and Pure FNO may be **under-tuned** relative to CF-HPINO unless trained with the same budget (see `scripts/run_ablation.py`). |
| Greeks | Model Greeks via autograd are **not** systematically validated against closed-form or FD Greeks in all configs. |
| Classical baseline | Evaluation uses analytic BS where applicable — **not** a full industrial pricer stack. |
| Statistical rigor | No built-in multi-seed confidence intervals or hypothesis tests. |
| Ablation coverage | `run_ablation.py` covers main architectural variants; **not** every combination of flags. |

---

## 6. Software and deployment

| Topic | Limitation |
|-------|------------|
| Cloud / DDP | SageMaker and `torchrun` paths are **starter** integrations — not load-tested at scale. |
| Dependencies | PyTorch version changes may affect AMP, FFT (FNO), and autograd behavior. |
| Security | No secrets management or production API hardening — research code only. |
| Monitoring | CSV/JSON logs only — no MLflow/W&B integration out of the box. |

---

## 7. Claims safe vs unsafe for publications

### Supported (with proper full training + evaluation)

- A **unified hybrid** neural-operator + physics-informed framework for parametric option surfaces under BS / fractional BS / Merton **in code**.
- **Held-out parameter** validation protocol (train/val/test splits).
- Comparison against **PINN**, **pure FNO**, and **classical BS** on the same synthetic pipeline.
- Extensions toward **Greeks** and **calibration hooks**.

### Not supported without additional work

- Production desk pricing or regulatory model-risk sign-off.
- “Arbitrage-free” or “market-complete” guarantees.
- State-of-the-art accuracy on **live SPX** chains without dedicated real-data experiments.
- American and path-dependent exotics as primary claims.
- Proof that hybrid fusion always dominates all methods on all metrics.

---

## 8. Mitigation roadmap (recommended order)

1. Complete **`research_accuracy.yaml`** training; report test **relative L2** and MSE on held-out \(\theta\).
2. Run **`python scripts/run_ablation.py`** and include ablation table in the paper.
3. Add **Greeks validation** vs analytic BS on a fixed parameter cell.
4. Improve **Merton** ground truth (more MC paths or Fourier method).
5. Pilot **one real expiry** from SPX CSV with implied vol per strike.
6. Strengthen **American** treatment (LCP-based loss or benchmark against industrial FD).

---

## References in this repo

- Training guide: [docs/RESEARCH_TRAINING.md](docs/RESEARCH_TRAINING.md)
- Ablation runner: [scripts/run_ablation.py](scripts/run_ablation.py)
- Configs: [configs/](configs/)
