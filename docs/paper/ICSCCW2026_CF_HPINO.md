# Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator for Financial Derivative Pricing

**[Author One]*** · **[Author Two]** · **[Author Three]**  
*[Department], [University], [City], [Country]*  
**Corresponding author:** [email@university.edu] · ORCID: [0000-0000-0000-0000]

*ICSCCW 2026 — Times New Roman, max 8 pages. See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).*

---

## Abstract

Fast and scalable pricing of derivative contracts under varying model parameters remains a central challenge in computational finance [1,2], particularly when surfaces must be generated repeatedly for risk, calibration, and cloud-based batch workloads [3]. We propose **CF-HPINO** (Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator), coupling a parametric neural operator backbone [4–6] with a physics-informed branch [7,8] and attention-based fusion [9]. Training combines PDE residuals for Black–Scholes [10], fractional Black–Scholes with Caputo differentiation [11,12], and Merton jump-diffusion [13], with boundary and market-quote terms. On synthetic benchmarks, CF-HPINO achieves lower relative L₂ error than PINN [7,14] and pure FNO [4] baselines. A preliminary SPY experiment (5,283 quotes) shows improved but non-production quote error (RMSE ≈ $379). The open implementation [24] aligns with soft-computing and AI-in-finance themes [18,19].

**Keywords:** physics-informed neural networks; neural operators; option pricing; fractional calculus; soft computing; cloud computing.

---

## 1. Introduction

Option pricing solves parabolic PDEs or PIDEs for the value function V(S,t) [10,13,1]. Classical finite-difference and Monte Carlo methods [2,15] are accurate but costly for full parametric surfaces over strikes, maturities, and model parameters [15,16].

Deep learning approaches include early neural pricers [16,17], **physics-informed neural networks (PINNs)** [7,14,8], high-dimensional deep PDE solvers [20,21], and **neural operators** [4,5,6]. PINNs embed PDE residuals [7]; operators amortize inference over parameter families [4,6]. Neither alone is ideal for cloud-scale pricing: PINNs struggle in high-dimensional parameter spaces [8]; pure operators may violate physics without constraints [7,4].

ICSCCW emphasizes soft computing and AI in finance. Fuzzy and neuro-fuzzy foundations [18,19] motivate learned fusion and uncertainty-aware loss weighting [22]. We present a hybrid architecture with cloud-ready training [3,24].

**Contributions:** (1) CF-HPINO architecture [4,5,7,9]; (2) unified loss for BS, fractional BS [11,12], and Merton [13]; (3) open pipeline [24]; (4) empirical study with transparent limitations.

---

## 2. Methodology

### 2.1 Problem

Approximate V̂(S,t;θ) ≈ V(S,t;θ) for θ = [r, σ, K, T, α, λ_J, μ_J, q] [1,13], on a normalized grid with optional log-spatial coordinates [16].

### 2.2 Governing equations

- **Black–Scholes** [10]: ∂V/∂t + ½σ²S²∂²V/∂S² + (r−q)S∂V/∂S − rV = 0.  
- **Fractional BS** [11,12]: Caputo ∂^α_t V with L₁ discretization.  
- **Merton** [13]: jump integral with quadrature [2].

### 2.3 Architecture

Operator path: FNO [4,6] or DeepONet [5]; physics MLP [7,8]; attention fusion [9]; Greeks via autograd [1].

### 2.4 Loss

L = λ_d L_data + λ_p L_physics + λ_o L_operator + λ_b L_boundary + λ_m L_market [7,10,22]. Adaptive λ via homoscedastic uncertainty [22], related to soft multi-objective balancing [18,19].

### 2.5 Training

Synthetic: analytic BS [10], fractional FDM [12], Merton MC [13,2]. Real: Yahoo SPY/QQQ/IWM chains [1]. Optimizer: Adam [23]; curriculum; DDP/cloud [3,24].

---

## 3. Results

### 3.1 Synthetic Black–Scholes

| Method | Config | Test rel-L₂ |
|--------|--------|-------------|
| CF-HPINO | research_medium (2 ep.) | 0.584 |
| CF-HPINO | smoke (3 ep.) | 0.68 |
| PINN [7] | smoke | 0.99 |
| Pure FNO [4] | smoke | 0.99 |
| Classical BS [10] | analytic | ≈ 0 |

### 3.2 Real SPY (preliminary)

Best val rel-L₂: **4.29**; test rel-L₂: **4.64**; quote RMSE: **$379** [1,16]. Not desk-level; GPU training recommended.

### 3.3 Cloud

DDP and export scripts [3,24]; amortized inference after training [4,6].

---

## 4. Discussion

CF-HPINO combines soft-computing fusion [18,19] with PDE constraints [7,8]. Versus PINNs [7,14,8]: adds operator generalization [4,5]. Versus FNOs [4,6]: explicit physics and boundaries. Limits: European calls; no Heston [16]; grid market quotes; large real-data error.

---

## 5. Conclusion

CF-HPINO unifies physics-informed and operator learning for option surfaces [10,13,12] with open code [24]. Future work: GPU training, Heston [15], Greeks validation [1], historical data.

---

## Acknowledgements (optional)

[Funding / supervisors]

---

## References

1. Hull, J. C. (2021). *Options, Futures, and Other Derivatives*, 11th ed. Pearson.  
2. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.  
3. Toke, I. M., et al. (2017). High-performance computing in finance. In *High-Performance Computing in Finance*. CRC Press.  
4. Li, Z., et al. (2021). Fourier neural operator for parametric PDEs. *ICLR*.  
5. Lu, L., Jin, P., Karniadakis, G. E. (2021). DeepONet. *Nature Machine Intelligence*, 3(3), 218–229.  
6. Kovachki, N., et al. (2023). Neural operators: A survey. *arXiv:2302.08127*.  
7. Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.*, 378, 686–707.  
8. Karniadakis, G. E., et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.  
9. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.  
10. Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities. *J. Polit. Econ.*, 81(3), 637–654.  
11. Caputo, M. (1967). Linear models of dissipation. *Geophys. J. Int.*, 13(5), 529–539.  
12. Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.  
13. Merton, R. C. (1976). Option pricing with discontinuous returns. *J. Financ. Econ.*, 3(1-2), 125–144.  
14. Sirignano, J., Spiliopoulos, K. (2018). DGM for PDEs. *J. Comput. Phys.*, 375, 1339–1364.  
15. Heston, S. L. (1993). Closed-form solution for stochastic volatility options. *Rev. Financ. Stud.*, 6(2), 327–343.  
16. Hutchinson, J. M., Lo, A. W., Poggio, T. (1994). Neural networks for option pricing. *J. Financial Engineering*, 1(4), 375–393.  
17. Hornik, K. (1991). Approximation capabilities of MLPs. *Neural Networks*, 4(2), 251–257.  
18. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.  
19. Jang, J.-S. R. (1993). ANFIS. *IEEE Trans. Syst. Man Cybern.*, 23(3), 665–685.  
20. Han, J., Jentzen, A., E, W. (2018). Solving high-dimensional PDEs with deep learning. *PNAS*, 115(34), 8505–8510.  
21. Becker, C., Cheridito, P., Jentzen, A. (2019). Deep learning for nonlinear parabolic PDEs and BSDEs. *Ann. Math. Artif. Intell.*, 84, 1–26.  
22. Kendall, A., Gal, Y., Cipolla, R. (2018). Multi-task uncertainty weighting. *CVPR*, 7482–7491.  
23. Kingma, D. P., Ba, J. (2015). Adam. *ICLR*.  
24. CF-HPINO Authors (2026). *Physics-informed neural operator pricing*. GitHub. https://github.com/Haseebcodejourney/physics-informed-neural-operator-pricing

*Use Zotero or Word cross-references when inserting into the ICSCCW template; numbering will auto-update.*

---

## Appendix

```bash
python scripts/train.py --config configs/smoke_test.yaml
python scripts/train_market.py --csv data/raw/spy_options_full.csv
```
