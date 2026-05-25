#!/usr/bin/env python3
"""Build ICSCCW 2026 paper PDF with user reference list [1]-[15]."""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "ICSCCW2026_CF_HPINO.pdf"


class PaperPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _ascii(text: str) -> str:
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2248", "~")
        .replace("\u03b8", "theta")
    )


def section(pdf: PaperPDF, title: str, body: str) -> None:
    pdf.set_font("Times", "B", 11)
    pdf.multi_cell(0, 6, _ascii(title))
    pdf.ln(1)
    pdf.set_font("Times", "", 10)
    pdf.multi_cell(0, 5, _ascii(body))
    pdf.ln(3)


def main():
    pdf = PaperPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Times", "B", 13)
    pdf.multi_cell(
        0,
        7,
        "Cloud-Enabled Fractional Hybrid Physics-Informed Neural Operator\n"
        "for Financial Derivative Pricing",
        align="C",
    )
    pdf.ln(3)
    pdf.set_font("Times", "", 10)
    pdf.multi_cell(0, 5, "[Author One]*, [Author Two], [Author Three] | [University], [Country]", align="C")
    pdf.ln(4)

    pdf.set_font("Times", "B", 10)
    pdf.cell(0, 6, "Abstract", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "I", 10)
    pdf.multi_cell(
        0,
        5,
        _ascii(
            "We propose CF-HPINO, a cloud-enabled hybrid Fourier neural operator with "
            "physics-informed losses for Black-Scholes, fractional, and Merton option surfaces, "
            "trained on synthetic data and real SPY quotes. Compared to PINN [1] and PINN-FNO "
            "hybrids [4], CF-HPINO adds fractional/Merton physics, market supervision, and open "
            "code. Synthetic rel-L2 ~0.68 vs baselines ~0.99; SPY val rel-L2 4.29, quote RMSE $379."
        ),
    )
    pdf.ln(4)

    section(
        pdf,
        "1. Introduction",
        "Recent work applies physics-informed neural networks (PINNs) to option pricing: "
        "Dhiman and Hu [1] (arXiv:2312.06711); Bansal et al. [2] on PIDEs (Applied Soft Computing); "
        "Gatta et al. [3] on American options (ScienceDirect); Elbayed and Qadi el Idrissi [4] on "
        "hybrid PINNs-FNO; Chen et al. [5] on barrier options; Lee et al. [6] DeepONet bond pricing; "
        "Feng et al. [7] hybrid neural operators for SPDE pricing; Hainaut and Casas [8] Heston PINNs; "
        "illiquid jump markets [9]; fractional BS [10,14]; regime switching PIRL [11]; American "
        "jump-diffusion under data scarcity [12]; implied vol operators [13]; KAN finance networks [15]. "
        "CF-HPINO extends [4] with fractional/Merton losses, attention fusion, and Yahoo SPY data.",
    )

    section(
        pdf,
        "2. Methodology",
        "Architecture: FNO backbone + physics MLP + attention fusion (cf. [4,7,14]). "
        "Loss: data + PDE residual + boundary + market quote terms following [1,3,raissi]. "
        "Models: Black-Scholes, fractional Caputo BS [10], Merton PIDE [2,9,12]. "
        "Data: synthetic surfaces; SPY full chain (5,283 quotes, 33 expiries). "
        "Training: Adam, EMA, curriculum; splits by parameter vector or expiry [11].",
    )

    section(
        pdf,
        "3. Results",
        "Synthetic (held-out theta): CF-HPINO rel-L2 0.68 (3 epochs) vs PINN/FNO 0.99. "
        "Real SPY (40 epochs, CPU): val rel-L2 4.29; test rel-L2 4.64; quote RMSE $379 (~51% spot). "
        "Learnability confirmed but below desk accuracy in [8,11,13].",
    )

    section(
        pdf,
        "4. Discussion",
        "CF-HPINO is closest to hybrid PINN-FNO [4] and SPDE operators [7]. "
        "ScienceDirect anchors include [3,5,11,12,14]. Limits: European focus; CPU training; "
        "grid-based market quotes. Future: Heston [8], regime switching [11], GPU full physics.",
    )

    section(
        pdf,
        "5. Conclusion",
        "CF-HPINO delivers an open hybrid physics-informed neural operator aligned with "
        "2023-2026 PINN/operator finance literature [1-15], with preliminary SPY validation.",
    )

    refs = [
        "[1] Dhiman, A., & Hu, Y. (2023). Physics Informed Neural Network for Option Pricing. arXiv:2312.06711.",
        "[2] Bansal, S., Boro, P., & Srinivasan, N. (2025). PINNs for PIDEs in financial modeling. Applied Soft Computing (Elsevier).",
        "[3] Gatta, F., et al. (2023). American option pricing via PINNs. Eng. Anal. Bound. Elem. ScienceDirect: S0955799723000978.",
        "[4] Elbayed, Z., & Qadi el Idrissi, A. (2025). Hybrid PINNs-FNO for option pricing. Preprints: 202501.0629.",
        "[5] Chen, Y., et al. (2025). PINN for barrier options. ScienceDirect (J. Comput. Sci.).",
        "[6] Lee, S., et al. (2026). DeepONet surrogate for bond option pricing. AIMS Mathematics.",
        "[7] Feng, M., et al. (2025). Hybrid neural operator for SPDE option pricing. Informatica.",
        "[8] Hainaut, D., & Casas, A. (2024). Heston pricing with physics-inspired NNs. Annals of Finance.",
        "[9] (2025). PINNs for option pricing and hedging in illiquid jump markets. ACM MLPR Proc.",
        "[10] Nuugulu, S.M., et al. (2025). PINN for time-fractional Black-Scholes PDEs. Optim. Eng. (Springer).",
        "[11] Pande, N.K., et al. (2025). Regime switching via physics-informed residual learning. Expert Syst. Appl. ScienceDirect: S0957417425018469.",
        "[12] Sun, Q., et al. (2025). Jump-diffusion informed NNs for American options. Applied Soft Computing.",
        "[13] Wiedemann, R., et al. (2024). Operator Deep Smoothing for Implied Volatility.",
        "[14] Sharma, A., et al. (2026). Attention-based PINN for time-fractional Black-Scholes. ScienceDirect (Appl. Math. Comput.).",
        "[15] Liu, C.Z., et al. (2024). Kolmogorov-Arnold finance-informed NN in option pricing. Applied Sciences.",
        "[16] Raissi, M., et al. (2019). PINNs. J. Comput. Phys. ScienceDirect: S0021999118307125.",
        "[17] CF-HPINO (2026). github.com/Haseebcodejourney/physics-informed-neural-operator-pricing",
    ]

    pdf.add_page()
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 8, "References (Google Scholar & ScienceDirect)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Times", "", 8)
    for r in refs:
        pdf.multi_cell(0, 4, _ascii(r))
        pdf.ln(1)

    pdf.output(OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
