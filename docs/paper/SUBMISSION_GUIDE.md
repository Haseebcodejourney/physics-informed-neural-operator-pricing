# ICSCCW 2026 — Submission checklist

Source: [ICSCCW 2026 Submission](https://icsccw2026.az/submission.html)

## Conference fit

**ICSCCW 2026** (Tbilisi, Georgia) welcomes soft computing, neuro-fuzzy, machine learning, and **AI applications in finance and economics**.

This CF-HPINO work aligns with:

- Machine learning / deep learning
- Soft computing and AI in **finance**
- Hybrid intelligent systems (physics-informed + neural operator)

## Mandatory format

| Requirement | Detail |
|-------------|--------|
| **Formats** | Submit **both** `.docx` (or LaTeX) **and** PDF |
| **Font** | Times New Roman |
| **Length** | **Maximum 8 pages** |
| **Structure** | Abstract → Introduction → Methodology → Results → Discussion → Conclusion → Acknowledgements (optional) → References → Appendices (optional) |
| **Template** | Use official **ICSCCW 2026 Template** from the conference site (insert manuscript for auto-formatting) |
| **Language** | High-quality English (American **or** British; do not mix) |
| **Originality** | Original work; Turnitin similarity **>20% likely rejected** |
| **Review papers** | Normally not accepted |
| **Authors** | Full affiliation (city, country, postal address); `*` on corresponding author; ORCID encouraged |
| **Figures** | ≥800 DPI; readable in PDF and Word; font ≥6 pt in figures |
| **Author limit** | Each author name on **at most two papers** |

## Files in this repo

| File | Purpose |
|------|---------|
| **`ICSCCW2026_CF_HPINO.pdf`** | **Ready PDF** (run `python build_paper_pdf.py`) |
| `ICSCCW2026_CF_HPINO.tex` | Full LaTeX paper with `\cite{}` + BibTeX |
| `ICSCCW2026_CF_HPINO.md` | Word-friendly draft |
| `references.bib` | 22 refs (10+ ScienceDirect URLs) |
| `REFERENCES_SCIENCEDIRECT.md` | Clickable ScienceDirect + Scholar links |
| `build_paper_pdf.py` | Generate PDF without LaTeX install |

## References

The paper cites **20+ sources** via `references.bib` and `\cite{...}` in the LaTeX file. Bibliography style: `plainnat` (numbered, sorted by first citation). Key clusters: option pricing theory [Hull, Black–Scholes, Merton, Heston], PINNs [Raissi, Karniadakis], neural operators [Li FNO, DeepONet, survey], soft computing [Zadeh, ANFIS], optimization [Adam, Kendall uncertainty].

## Build PDF (LaTeX)

```bash
cd docs/paper
pdflatex ICSCCW2026_CF_HPINO.tex
bibtex ICSCCW2026_CF_HPINO
pdflatex ICSCCW2026_CF_HPINO.tex
pdflatex ICSCCW2026_CF_HPINO.tex
```

Then copy content into the official Word template if the conference requires `.docx` as primary.

## Before you submit

1. Replace `[Author Name]` and affiliations in the `.tex` / `.md`.
2. Add ORCID and corresponding author email.
3. Run Turnitin or institutional checker; keep similarity **below 20%**.
4. Export figures at 800 DPI (architecture diagram, error plots from `results/`).
5. Confirm page count ≤8 after inserting into the template.
6. Register and upload at the conference portal linked from [icsccw2026.az](https://icsccw2026.az/submission.html).

## Editing service (optional)

Conference suggests: https://secure.authorservices.springernature.com/
