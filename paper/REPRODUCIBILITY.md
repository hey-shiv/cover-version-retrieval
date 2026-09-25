# Reproducing the paper

## Source-of-truth chain
```
configs/ + src/ + scripts/ + research/scripts/          code and settings
      ↓
reports/results/   (frozen runs 1–4, development, calibration, training)
research/results/  (X*, A1/A2/F1/H1, analyses B1/C1/D1/E1/F1s/H1s/A2s, P1/P2)
      ↓
paper/tables/*.tex + paper/tables/numbers*.tex    ← research/scripts/make_tables.py
paper/figures/*                                   ← research/scripts/make_paper_assets.py (+ make_figures.py)
      ↓
paper/main.tex → paper/main.pdf                   (no number typed by hand; checked by check_paper_consistency.py)
      ↓
web/  (companion site; built from the same result files by web/scripts/build_data.py)
```

## Environment
- Python 3.11 or newer (the frozen runs used 3.12; the cloud analyses 3.11). Install with `pip install -e ".[dev]"` (the author's machine uses `uv pip install -e ".[dev]"`).
- Analysis and paper assets need numpy, matplotlib, PyYAML and Pillow. The runners also need torch.
- LaTeX: TeX Live with `latexmk`, natbib, booktabs, caption, subcaption and mathptmx. In a fresh Ubuntu container: `apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk`.
- Seed `20260817` everywhere. Bootstrap: 10,000 resamples of query works. Cross-fitting: 5 work-disjoint folds.

## Data and checkpoints (not in Git)
- **Da-TACOS HPCP features and metadata** (CC BY-NC-SA 4.0; no audio). Download: `python scripts/download_datacos.py` (see `README.md` → Run it). Manifests are committed and rebuild byte-identically with `python scripts/build_manifests.py`.
- **Checkpoints** `runs/*/encoder_best.pt` are not redistributed, because derived weights inherit the non-commercial licence. Retrain with `scripts/train_encoder.py` and the committed configs. Training on CPU is bitwise-reproducible; the 1,500-work checkpoint reproduces at epoch 50 with validation MAP 0.18097.

## Commands
| step | command | needs data? | time |
|---|---|---|---|
| tests | `pytest` (142 tests) | no | about 1 min |
| frozen runs 1–4 | `scripts/run_benchmark.py` with the tags in `README.md` | yes | hours |
| local research runs A1/A2/F1/H1 | `research/LOCAL_EXPERIMENTS.md` | yes | A1 or A2 about 80 min; F1 20 min; H1 2 h |
| validate local outputs | `python research/scripts/validate_results.py` | no | seconds |
| artifact analyses X0–X7 | `python research/scripts/artifact_analysis.py` | no | about 30 s |
| analyses of local runs | `python research/scripts/analyze_local.py` | no | about 40 s |
| research figures | `python research/scripts/make_figures.py` | no | seconds |
| paper analyses P1/P2 and figures | `python research/scripts/make_paper_assets.py` | no | about 30 s |
| tables and number macros | `python research/scripts/make_tables.py` | no | seconds |
| consistency check | `python research/scripts/check_paper_consistency.py` (exit 1 on any failure) | no | about 1 min |
| paper | `cd paper && latexmk -pdf main.tex` | no | about 30 s |
| website data | `cd web && npm run data` | no | seconds |

## What was executed where
- **Author's laptop:** frozen runs 1–4, training, and the local runs A1, A2, F1 and H1.
- **Cloud session:** every analysis, figure, table, and the paper, from the committed files. Nothing in the cloud touched Da-TACOS features or checkpoints.
- **Provenance to disclose:** the frozen runs record `-dirty` commits, and the local runs record `dirty: True`. A1 reproduces run 4 exactly, and `validate_results.py` checks cross-run equalities (F1 = A1 at Stage 1, H1 = A1 rerank-only at K = 30, A2 = F1 win_fuse coverage at every K).
