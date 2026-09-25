# Reproducing research/

## Source of truth

```
source code + configs            src/, scripts/, research/scripts/, configs/, experiments/registry.yaml
        ↓
experiment result files          reports/results/ (frozen runs 1-4), research/results/ (X*, A1, F1, H1, B1, D1, E1, ...)
        ↓
generated tables and figures     research/tables/, paper/generated/, research/figures/
        ↓
paper                            paper/main.tex (inputs only generated tables and figures)
        ↓
website                          web/ (only after the science is final; built from the same files)
```

No number is typed into the paper or the website if a script can generate it.

## Environment

- Python 3.11 or newer (the cloud analysis used 3.11.15; the frozen runs used 3.12), `pip install -e ".[dev]"`.
- The analysis scripts need only numpy, matplotlib, PyYAML and Pillow. The local runners also need torch and the repository package.
- Seed `20260817` everywhere. Bootstrap: 10,000 work-level resamples. Cross-fitting: 5 work-disjoint folds.

## What runs where

| step | needs Da-TACOS? | command | time |
|---|---|---|---|
| artifact analyses X0–X7 | no | `python research/scripts/artifact_analysis.py` | ~30 s |
| figures | no | `python research/scripts/make_figures.py` | ~10 s |
| tables | no | `python research/scripts/make_tables.py` | ~2 s |
| tests | no | `pytest` (142 tests) | ~15 s |
| synthetic mechanics test of every runner | no | see below | ~1 min |
| A1 / F1 / H1 | **yes**, plus checkpoints | [`LOCAL_EXPERIMENTS.md`](LOCAL_EXPERIMENTS.md) | ~3 h tier 1 |
| validate local outputs | no | `python research/scripts/validate_results.py` | seconds |
| analyses of local outputs B1, D1, E1, C1 (post-hoc), F1s, H1s (A2s once A2 exists) | no | `python research/scripts/analyze_local.py` | ~40 s |
| A2 (next cycle) | **yes** | [`LOCAL_EXPERIMENTS.md`](LOCAL_EXPERIMENTS.md), "Next cycle: A2" | ~80 min |

## Synthetic mechanics test (no data needed)

This proves every runner and analysis works end to end. Its numbers mean nothing about music and are written under `runs/smoke/`, which is git-ignored.

```bash
PYTHON=python ./scripts/run_smoke.sh
R=runs/smoke/research
python research/scripts/local/run_shortlist_sweep.py --config configs/smoke.yaml --hub-correction runs/smoke/reports/results/hubness_correction.json \
  --n-probes 8 --ks 5 8 10 20 30 50 --frozen runs/smoke/reports/results/benchmark.json --out $R/A1_k_sweep_long384
python research/scripts/local/run_stage1_variants.py --config configs/smoke.yaml --checkpoint base=runs/smoke/runs/sweep_base/encoder_best.pt \
  --primary base --n-probes 8 --out $R/F1_stage1_variants
python research/scripts/local/run_alignment_ablation.py --config configs/smoke.yaml --checkpoint runs/smoke/runs/sweep_base/encoder_best.pt \
  --k 8 --frames 96 32 --out $R/H1_alignment_ablation
python research/scripts/validate_results.py --root $R --synthetic
python research/scripts/analyze_local.py --root $R --menus 5,10,30 5,20,50 --budgets 10 20 30
```

Expected results:
- A1 prints `reproduction ... PASS`: at the locked K, deriving every K from one K_max pass reproduces `run_benchmark.py` exactly.
- The validator reports `0 failed`.

## Data assumptions

- Da-TACOS benchmark and Cover Analysis HPCP (Zenodo 10.5281/zenodo.4717628, CC BY-NC-SA 4.0), laid out as in `DATASET_CARD.md`.
- Manifests are committed and rebuild byte-identically.
- Checkpoints are **not** distributed (licence). Retraining with `scripts/train_encoder.py` reproduces them bit for bit on CPU (README → Reproducibility).

## Limitations of this package

- The cloud session that wrote `research/` had no Da-TACOS data. Everything labelled LOCAL-FULL must be produced on a machine that has it.
- The ARTIFACT analyses need only committed files and are reproducible anywhere.
- Literature entries tagged [S] were verified by search results only; see `literature_review.md`.
