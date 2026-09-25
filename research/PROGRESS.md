# Research progress

## Workflows and skills used

The brief asks for installed skills per task. The skills installed in this environment are:
- `dataviz`, `code-review`, `simplify`, `artifact-*`, `run`, `security-review`, `init`;
- anthropic-skills: `creativity`, `learn`, `docs`, `pdf`, `docx`, `pptx`, `xlsx`, `canvas-design`, `skill-creator`, `mcp-builder`, and a few unrelated.

**No literature-review, statistics or scientific-writing skill is installed.** Those tasks followed the documented workflows in this directory instead.

| task | skill / workflow | outcome |
|---|---|---|
| environment and artifact inventory | direct inspection | `ENVIRONMENT.md`: no Da-TACOS features or checkpoints in the cloud; Zenodo, arXiv and MIREX blocked |
| literature | none installed; WebSearch (WebFetch blocked for arXiv/S2/MIREX) plus the repository's Crossref-verified notes | `literature_review.md`, `literature_matrix.csv`, with a verification tag per claim |
| experiment design | registry-first (`experiments/registry.yaml`) | 8 cloud experiments done; 3 local plus 3 derived analyses registered before any local result |
| statistics | none installed; the repository's work-level bootstrap (D-009) extended in `scripts/stats.py` | `statistics.md` |
| figures | **`dataviz` skill**: palette validated with its script (first palette FAILED CVD and chroma checks and was replaced by the validated slots 1–5); mark specs; one axis per panel; direct labels plus tables for the contrast WARN | `figures/fig2–fig7` |
| code review | **`code-review` skill** (medium) on research/scripts + tests | 4 findings, all fixed: alignment-cache key now covers alignment settings and checkpoint, and sanity runs use a separate cache; oracle index float rounding; K-sweep figure label read from the result; F1 summary guarded when no global variant |
| code quality / testing | synthetic end-to-end runs; regression tests pinning the shortcut to `HybridRun`; negative test of the validator | `tests/test_research_lib.py` (23 tests); full suite 142 passed |
| scientific writing | none installed; claims only from result files | `research_question.md`, audits, `paper/` draft |

## Cloud work (2026-09-24/25)

- X0–X7 executed on committed files (ARTIFACT-*). Key results are in `research_question.md` → "What is already established".
- Local runners implemented and tested on the synthetic fixture:
  - A1 (`run_shortlist_sweep.py`) reproduces the official pipeline exactly: \|Δ\| = 0.
  - F1 (`run_stage1_variants.py`) and H1 (`run_alignment_ablation.py`) run end to end.
  - H1's profile-cosine variant equals the frozen rerank-only result exactly.
- Validator, analysis of local outputs (B1, D1, E1, F1s, H1s), figure and table generators built. Each skips cleanly and lists what is missing.
- Documentation inconsistencies found and recorded in `baseline_audit.md` (five items). The README's "still climbing at epoch 150" was corrected.
- Corrections made while writing:
  - the hub-transition share was restated per query, not per success;
  - the embedding/alignment crossover claim was downgraded from "K ≥ 50" to "tie at 50, embedding better from 100", with a paired test added.

## Local runs

*(One line per experiment after running `LOCAL_EXPERIMENTS.md`: ID · completed / partial / failed · wall-clock · deviations.)*

- A1 · not yet run
- F1 · not yet run
- H1 · not yet run

## Current state

**PROMISING BUT NEEDS ONE MORE RESEARCH CYCLE** (the local tier-1 run, about 3 h).

- The analysis contribution (Option C) is supported by executed ARTIFACT analyses.
- The intervention questions (adaptive K, E1; cheap Stage-1 improvements, F1) and the full K sweep (A1) are designed, implemented, tested and registered, but not executed.
- The paper in `paper/` is a draft built only on established results. Its A1/E1/F1 sections are marked pending.
