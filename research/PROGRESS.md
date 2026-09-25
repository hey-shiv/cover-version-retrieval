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
| code quality / testing | synthetic end-to-end runs; regression tests pinning the shortcut to `HybridRun`; negative test of the validator; cross-experiment equality checks | `tests/test_research_lib.py`; see the test count in the final report |
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

- A1 · completed · ~80 min (alignment 4708 s, 6.5M pairs, 0.72 ms/pair; probe cache reused) · reproduction of benchmark_long384.json: PASS · no deviations
- F1 · completed · ~20 min · all three checkpoints (full_long, full_60, base_1500) · no deviations
- H1 · completed · ~2 h 12 min · full rotation set including exhaustive_dtw · no deviations
- A2 · completed · ~82 min full run (~88 min including the 200-query sanity run, 200 rows as expected) · validate_results.py: 0 failed · no deviations
- Environment notes (no effect on results): the venv is uv-managed, so step 0 used `uv pip install -e ".[dev]"` instead of `pip`; pytest gave 142 passed; the step-2 sanity outputs went to a session scratch dir instead of `/tmp`, and both passed (A1 per_query.csv had 200 rows); `validate_results.py`: 43 ok, 0 failed.

## Cloud analysis of the local runs (2026-09-25)

Nothing was re-run in the cloud. Everything below reads the files pushed in `6cf9999`.

1. **Validation.** `validate_results.py`: 45 ok, 0 failed. This includes two cross-experiment checks added in this round: F1's reference Stage 1 equals A1's, and H1's locked variant equals A1's rerank-only K = 30, both exactly.
2. **Reproduction.** A1 reproduces the frozen run-4 file exactly (|Δ| = 0 for four MAPs and shortlist recall). All three runs record a dirty tree at `e6c9c9e` in `env.json`; this is disclosed in the paper and in `reviewer_audit.md`.
3. **Analyses.** `analyze_local.py` wrote B1, D1, E1, C1 (post-hoc) and F1s/H1s. It took 40 s after vectorising the D1/E1 bootstrap; the first version timed out.
4. **Figures.** fig8–fig11 were added (K sweep, failure classes by K, adaptive vs fixed, Stage-1 variants) and inspected visually. The fig8 reference label was moved off the curves.
5. **Tables and macros.** t8–t12 were added, plus `paper/generated/numbers_local.tex`. The LaTeX tables now escape Δ, ·, − and scale wide tables to the text width.
6. **Literature (round 2).** One new directly relevant paper was found and added to the review, the matrix and the bib: Jacob et al. 2024, rerankers degrade with depth [S].
7. **Audits.** Round 2 of `novelty_audit.md` and `reviewer_audit.md` is done. `research_question.md` now answers RQ1–RQ8 and gives the revised thesis.
8. **Paper.** `paper/main.tex` was rewritten around the supported findings. Only A2 is still `\pending`.
9. **Corrections made while writing.**
   - A discussion sentence ("predicted-hard queries remain uncovered at K = 100 or 200") was checked and found true at 100 but not at 200. It was not in any result file, so it was rewritten as an untested explanation.
   - The E1 budget overshoot is stated as < 0.4% (max 0.36%), not 0.3%.
   - `analyze_local.py` wrote absolute machine paths into `sources`; they are now repository-relative, and the outputs were regenerated. Seeds are fixed, so the numbers are unchanged.
10. **Next experiment (A2).** A2 is registered, with commands in `LOCAL_EXPERIMENTS.md`. On the synthetic fixture, its fused Stage 1 ranks identically to F1's `win_fuse` (48/48 queries).

## Cloud analysis of A2 (2026-09-25)

Nothing was re-run in the cloud; this reads the files pushed in `ca67dea`.
- **Validation:** 71 ok, 0 failed. A new check confirms that A2's Stage-1 coverage equals F1's `win_fuse` exactly at every K.
- **Disclosure:** `env.json` records a dirty tree at `e3ff315`, although the run log says "no deviations". This is disclosed in the paper and in the reviewer audit.
- **Result (A2s):** the registered criterion is **met**. Hub-corrected ΔMAP versus A1 is:
  - +0.017 [+0.015, +0.019] at K = 30 (0.122 → 0.139);
  - +0.014 [+0.013, +0.016] at K = 100;
  - above zero at every K, shrinking to +0.010 at K = 500.

  Hit@1 is +3.0 points at K = 30. With fusion, K = 20 matches the MAP of the global Stage 1 at K = 30.
- **Outputs:** fig12, table t13 and `ATwo*` macros were added. The paper, audits (round 3), RQ5, README and website were updated.
- **Housekeeping:** re-running the analyses and figures only changed timestamps in unrelated `runtime.json` and PDF files; those changes were reverted.

## Current state

**PAPER READY WITH MINOR ADDITIONS**, as an empirical analysis paper **with one registered positive intervention** (global + window fusion in Stage 1). This is not a method paper.

What is supported, by executed and validated experiments:
- the K sweep: MAP 0.122 → 0.212 from K = 30 to 500, not saturated;
- the bottleneck moving from coverage to reranking around K = 100–200;
- hub correction's value growing with K;
- the Hit@1 decomposition and attribution;
- a registered negative result for adaptive K;
- a registered positive end-to-end result for Stage-1 fusion, at every K;
- development-protocol blindness, with three cases.

Minor additions before submission:
1. Read every [S] reference in full and fix the tags.
2. Compile the LaTeX on a machine that has it; the cloud has none. Proofread the generated tables.

The main threat to the paper is unchanged: the system is weak in absolute terms (best measured 0.222 MAP, with fusion at K = 500, vs Qmax 0.333). Whether the findings hold for a strong system is untested.
