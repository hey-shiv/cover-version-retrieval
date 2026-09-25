# Result verification

Every number below was read from the named file by `research/scripts/make_paper_assets.py` (tables) or by hand in the audit, and cross-checked. Values are unrounded where rounding could change the reading.

## Frozen benchmark runs (13,000 queries × 15,000 candidates, 12 relevant per query)

| run | file | Stage 1 MAP | hybrid MAP | hybrid + hub MAP | exhaustive classical MAP | shortlist recall @30 |
|---|---|---|---|---|---|---|
| 1 | `benchmark.json` | 0.01008 | 0.01848 (α 0.05) | — | 0.08354 (96 fr.) | 0.02112 |
| 2 | `benchmark_full96.json` | 0.03438 | 0.05768 (α 0.15) | 0.06217 | 0.08354 → **0.13565** with hub correction (96 fr.) | 0.07171 |
| 3 | `benchmark_full384.json` | 0.03438 | 0.06241 (α 0.15) | 0.06780 | not run at 384 | 0.07171 |
| 4 | `benchmark_long384.json` | **0.07601** | **0.11278** (α 0.10) | **0.12166** | not run at 384 | **0.13584** |

Checks against the brief:
- Stage 1 goes 0.010 → 0.034 → 0.076 across encoders ✓. Run 4 is 0.0760 / 0.1128 / 0.1217, with shortlist recall 0.1358 ✓.
- Exhaustive corrected classical: 0.13565 ✓. It is a **96-frame** system, while the run-4 hybrid reranks at 384 frames. They are different computations and are never presented as a like-for-like pair.
- Hub correction on the benchmark:
  - classical: 0.0835 → 0.1357 ✓;
  - hybrid (run 4): 0.1128 → 0.1217 ✓.
- Rerank-only equals the hybrid with α = 0: run 4 gives 0.11159.

Paired work-level intervals are in each file's `bootstrap_vs_global`. For example, in run 4 the hub-corrected hybrid minus Stage 1 is +0.0457 [+0.0426, +0.0487].

## Runtime
- Exhaustive 96-frame alignment only: 9,458 s = 2.63 h ✓. It is **alignment only**: it excludes feature loading and the 146 s probe reference used by the hub-corrected variant. Run 2 reused run 1's cached 96-frame alignment matrix.
- 384-frame exhaustive: about 42 h is an **estimate** (16 × per-pair work), never measured ✓.
- K = 30 rerank: 69 s at 96 frames and 494 s at 384 frames ✓. The embedding is 68 s and exact search 0.09 s ✓.
- **Pair counts.** The full benchmark has 13,000 × 14,999 = 1.95 × 10⁸ pairs. K = 30 aligns 390,000 pairs, **0.20%** of them.
- **A1 (laptop, 384 frames).** Measured 0.724 ms per aligned pair for a K = 500 pass (6.5 × 10⁶ pairs, 4,708 s). This is a different machine from the frozen runs, so its timings are not mixed with theirs.

## Development protocol (20 queries × 119 candidates, 1 relevant)

| system | 96 frames | 384 frames | file |
|---|---|---|---|
| classical (profile-cosine rotation) | 0.2481 | 0.3840 | `hybrid_dev.json`, `hybrid_dev_n384.json` |
| hybrid, 1,500-work encoder | 0.2624 (α 0.05) | 0.3621 (α 0.00) | same |

The brief's 0.248 / 0.384 and 0.262 / 0.362 ✓. **These are development numbers.** No benchmark run executed 384-frame exhaustive alignment.

**Development reversals** (checked):
1. On development the run-1 hybrid (0.262) beat classical (0.248); on the benchmark classical is 4.5× the hybrid (0.0835 vs 0.0185).
2. With the 150-epoch encoder, development says reranking hurts: Stage 1 0.572 → hybrid 0.415 (`hybrid_dev_long384.json`). The benchmark shows it helps: +0.0368 [+0.0340, +0.0395].
3. Hub correction on development:
   - at 96 frames: +0.041 [−0.045, +0.131];
   - at 384 frames: −0.030 [−0.158, +0.106].

   Both are inconclusive, while the benchmark gains are large.
4. Key handling (H1, LOCAL-FULL):
   - development: "no key handling" scored 0.303 against 0.248 with the rotation, an interval including zero;
   - benchmark: removing the rotation costs −0.0386 AP [−0.0422, −0.0350].

Development shortlist recall at K = 30 is 0.75–0.90, against 0.021–0.136 on the benchmark.

## Training
- Validation MAP: 0.1810 (1,500 works, 60 epochs) → 0.3187 (4,780 works, 60 epochs) → 0.4306 (4,780 works, 150 epochs, best epoch 138) ✓.
- **Correction to the brief.** The curve is *not* "still drifting upward" at epoch 150:
  - the slope over epochs 121–150 is −0.00013 per epoch;
  - the learning rate is 0 at epoch 150 (cosine schedule);
  - the mean did rise from epochs 91–120 (0.409) to 121–150 (0.421).

  Supported wording: *the curve rose and then flattened as the learning-rate schedule decayed; neither convergence nor continued improvement is established.*

## Hubness (development pool, 120 candidates)
- Tonal dispersion vs top-10 false-positive count: Spearman −0.771 for classical and −0.574 for hybrid ✓ (`hubness_dev.json`). The worst hub (P_400551) appears in 11 of 20 top-10 lists, against 1.68 expected under uniform chance.
- Probe reference vs dispersion: −0.808 at 96 frames and −0.900 at 384 frames ✓ (`hubness_correction*.json`).

## Error cases (development, `error_cases.csv`)
- P_797406 → P_797408: Stage-1 rank 15 → hybrid rank 1 (classical 1) ✓.
- P_310762: partner rank 48 in global and hybrid, 11 in classical, not in the shortlist ✓.
- P_207372: global rank 5 → hybrid rank 22 ✓ (`error_cases.csv`, false-positive case).

## LOCAL-FULL results used by the paper (validated, `research/results/`)
- **A1**, at K = 30, reproduces run 4 exactly (|Δ| = 0 for four MAPs and the shortlist recall). Hub-corrected hybrid MAP:
  - 0.0847 at K = 5;
  - 0.1217 at K = 30;
  - 0.1599 at K = 100;
  - 0.2116 at K = 500.
- **B1.** Failure classes by K, full benchmark, exact.
- **A2** (registered). Hub-corrected ΔMAP versus A1:
  - +0.0171 [+0.0153, +0.0190] at K = 30;
  - +0.0145 [+0.0129, +0.0161] at K = 100;
  - above zero at every K.
- **C1** (post-hoc). The hybrid ties exhaustive corrected alignment on MAP at K = 50 and exceeds it from K = 100. The frame counts differ (384 vs 96) and so do the machines.
