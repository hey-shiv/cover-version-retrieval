# Baseline audit

The frozen baseline, traced from **source code, configs and result files**. Where README prose disagrees with those, the files win and the disagreement is listed at the end.

The reference configuration is **benchmark run 4** (`reports/results/benchmark_long384.json`).

## Pipeline, as implemented

| stage | what the code does | where |
|---|---|---|
| Input | Da-TACOS HPCP, `(T, 12)` float32, per-frame max-normalised to [0, 1]; validated for shape, labels, finiteness and minimum length | `data/datacos.py`, `features/hpcp.py` |
| Preprocessing | transpose to `(12, T)` → frame L2 → exact area-average resampling to N frames → frame L2; silent frames stay zero | `features/preprocessing.py` |
| Views | `classical`: 12 × `features.n_frames` (96 default, 384 in runs 3–4). `encoder`: 12 × 512 cache (`encoder.cache_frames`), area-pooled to 256 at input | `configs/base.yaml` |
| Encoder | 1×1 conv 12→128; 6 residual blocks (two dilated convs, kernel 3, dilation 2^b, BN, GELU, dropout 0.1, skip); concat(mean, max) over time → 256; Linear 256→128, GELU, Linear 128→128; L2 | `models/tcn_encoder.py` |
| Size | **645,504** parameters; receptive field **253** frames (of 256) | `training_summary_encoder_full_long.json` |
| Training | SupCon, τ = 0.1; batches of 32 works × 2 recordings; AdamW (lr 1e-3, wd 1e-4) with cosine annealing over the epoch budget; random crops ≥ 60% of the track; random pitch rotation; CPU, deterministic, 8 threads | `models/training.py`, `configs/base.yaml` |
| Run-4 encoder | 4,780 works / 9,560 recordings, 150 epochs, best validation MAP **0.4306** at epoch 138, 8,362 s | `training_summary_encoder_full_long.json` |
| Stage 1 | cosine = dot product of L2 embeddings; exact search over all candidates; query itself excluded; ties broken by candidate order (PID-sorted manifests) | `retrieval/index.py`, `evaluation/metrics.py::ranking_from_scores` |
| Shortlist | first **K = 30** Stage-1 candidates (declared, never tuned) | `retrieval.shortlist_k` |
| Key handling | 12 cyclic rotations of the candidate; choose the one maximising cosine of time-averaged profiles (`profile_cosine`) | `alignment/transposition.py` |
| Cross-similarity | cosine distance between every query frame and every rotated-candidate frame | `alignment/csm.py` |
| Alignment | subsequence DTW; steps (1,1), (2,1), (1,2), weights 1, 2, 1; free start and end on the candidate axis; normalised cost = accumulated cost / accumulated weight (= N); score = 1 − cost | `alignment/dtw.py` |
| Hubness correction | `corrected[q,c] = align[q,c] − λ·ref[c]`, ref = mean alignment score of c against **200 fixed training-split probe tracks**; method and λ chosen on calibration works | `retrieval/normalization.py`, `scripts/run_benchmark.py` |
| Hybrid score | within each query's shortlist: `α·z(global) + (1−α)·z(alignment)`, z = per-shortlist z-score; stable sort (ties keep Stage-1 order); positions > K keep Stage-1 order | `retrieval/hybrid.py` |
| Run-4 settings | α = **0.10** (calibration), λ = **0.6**, mean reference, 384 frames | `benchmark_long384.json → locked_from_calibration` |

## Protocols

| protocol | queries | candidates | relevant / query | used for |
|---|---|---|---|---|
| Calibration | 20 (10 works) | 320 (calibration + validation tracks) | 1 | α, λ, resolution (only) |
| Development | 20 | 120 → 119 ranked | 1 | plumbing checks; MAP = MRR |
| Benchmark | **13,000** (1,000 cliques × 13) | **15,000** (incl. 2,000 noise) → 14,999 ranked | **12** | final test; never used for selection |

- Splits are work-disjoint along a permutation seeded with `20260817`.
- The Cover Analysis and benchmark subsets share no works and no recordings (DATASET_CARD.md).

## Metrics and statistics

- AP uses every relevant item (12 per benchmark query). MAP and MRR are means over queries.
- Hit@k: at least one relevant item in the top k. Recall@k: the fraction of all relevant items in the top k.
- Shortlist recall: relevant items in the top K, divided by all relevant items (`HybridRun.shortlist_recall`).
- Significance: paired **work-level** percentile bootstrap (resampling the 1,000 query cliques), 10,000 resamples (`evaluation/analysis.py::bootstrap_delta_ci`; D-009).

## Frozen results used as the baseline (exact, from `benchmark_long384.json`)

| system | MAP | MRR | Hit@1 | Hit@10 | Recall@100 | median first rank |
|---|---|---|---|---|---|---|
| Stage 1 | 0.07601 | 0.23168 | 0.1550 | 0.3799 | 0.2343 | 26 |
| Hybrid K = 30, α = 0.10 | 0.11278 | 0.37258 | 0.3303 | 0.4428 | 0.2343 | 23 |
| Rerank only, α = 0 | 0.11159 | 0.36884 | 0.3260 | 0.4418 | 0.2343 | 23 |
| Hybrid + hub correction | 0.12166 | 0.40244 | 0.3658 | 0.4620 | 0.2343 | 21.5 |

- Shortlist recall 0.13584.
- Runtime (author's laptop CPU):
  - embedding 69.0 s;
  - exact search 0.093 s;
  - shortlist 9.5 s;
  - rerank 495.3 s (38.1 ms/query; about 1.27 ms per aligned pair at 384 frames);
  - hub reference 2,204 s (200 × 15,000 pairs, one-off).
- Best overall system: classical exhaustive alignment + hub correction, 96 frames, **run 2** (`benchmark_full96.json`). MAP 0.13572, MRR 0.39681, Hit@1 0.3499, median rank 14. It was measured once, in a different run from the best hybrid.

**Reproduction status.**
- **ARTIFACT-REPRODUCTION** (done, cloud; X0): every aggregate metric of all four runs is exactly the mean of its committed per-query file (max |Δ| = 0).
- **Full reproduction** (done, local, 2026-09-25; LOCAL-FULL): experiment A1 recomputed run 4 from features and checkpoint on the author's machine and reproduced MAP of all four systems and the shortlist recall at K = 30 exactly (|Δ| = 0; `research/results/A1_k_sweep_long384/metrics.json` → `reproduction`). The run was made from a working tree with uncommitted changes at `e6c9c9e` (`env.json`).

## Known inconsistencies in the repository's documentation

1. **Run 4, Stage-1 row.**
   - The README before PR #2 and `reports/improvement_experiments.md` give MRR 0.297, Hit@1 0.234, Hit@10 0.373, median 23.
   - The result file gives 0.232, 0.155, 0.380 and 26.
   - The README was corrected in PR #2; `improvement_experiments.md` still carries the old values.
2. **Test count.** The README said 112 tests; the suite had 119. It has 142 including this work's `tests/test_research_lib.py`.
3. **Ratios.** Headline ratios such as 4.6×, 7.6× and 6.8× were computed from rounded MAPs. Unrounded, they are 4.5×, 7.5× and 6.6×.
4. **"Still drifting upward at epoch 150"** (README, D-021). The validation-MAP slope over epochs 121–150 is −0.00013 per epoch, and the best epoch is 138. That is a plateau, not a rise (X7).
5. **Stale limitation.** The README before PR #2 said shortlist recall "improved from 0.021 to 0.072" when describing the final state; run 4 reached 0.136. Corrected in PR #2.
