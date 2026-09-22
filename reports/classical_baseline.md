# Classical baseline: transposition-aware subsequence DTW

Command: `python scripts/run_classical_baseline.py --config configs/classical_baseline.yaml --resolution-sweep 48 96 192 384`
Raw outputs: `results/classical_dev.json`, `results/classical_dev_per_query.csv`, `results/classical_dev_rankings.csv` (full rankings with the chosen shift per pair), `results/classical_dev_table.md`.

## Protocol

* **Development protocol** (D-001, D-008): 20 queries (the lexicographically lowest PID of each query WID) ranked against 120 Cover Analysis candidates (query + distractor works), self excluded, so 119 ranked items per query and exactly **one** relevant item. **AP = RR by construction, so MAP = MRR.**
* Nothing was tuned on this protocol. All settings are the declared defaults in `configs/base.yaml`.
* Significance: paired, work-level bootstrap (10,000 resamples of the 20 query WIDs, D-009).

## Method (all written in this repo; D-003 – D-006)

1. HPCP `(T, 12)` → `(12, T)`, frame L2, area-average resampling to **96 frames**, frame L2.
2. Key: 12 cyclic rotations of the candidate, choosing the one with the highest cosine between time-averaged profiles.
3. Cosine-distance cost matrix (96 x 96).
4. Subsequence DTW with steps (1,1), (2,1), (1,2) and weights 1, 2, 1. Normalised cost = accumulated cost / weighted path length (always 96).
5. Score = 1 − normalised cost.

## Results (development protocol, real Da-TACOS data)

| system | MAP (= MRR) | Hit@1 | Hit@10 | Hit@100 | mean first rank | median first rank |
|---|---|---|---|---|---|---|
| profile only (best-rotation cosine of mean chroma, no temporal model) | 0.266 | 0.150 | 0.550 | 0.850 | 24.9 | 8.5 |
| subsequence DTW, **no key handling** | 0.303 | 0.200 | 0.450 | 0.900 | 33.5 | 13.0 |
| **baseline: profile-selected rotation + subsequence DTW** | **0.248** | 0.150 | 0.450 | 0.950 | 31.3 | 13.0 |
| DTW over all 12 rotations (keep min cost) | 0.248 | 0.150 | 0.450 | 0.950 | 30.1 | 13.0 |

For reference, a random ranking of 119 items with one relevant item has an expected MAP of about 0.045 (the mean of 1/r for r = 1…119).

Paired work-level bootstrap, ΔAP vs. the baseline (95% CI):

| comparison | ΔAP | 95% CI |
|---|---|---|
| profile only − baseline | +0.018 | [−0.108, +0.153] |
| no key handling − baseline | +0.055 | [−0.098, +0.215] |
| exhaustive rotation − baseline | +0.000 | [−0.000, +0.001] |

**No difference between these variants is statistically supported on 20 queries.** Every interval contains zero.

### Runtime (development protocol, 2,380 pairs; laptop CPU, PyTorch CPU backend, 8 threads)

| system | total | per pair |
|---|---|---|
| profile only | 0.5 ms | — |
| subsequence DTW, profile-selected rotation | 0.20 s | 0.084 ms |
| subsequence DTW, no rotation | 0.20 s | 0.086 ms |
| DTW over all 12 rotations | 2.3 s | 0.99 ms |

Batched DTW is exactly equal to the reference implementation (`tests/test_dtw.py`). At benchmark scale (1.95 x 10^8 pairs, batches of 4,096) the per-pair cost falls to about 0.03 ms (see `hybrid_results.md` §6).

## Analysis

**1. Key invariance neither helps nor hurts on this sample, and the rankings explain why.** For the 20 true partners, the rotation chosen by the profile was 0 (same key) in 10 cases. Across all 2,380 query–candidate pairs the chosen shifts are spread roughly uniformly over 0–11. Rotation gives *every* candidate 12 chances to match, so it raises non-cover scores (the noise floor) as much as it rescues transposed covers. With half the partners untransposed, the net effect here is zero. Exhaustive DTW over all 12 rotations picks the same result as the profile choice (ΔAP ≈ 0), so the profile heuristic is not the bottleneck.

**2. At 96 frames the alignment adds little over a global chroma profile.** Profile-only retrieval is as good as the aligned systems (ΔAP +0.018, CI spans 0). The cost-matrix plots (`figures/calibration_alignment_paths.png`) show why. Averaging about 190 raw frames into each of 96 columns leaves matrices dominated by horizontal and vertical *stripes* (frames that are tonally central or atypical for the whole track) rather than diagonal bands of matching harmonic sequences. Non-covers that share a tonal centre get costs close to those of true covers (calibration example: cover 0.124 vs. non-cover 0.109 for the same query).

**3. Resolution is the main limiting factor (analysis only, D-005).** The declared default was *not* changed on the basis of this sweep:

| n_frames | calibration MAP | dev MAP | dev mean first rank |
|---|---|---|---|
| 48 | 0.206 | 0.285 | 26.3 |
| 96 (default) | 0.206 | 0.248 | 31.3 |
| 192 | 0.290 | 0.293 | 33.0 |
| 384 | 0.304 | 0.384 | 34.2 |

The calibration protocol (20 queries against 320 tracks) and the dev protocol both trend upward from 96 to 384 frames. This agrees with the classical CSI literature, which aligns at much finer (often beat-synchronous) resolution. Cost grows about quadratically, so 384 frames is roughly 16x the DTW work, which is what makes alignment-only retrieval on the full benchmark expensive. Choosing the resolution on calibration data would be legitimate future work. It was not done here because the project brief fixes 96 as the default.

## Limitations

* 20 queries: the confidence intervals are about ±0.15 MAP. These numbers support qualitative statements only.
* Whole-query subsequence alignment (not Serrà's Qmax local alignment) penalises covers with added or removed sections.
* The development protocol has one relevant item per query. The benchmark (12 relevant items per query, 15,000 candidates) is much harder, and its numbers are not comparable to the ones above.
