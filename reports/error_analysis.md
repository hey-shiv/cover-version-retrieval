# Error analysis: development protocol, hybrid system (K = 30, alpha = 0.05)

Cases were selected **deterministically** by `evaluation/analysis.py::select_error_cases`, not by eye:

* **successes:** the three queries whose partner ranks highest (all at rank 1);
* **false positives:** the worst-ranked queries whose top result is wrong, excluding those already shown as false negatives;
* **false negatives:** the three queries whose partner ranks lowest (> 10).

Data: `results/error_cases.csv` / `results/hybrid_dev.json → error_cases`. Each case has a figure pairing the cost matrix and DTW path for *query vs. true partner* with the same for *query vs. top non-cover* (`figures/error_<case>_<PID>.png`).

"Dispersion" is tonal dispersion: the mean cosine distance of a track's 96 frames from its own mean chroma (dev-pool median **0.053**). Low dispersion means the track is harmonically static at this resolution. "Length ratio" is raw partner frames divided by raw query frames (an audio-derived diagnostic only; no metadata is used).

Across the 20 queries, the hybrid puts the partner at rank 1 for 3 queries and in the top 10 for 11. Stage 1 missed the partner entirely (outside K = 30) for 5 queries; reranking cannot recover those.

## Successes

| query (WID) | partner | rank hybrid / global / classical | shift | global cos | align score | top non-cover (rank 2) |
|---|---|---|---|---|---|---|
| P_797406 (W_203093) | P_797408 | **1** / 15 / 1 | 0 | 0.917 | 0.979 | P_31055: cos 0.940, align 0.964, shift 5 |
| P_36279 (W_36279) | P_576649 | **1** / 1 / 1 | 8 | 0.987 | 0.966 | P_29344: cos 0.971, align 0.966, shift 0 |
| P_495015 (W_68889) | P_572966 | **1** / 1 / 1 | 5 | 0.996 | 0.976 | P_144424: cos 0.985, align 0.969, shift 8 |

* **P_797406 is the clearest case of reranking fixing Stage 1.** The embedding put the partner at rank 15. Alignment gives it the best score in the shortlist (0.979), and the path in `figures/error_success_P_797406.png` is a clean diagonal band. The versions are almost the same length (ratio 0.98) and in the same key (shift 0). *Hypothesis:* a faithful cover with the same structure is exactly what subsequence DTW rewards and what a pooled embedding blurs.
* **P_36279 and P_495015** were already at rank 1 in all three systems. Both partners were found with a non-zero key shift (8 and 5 semitones), so the transposition handling did work there. For P_36279 the partner is about 1.9x longer in raw frames; whole-track resampling plus slope-constrained DTW absorbed the tempo and length difference.
* **Note:** in all three successes the runner-up non-cover is close behind (for example, align 0.966 vs. 0.966 for P_36279). The margins are small, which matches the compressed score range at 96 frames.

## False positives (wrong track at rank 1)

| query | partner rank hybrid / global / classical | partner in shortlist? | top non-cover | FP global cos / align | partner global cos / align | dispersion q / partner / FP |
|---|---|---|---|---|---|---|
| P_310762 | 48 / 48 / 11 | no | P_31055 (W_31054) | 0.939 / 0.950 | 0.837 / 0.945 | 0.047 / 0.031 / **0.020** |
| P_419983 | 46 / 46 / 15 | no | P_599130 (W_129713) | 0.907 / 0.957 | 0.823 / 0.946 | 0.045 / 0.062 / **0.031** |
| P_207372 | 22 / 5 / 25 | yes | P_495015 (W_68889) | 0.995 / 0.961 | 0.952 / 0.939 | 0.055 / 0.143 / **0.041** |

* **P_310762, P_419983: Stage-1 recall failures.** Classical alignment over the full pool ranks their partners 11 and 15, but the embedding scored the partners too low (cos 0.84 and 0.82) to reach the shortlist, so the hybrid inherits rank 46–48. *Hypothesis:* the encoder, trained on only 1,500 works, has not learned these particular arrangements; a larger K or a better encoder would fix it, a better reranker would not.
* **P_207372: the reranker hurt.** The partner was at global rank 5 and fell to 22. It is the most tonally *varied* track of all nine cases (dispersion 0.143, nearly 3x the median); the query is at 0.055. Its alignment score (0.939) is lower than those of several static candidates. *Hypothesis:* cosine-distance DTW at 96 frames penalises harmonically busy material, which has more frame-to-frame mismatch, and favours static material. This is the hubness mechanism seen from the other side.
* **Common thread:** all three top false positives have below-median dispersion (0.020, 0.031, 0.041). P_31055 is a recurring hub: it is also the runner-up for a success above and a top-10 false positive for 5 of 20 hybrid queries (10 of 20 classical).

## False negatives (partner ranked > 10)

| query | partner rank hybrid / global / classical | partner global cos / align | top non-cover (rank 1) | FP global cos / align | dispersion q / partner / FP |
|---|---|---|---|---|---|
| P_29344 | 75 / 75 / 102 | 0.726 / 0.882 | P_576649 (W_36279) | 0.992 / 0.985 | **0.011** / 0.042 / 0.024 |
| P_411001 | 70 / 70 / 91 | 0.768 / 0.896 | P_576649 (W_36279) | 0.939 / 0.957 | 0.040 / 0.055 / 0.024 |
| P_385477 | 57 / 57 / 43 | 0.726 / 0.922 | P_16488 (W_16487) | 0.891 / 0.940 | 0.073 / 0.067 / 0.050 |

* **P_29344: a static query meets a static hub.** This query is the least varied track in the pool (dispersion 0.011). Its cost matrix against the non-cover P_576649 is almost uniformly minimal (cost 0.015; `figures/error_false_negative_P_29344.png`), so any path is cheap. Against its true partner the cost is 0.118 with a visible but noisy path. *Hypothesis:* for near-static chroma, DTW has no temporal structure to exploit and degenerates into comparing global profiles, where a static hub in a nearby key wins. Both systems fail (global 75, classical 102).
* **P_411001:** both systems again rank the same hub, P_576649, first (global cos 0.939). The partner's alignment score (0.896) is low for a cover. *Hypothesis:* a substantial arrangement or structure change that whole-query subsequence DTW cannot model. Local alignment (Qmax) or a feature with melody content would be the test.
* **P_385477:** partner at rank 57 (global) and 43 (classical). The partner is 1.37x longer in raw frames and dispersion is normal. *Hypothesis:* added sections (an extended intro/outro or repeats) force the whole-query alignment through unrelated material. Matching a shared section instead is what local alignment is designed for.

## Cross-cutting findings

1. **Hubness dominates the false positives.** Across all 120 candidates, tonal dispersion and top-10 false-positive count correlate at Spearman ρ = −0.77 (classical) and −0.57 (hybrid); see `results/hubness_dev.json`. All 9 top false positives above have below-median dispersion.
2. **Stage-1 recall is the hybrid's other limit.** 5 of 20 partners are outside the K = 30 shortlist. Two of those (P_310762, P_419983) are ones classical alignment alone would have ranked 11 and 15.
3. **Key handling behaves as designed.** Correct non-zero shifts (8, 5) were found for successful matches. Rotation also gives static non-covers 12 chances to match, which feeds the hubs.
4. **The 96-frame resolution compresses scores.** Alignment scores for covers and hubs overlap in 0.94–0.98, so small score noise reorders rankings (see the resolution sweep in `classical_baseline.md`).

## What should be tested next (none of it was done here, to avoid tuning on dev)

* Hubness correction of the alignment score (per-candidate mean centring, or CSLS), calibrated on calibration works.
* A tonal-dispersion prior or penalty, and whether it survives on the benchmark.
* Local (Qmax-style) alignment versus subsequence DTW for structure-changed covers (P_411001, P_385477).
* A larger K or a better Stage-1 encoder (full Cover Analysis training) for the recall failures.
