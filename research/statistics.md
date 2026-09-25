# Statistical methodology

The code for everything here is in `research/scripts/stats.py`, and its behaviour is tested in `tests/test_research_lib.py`.

## Unit of resampling

Benchmark queries are not independent: the 13 recordings of a work are all queries, and each is relevant to the other 12. Every interval therefore resamples **works** (the 1,000 query cliques), never individual queries. This is the same choice as the repository's `bootstrap_delta_ci` (D-009; Smucker, Allan & Carterette 2007).

## Estimators

| quantity | estimator | function |
|---|---|---|
| mean of a per-query value (MAP, Hit@k, coverage, class share) | work-level percentile bootstrap of the mean | `boot_mean` |
| paired difference b − a on the same queries | bootstrap of the mean per-query difference, resampling works | `boot_delta` (same estimator as `evaluation.analysis.bootstrap_delta_ci`) |
| conditional rate, e.g. P(rank 1 given a cover in the shortlist) | ratio of sums, works resampled jointly for numerator and denominator | `boot_ratio` |
| discrimination of a difficulty score | AUROC (Mann–Whitney, ties averaged); interval from 1,000 work-level resamples of the out-of-fold scores | `auroc` |

- Settings: 10,000 resamples (1,000 for AUROC), 95% percentile intervals, seed `20260817`.
- An interval "excludes zero" only if it lies entirely on one side of it (`excludes_zero` in the output).

## Reporting rules

1. Every comparison reports the point estimate, the 95% interval, and the paired Δ with its own interval. Two overlapping intervals are **not** evidence of no difference, and two separated intervals are not the test either; the paired Δ is.
2. "Improvement" requires the paired Δ interval to exclude zero **in the claimed direction**. Otherwise the text says "no detectable difference at this sample size".
3. The development protocol (20 queries) is reported separately and never pooled with the benchmark. Its intervals are about ±0.15 MAP, so it supports no ranking claims (D-022, X6).
4. Comparisons across runs are paired by query but may differ in several settings. Each comparison states what differs (`X5 → what`).
5. **No multiple-comparison correction is applied** to the X5 family. Every interval there excludes zero by a wide margin, so no conclusion depends on it. For D1's univariate AUROCs (12 signals) the values are descriptive only; the pre-specified test is the single cross-fitted multivariate AUROC.

## Leakage control for fitted components (D1, E1)

Anything fitted to benchmark labels is fitted and evaluated with **work-disjoint 5-fold cross-fitting**. Works are assigned to folds by a seeded permutation (`group_folds`). The difficulty model, and for E1 the tier thresholds, are fit on four folds and applied to the fifth. Every reported number is out-of-fold. The protocol was fixed in `experiments/registry.yaml` before any local result existed.

This differs from the repository's calibration protocol (10 works). That protocol is too small, and too unlike the benchmark (1 relevant item in a pool of 320), to fit a difficulty model whose inputs depend on pool size. The difference is disclosed in the paper.

## Structural facts used as checks, not assumptions

- A hybrid that reranks the first K Stage-1 candidates cannot change the first-cover rank of a query with no cover in its top K. The artifact analysis asserts this on all 13,000 queries of every run, and the validator checks it for every K in A1.
- Hit@1 = coverage(K) × P(rank 1 | covered). This is an identity, so its decomposition needs no model.
