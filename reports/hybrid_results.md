# Hybrid retrieval results

Commands:

```bash
python scripts/train_encoder.py        --config configs/hybrid_dev.yaml   # D-011, D-016
python scripts/run_hybrid_retrieval.py --config configs/hybrid_dev.yaml   # calibration + dev
python scripts/analyze_hubness.py      --config configs/hybrid_dev.yaml   # analysis only
python scripts/run_benchmark.py        --config configs/benchmark.yaml --with-alignment-only
```

Raw outputs: `results/hybrid_calibration.json` (locked alpha/K), `results/hybrid_dev.json`, `results/hybrid_dev_per_query.csv`, `results/hybrid_dev_rankings_top10.csv`, `results/encoder_sweep.csv`, `results/training_history_sweep_*.csv`, `results/hubness_dev.json`, `results/benchmark.json`, `results/benchmark_per_query.csv`.

## 1. Encoder (Stage 1)

TCN, 645,504 parameters, SupCon (τ = 0.1), trained on **1,500 training works (3,000 tracks)**, CPU, deterministic. The checkpoint was selected by validation MAP on **150 disjoint validation works** (each of 300 tracks queried against the other 299):

| run | change | best validation MAP | best epoch |
|---|---|---|---|
| **sweep_base (selected)** | none; early stopping off | **0.181** | 50 / 60 |
| sweep_b64 | 64 works per batch | 0.180 | 50 / 60 |
| sweep_in512 | 512 input frames | 0.168 | 58 / 60 |
| sweep_lr3e-3 | lr 3e-3 | 0.146 | 53 / 60 |
| first run (patience 12) | — | 0.134 | 12 (stopped at 24) |

The training loss falls slowly, from 4.09 to about 3.3 (chance is ln 63 ≈ 4.14). The encoder learns a real but weak signal from two-recording works summarised at 256 frames. See `figures/training_history_sweep_base.png`.

## 2. Calibration (locks alpha before any dev or benchmark query is scored)

Protocol: the 20 calibration tracks as queries (10 works), candidate pool = calibration + validation tracks (320), self excluded, one relevant item per query (D-012). K = 30 (declared default, not tuned). The alpha grid step is 0.05.

| alpha | 0.00 | **0.05** | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 | 0.90 | 1.00 (= global only) |
|---|---|---|---|---|---|---|---|---|---|
| calibration MAP | 0.304 | **0.337** | 0.337 | 0.324 | 0.322 | 0.286 | 0.182 | 0.116 | 0.101 |

The chosen **alpha = 0.05** (MAP 0.3371, just ahead of 0.3370 at alpha = 0.10) is saved to `results/hybrid_calibration.json`. On calibration, Stage 1 keeps the true partner in its top 30 for 60% of queries. Nearly all of the blend weight goes to the alignment score, so in practice the global embedding acts as a candidate filter and a small tie-breaker.

## 3. Development protocol (20 queries x 120 candidates, one relevant each, MAP = MRR)

All settings were locked before this run.

| system | MAP (= MRR) | Hit@1 | Hit@10 | Hit@100 | mean first rank | median first rank |
|---|---|---|---|---|---|---|
| global embedding (Stage 1 only) | 0.189 | 0.10 | 0.35 | 1.00 | 23.5 | 15.0 |
| global embedding + 12-rotation test-time max | 0.200 | 0.10 | 0.55 | 1.00 | 20.6 | 9.5 |
| classical alignment only (every candidate) | 0.248 | 0.15 | 0.45 | 0.95 | 31.3 | 13.0 |
| **hybrid, K = 30, alpha = 0.05** | **0.262** | 0.15 | **0.55** | **1.00** | **20.8** | **10.0** |
| rerank only, K = 30, alpha = 0 (ablation) | 0.265 | 0.15 | 0.55 | 1.00 | 20.6 | 9.0 |

Random ranking: expected MAP ≈ 0.045.

Paired work-level bootstrap (10,000 resamples of the 20 query works), ΔAP with 95% CI:

| comparison | ΔAP | 95% CI | supported? |
|---|---|---|---|
| hybrid − global | +0.073 | [−0.000, +0.178] | borderline: the interval touches 0 |
| rerank-only − global | +0.076 | [+0.003, +0.181] | yes, marginally |
| hybrid − classical alignment | +0.014 | [−0.009, +0.042] | no |
| classical alignment − global | +0.059 | [−0.017, +0.167] | no |

**Does reranking improve the Stage-1 shortlist? (ablation)** The Stage-1 shortlist at K = 30 contains the partner for 15 of the 20 queries (shortlist recall 0.75). Reranking those 30 candidates with alignment raises MAP from 0.189 to 0.262–0.265, and the rerank-only interval excludes zero. Against alignment over *all* candidates, the hybrid is at least as good (+0.014, CI spans 0) while aligning 30 instead of 119 candidates per query, which becomes a large compute saving at benchmark scale.

Shortlist-size sweep (analysis only; K stays 30):

| K | shortlist recall | hybrid MAP | rerank-only MAP | rerank time (20 queries) |
|---|---|---|---|---|
| 10 | 0.35 | 0.219 | 0.220 | 0.09 s |
| **30** | **0.75** | **0.262** | **0.265** | **0.11 s** |
| 50 | 0.85 | 0.254 | 0.257 | 0.13 s |
| 100 | 1.00 | 0.247 | 0.249 | 0.18 s |
| 119 (all) | 1.00 | 0.246 | 0.248 | 0.20 s |

MAP peaks at intermediate K. A larger shortlist lets more alignment "hubs" (§5) into the reranked head, and the global filter was removing some of them. Post-hoc, the dev MAP-versus-alpha curve (in `hybrid_dev.json`, *not used for selection*) peaks at 0.287 (alpha = 0.35). The locked alpha = 0.05 is therefore not dev-optimal, which is the expected price of honest calibration on 10 works.

### Runtime (development protocol; Apple-silicon laptop CPU, PyTorch CPU backend)

| stage | time | per unit |
|---|---|---|
| embedding (120 tracks, incl. pooling) | 0.54 s | 4.5 ms / track |
| global search (20 x 120 exact cosine) | < 0.1 ms | — |
| shortlist construction | 0.2 ms | — |
| alignment rerank (20 x 30 pairs) | 0.11 s | 0.18 ms / pair (small batches) |
| classical alignment only (20 x 119 pairs) | 0.20 s | 0.084 ms / pair |

## 4. What the development numbers do and do not show

* **Supported:** alignment reranking of an embedding shortlist beats embedding-only retrieval on these 20 queries (rerank-only, CI excludes zero). Stage 1 alone is the weakest system.
* **Not supported:** that the hybrid beats classical alignment over the full pool, or that any classical variant beats another. With 20 queries, differences under about 0.15 MAP cannot be resolved.
* The development protocol has one relevant item per query and 119 candidates. It cannot predict benchmark behaviour (12 relevant items, 14,999 candidates, 2,000 noise tracks), which is why the benchmark is evaluated separately and only once.

## 5. Hubness (analysis only; see `error_analysis.md`)

Tonally static tracks, whose frames all lie close to their own mean chroma, get uniformly low DTW cost against almost everything, so they become hubs. Across the 120 dev candidates, *tonal dispersion* versus the number of times a track appears as a top-10 false positive gives Spearman ρ = **−0.77** (p ≈ 7e-25) for classical alignment and **−0.57** (p ≈ 7e-12) for the hybrid. The worst classical hub, P_400551 (dispersion 0.009 vs. a pool median of 0.053), is a top-10 false positive for 11 of 20 queries; about 1.7 would be expected by chance. The global filter reduces but does not remove this. Hubness-aware score normalisation (for example, per-candidate centring or CSLS-style correction) is the most promising next step, and it must be calibrated on calibration works only.

## 6. Final benchmark evaluation

<!-- BENCHMARK-RESULTS -->
