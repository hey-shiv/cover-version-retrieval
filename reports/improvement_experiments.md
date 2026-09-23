# Improvement experiments

Three experiments run after the first benchmark evaluation exposed its two failure modes (hubness and Stage-1 recall). Every setting is still chosen on calibration works (alpha, lambda, resolution) or validation works (the encoder); no development or benchmark result selects anything.

| # | Experiment | Targets | Decision |
|---|---|---|---|
| 1 | Hubness correction of the alignment score | false positives from tonally static tracks | D-018 |
| 2 | 384-frame alignment for the reranker | the 96-frame pilot resolution | D-019 |
| 3 | Encoder trained on all Cover Analysis works | Stage-1 shortlist recall (0.021) | D-020 |

Commands:

```bash
# 1. hubness correction (lambda/method calibrated on calibration works)
python scripts/run_hubness_correction.py --config configs/hybrid_dev.yaml
python scripts/run_hubness_correction.py --config configs/hybrid_dev.yaml --set features.n_frames=384 --tag n384
# 2. 384-frame reranking
python scripts/run_hybrid_retrieval.py   --config configs/hybrid_dev.yaml --set features.n_frames=384 --tag n384
# 3. encoder trained on every usable work
python scripts/build_manifests.py        --config configs/full_train.yaml
python scripts/train_encoder.py          --config configs/full_train.yaml
python scripts/run_hybrid_retrieval.py   --config configs/full_train.yaml --tag full96
python scripts/run_hybrid_retrieval.py   --config configs/full_train.yaml --set features.n_frames=384 --tag full384
# final benchmark runs
python scripts/run_benchmark.py --config configs/full_train.yaml --tag full96 --with-alignment-only \
    --hub-correction reports/results/hubness_correction.json
python scripts/run_benchmark.py --config configs/full_train.yaml --tag full384 \
    --set features.n_frames=384 --hub-correction reports/results/hubness_correction_n384.json
```

---

## 1. Hubness correction

**Method (D-018).** `corrected[q, c] = score[q, c] - lam * reference[c]`. The reference is each candidate's general score level against a **fixed probe set of 200 training-split tracks** — inductive, label-free, and identical in meaning at any protocol size. It costs 200 x C alignments (3M pairs on the benchmark) rather than the C x C matrix (225M). `lam` and the reference method (`mean`, `topk` with k = 10 or 50) are grid-searched on calibration works only, over lambda in [0, 1.5].

**Selected on calibration:** `mean` reference, `lam = 0.5` (96 frames); `mean`, `lam = 0.6` (384 frames).

| protocol | resolution | uncorrected MAP | corrected MAP |
|---|---|---|---|
| calibration (selection) | 96 | 0.206 | **0.379** |
| calibration (selection) | 384 | 0.304 | **0.407** |
| development (locked) | 96 | 0.248 | **0.289** |
| development (locked) | 384 | 0.384 | 0.355 |

Development, 96 frames: ΔAP +0.041 [−0.045, +0.131]; Hit@10 0.45 → 0.55; mean first rank 31.3 → 25.5. Development, 384 frames: ΔAP −0.030 [−0.158, +0.106], but median first rank improves 35.5 → 10.0 and Hit@10 0.45 → 0.55.

**Reading.** The correction clearly attacks the intended mechanism: the probe reference correlates with tonal dispersion at Spearman −0.81 (96 frames) and −0.90 (384 frames), i.e. it independently rediscovers "harmonically static tracks are hubs". At 96 frames it helps on both protocols. At 384 frames the two fixes overlap — higher resolution already makes scores discriminative enough that penalising popularity starts costing precision, and calibration and development disagree about the sign. With 20 queries neither development interval excludes zero, so the honest summary is: **the mechanism is real, the size of the benefit at high resolution is unresolved**, and the benchmark is the arbiter.

## 2. 384-frame alignment for the reranker

**Method (D-019).** The base default stays 96 frames. The variant preprocesses to 384 frames and is affordable because only the K = 30 shortlist is aligned: 16x more work per pair on 0.2% of the pairs. Alignment-only retrieval at 384 frames over the whole benchmark would take about 42 h and is not run.

Development protocol, all settings locked (alpha recalibrated on calibration works at this resolution, giving alpha = 0.00):

| system | 96 frames | 384 frames |
|---|---|---|
| global embedding (unchanged) | 0.189 | 0.189 |
| classical alignment (all candidates) | 0.248 | **0.384** |
| hybrid, K = 30 | 0.262 | **0.362** |
| hybrid Hit@10 | 0.55 | **0.60** |

Bootstrap at 384 frames: hybrid − global **+0.173 [+0.044, +0.324]** — the first development comparison whose interval clears zero by a comfortable margin. Hybrid − classical is −0.022 [−0.133, +0.043], i.e. the shortlist hybrid matches full alignment while doing a quarter of the alignment work.

Shortlist sweep at 384 frames shows MAP still rising at K = 50 (0.394), consistent with Stage-1 recall (0.85 at K = 50) being the limit rather than the reranker.

**Reading.** Resolution, not the alignment algorithm, was the dominant limitation of the classical stage. This is the clearest positive result of the three experiments, and it is cheap precisely because it is confined to the shortlist — which is the architectural argument for the hybrid in the first place.

## 3. Encoder trained on all Cover Analysis works

**Method (D-020).** With the whole Cover Analysis subset downloaded, the training split was rebuilt with `splits.n_train: all` and the encoder retrained with identical hyperparameters. Role assignment walks the same seeded permutation, so calibration, query, distractor and validation works are byte-identical to the bandwidth-budget run, and the previous 1,500 training works are exactly positions 220–1720 of that order and a strict subset of the new 4,780 (verified).

| | bandwidth budget (D-010) | full data |
|---|---|---|
| training works / tracks | 1,500 / 3,000 | **4,780 / 9,560** |
| best validation MAP | 0.181 | **0.319** |
| best epoch | 50 / 60 | 58 / 60 |
| training time (8 CPU threads) | 37 min | 56 min |

Validation MAP improves **76%** from data alone. The training loss was still falling at epoch 60 (2.50), so the model is not yet converged; longer training is an obvious and untested further gain.

**Effect on the development protocol** (alpha recalibrated on calibration works for each resolution):

| system | old encoder, 96 | full encoder, 96 | full encoder, 384 |
|---|---|---|---|
| global embedding (Stage 1) | 0.189 | **0.394** | 0.394 |
| global + 12-rotation test-time max | 0.200 | 0.436 | 0.436 |
| classical alignment (all candidates) | 0.248 | 0.248 | 0.384 |
| hybrid, K = 30 | 0.262 | 0.353 | **0.423** |
| shortlist recall at K = 30 | 0.75 | 0.80 | 0.80 |

Two things change qualitatively. First, Stage 1 stops being the weakest system and becomes stronger than classical alignment. Second, the **96-frame reranker now hurts** (hybrid 0.353 below global 0.394): a weak alignment score drags down a good shortlist. Only at 384 frames does reranking add to the better Stage 1 (hybrid 0.423, ΔAP over classical +0.039 [+0.019, +0.063]).

The best development system overall is Stage 1 with 12-rotation test-time matching (0.436), which was evaluated as an ablation and not carried into the benchmark; see Limitations.

## Final benchmark evaluation of the improved system

Two additional benchmark evaluations, both with every setting fixed beforehand (encoder from validation works, alpha and lambda from calibration works). No benchmark number changed any setting; the first evaluation's results are kept and reported unchanged.

**Benchmark run 2** — `--tag full96`, full-data encoder, 96-frame alignment, cached alignment matrix reused, hubness correction lambda = 0.5.
**Benchmark run 3** — `--tag full384`, full-data encoder, 384-frame reranking, hubness correction lambda = 0.6. Alignment-only at 384 frames is not run (about 42 h).

| system | encoder | frames | hub corr. | MAP | MRR | Hit@1 | Hit@10 | median first rank |
|---|---|---|---|---|---|---|---|---|
| global embedding | 1,500 works | — | — | 0.010 | 0.052 | 0.025 | 0.096 | 185 |
| hybrid K=30 | 1,500 works | 96 | no | 0.018 | 0.124 | 0.102 | 0.154 | 185 |
| global embedding | **full** | — | — | 0.034 | 0.130 | 0.072 | 0.243 | 59 |
| hybrid K=30 | full | 96 | no | 0.058 | 0.247 | 0.205 | 0.318 | 59 |
| hybrid K=30 | full | 96 | yes | 0.062 | 0.271 | 0.234 | 0.330 | 59 |
| hybrid K=30 | full | 384 | no | 0.062 | 0.271 | 0.236 | 0.326 | 59 |
| **hybrid K=30** | **full** | **384** | **yes** | **0.068** | **0.301** | **0.272** | **0.343** | **59** |
| classical alignment (all pairs) | — | 96 | no | 0.084 | 0.282 | 0.244 | 0.352 | 77 |
| **classical alignment (all pairs)** | — | **96** | **yes** | **0.136** | **0.397** | **0.350** | **0.481** | **14** |

Work-level bootstrap vs. the full-data Stage 1 (1,000 cliques, 10,000 resamples): hybrid 96 +0.0233 [+0.0212, +0.0255]; hybrid 96 + hub +0.0278 [+0.0254, +0.0302]; hybrid 384 + hub +0.0334 [+0.0308, +0.0361]; classical +0.0492 [+0.0439, +0.0545]; **classical + hub +0.1013 [+0.0936, +0.1090]**. Every interval excludes zero.

### What the three experiments bought

| | first evaluation | best after experiments | change |
|---|---|---|---|
| Stage 1 (global) MAP | 0.010 | 0.034 | **3.4x** |
| hybrid MAP | 0.018 | 0.068 | **3.8x** |
| best system MAP | 0.084 (classical) | 0.136 (classical + hub) | **1.6x** |
| shortlist recall at K = 30 | 0.021 | 0.072 | **3.4x** |
| median rank of first correct cover | 77 | 14 | **5.5x better** |

**Hubness correction transfers to scale, and is the single biggest win.** Predicted from a dev-set correlation, calibrated on 10 works, it raises classical alignment by 62% relative (0.084 → 0.136) and cuts the median first-correct rank from 77 to 14. It also helps the hybrid at both resolutions. This is the clearest confirmation that the error analysis identified a real mechanism rather than noise.

**The full-data encoder tripled Stage 1** and, with it, the hybrid. Shortlist recall rose from 0.021 to 0.072 — a 3.4x gain that is still the binding constraint.

**The architectural conclusion is unchanged, and still negative.** The best hybrid (0.068) remains **half** as accurate as exhaustive corrected alignment (0.136). A K = 30 shortlist over 15,000 candidates simply discards too many covers, no matter how good the reranker. What the hybrid buys is cost: 38 ms per query of reranking at 384 frames versus 2.63 h of exhaustive alignment at 96 frames.

**Untested and probably the strongest system: classical alignment at 384 frames with hubness correction.** Development suggests resolution helps the classical stage substantially (0.248 → 0.384), and hubness correction helps at scale, but the combination over all 1.95 x 10^8 pairs needs about 42 h of CPU and was not run. Any claim about it would be extrapolation.

### Runtime (13,000 x 15,000)

| stage | 96 frames | 384 frames |
|---|---|---|
| embedding 15,000 tracks | 68 s | 68 s |
| exact cosine search | 0.09 s | 0.07 s |
| shortlist construction | 9.3 s | 9.4 s |
| alignment rerank (13,000 x 30) | 69 s (5.3 ms/query) | 494 s (38 ms/query) |
| hubness probe reference (200 x 15,000) | 148 s | 2,220 s |
| alignment-only (1.95 x 10^8 pairs) | 9,458 s (2.63 h) | about 42 h (not run) |

The probe reference is a one-off per catalogue, not per query.


---

## 4. Longer training (the encoder had not converged)

The 60-epoch full-data run still had a falling loss, so the same configuration was trained for 150 epochs (`encoder.run_name=encoder_full_long`), again selecting the checkpoint by validation MAP.

| | 60 epochs | 150 epochs |
|---|---|---|
| best validation MAP | 0.319 | **0.431** (epoch 138) |
| training loss at the end | 2.50 | 1.73 |
| training time (8 CPU threads) | 56 min | 139 min |

**Development protocol** (alpha recalibrated on calibration works, giving alpha = 0.10): global embedding **0.572**, classical alignment 0.384, hybrid 0.415, rerank-only 0.424, global + test-time rotations 0.566. On development, reranking now *hurts* a strong Stage 1 (0.572 → 0.415).

**Benchmark run 4** — `--tag long384`, long-trained encoder, 384-frame reranking, hubness correction lambda = 0.6:

| system | MAP | MRR | Hit@1 | Hit@10 | Recall@100 | median first rank |
|---|---|---|---|---|---|---|
| global embedding (Stage 1) | 0.076 | 0.297 | 0.234 | 0.373 | 0.234 | 23 |
| hybrid K=30 | 0.113 | 0.373 | 0.330 | 0.443 | 0.234 | 23 |
| **hybrid K=30 + hub correction** | **0.122** | **0.402** | **0.366** | **0.462** | 0.234 | 21.5 |

Bootstrap vs. Stage 1: hybrid +0.0368 [+0.0340, +0.0395]; hybrid + hub +0.0457 [+0.0426, +0.0487]. Shortlist recall at K = 30 rose to **0.136**.

**The development protocol and the benchmark disagree about reranking.** On 20 queries with one relevant item each, reranking a strong Stage 1 looked harmful (−0.157 MAP). On 13,000 queries with 12 relevant items each it clearly helps (+0.037, interval excluding zero). With 15,000 candidates, reordering the top 30 lifts precision at the head; with 119 candidates and a single target, the same operation mostly adds noise. This is the second time the small protocol misled — the first was the original hybrid-versus-classical ordering.

### Summary across all four benchmark evaluations

| | run 1 | run 2 (96) | run 3 (384) | run 4 (384, long) |
|---|---|---|---|---|
| encoder | 1,500 works, 60 ep | full, 60 ep | full, 60 ep | **full, 150 ep** |
| Stage-1 MAP | 0.010 | 0.034 | 0.034 | **0.076** |
| best hybrid MAP | 0.018 | 0.062 | 0.068 | **0.122** |
| shortlist recall (K = 30) | 0.021 | 0.072 | 0.072 | **0.136** |
| best system overall | 0.084 classical | **0.136 classical + hub** | — | 0.136 classical + hub |
| hybrid vs. best system | 4.6x worse | 2.2x worse | 2.0x worse | **1.11x worse** |

**Revised verdict.** The two-stage design is no longer clearly the wrong architecture. Every improvement to Stage 1 narrowed the gap — 4.6x, then 2.2x, then 1.11x — while the hybrid keeps its cost advantage (38 ms per query of reranking versus 2.63 h of exhaustive alignment). Extrapolating the trend would be unjustified, but the direction is consistent across four pre-registered evaluations and the mechanism (shortlist recall) is measured, not inferred. Exhaustive corrected alignment still wins on accuracy, and the untested classical 384 + hub configuration (about 42 h) would likely widen its lead again.