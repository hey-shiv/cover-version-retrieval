# Improvement experiments

Three experiments run after the first benchmark evaluation exposed its two failure modes (hubness and Stage-1 recall). Every setting is still chosen on calibration works (alpha, lambda, resolution) or validation works (the encoder); no development or benchmark result selects anything.

| # | Experiment | Targets | Decision |
|---|---|---|---|
| 1 | Hubness correction of the alignment score | false positives from tonally static tracks | D-018 |
| 2 | 384-frame alignment for the reranker | the 96-frame pilot resolution | D-019 |
| 3 | Encoder trained on all Cover Analysis works | Stage-1 shortlist recall (0.021) | D-020 |

Commands:

```bash
python scripts/run_hubness_correction.py --config configs/hybrid_dev.yaml
python scripts/run_hubness_correction.py --config configs/hybrid_dev.yaml --set features.n_frames=384 --tag n384
python scripts/run_hybrid_retrieval.py   --config configs/hybrid_dev.yaml --set features.n_frames=384 --tag n384
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

<!-- EXPERIMENT-3 -->

## Final benchmark evaluation of the improved system

<!-- FINAL-BENCHMARK -->
