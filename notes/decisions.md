# Decision log

Every design choice that can affect validity or compute. Format: date, decision, rationale, evidence/resource, rejected alternative, validity/compute impact.

---

## D-001 — Cover Analysis for development; Benchmark is test-only
- **Date:** 2026-09-22
- **Decision:** All training, validation, calibration and development evaluation use the Cover Analysis subset. The Benchmark subset is evaluated once, at the end, with every setting already locked.
- **Rationale:** Tuning on the benchmark would turn it into a validation set and invalidate the final numbers.
- **Evidence:** Da-TACOS paper §3 (subset purposes). Local check: the two subsets share 0 WIDs and 0 PIDs.
- **Rejected:** carving validation data out of the Benchmark.
- **Impact:** Keeps the final numbers clean. Cover Analysis has only 2 recordings per work, so development metrics are single-relevant-item metrics.

## D-002 — HPCP is the only audio representation
- **Date:** 2026-09-22
- **Decision:** Both stages consume only the Da-TACOS `hpcp` feature. CREMA, CENS, MFCC, key, madmom and tags are unused.
- **Rationale:** HPCP is a harmonically weighted pitch-class profile. It is robust to timbre and instrumentation, which is what cover versions change. One feature keeps comparisons controlled.
- **Evidence:** Serrà et al. 2008; FMP §3.1 (chroma features); Essentia CSI tutorial.
- **Rejected:** feature fusion (CREMA + HPCP), which is a later, separate experiment.
- **Impact:** Simple and leakage-free. Rhythm, melody and timbre cues are left on the table.

## D-003 — Per-frame L2 normalisation, before and after resampling
- **Date:** 2026-09-22
- **Decision:** Normalise each frame to unit L2 norm; silent frames stay exactly zero (no NaN). Re-normalise after temporal resampling.
- **Rationale:** Removes loudness differences between versions and makes the cosine cost well defined in [0, 1] for non-negative HPCP. Bin averaging shortens vectors, so a second normalisation is needed.
- **Evidence:** FMP §3.1.2 (feature normalisation); tests in `tests/test_hpcp.py`.
- **Rejected:** max-normalisation (Da-TACOS storage) as the working form, because it does not give a bounded cosine geometry.
- **Impact:** Negligible compute. Silent frames get cost 1 against everything.

## D-004 — Key invariance by 12 cyclic rotations, chosen from the features
- **Date:** 2026-09-22
- **Decision:** Test all 12 rotations of the candidate; pick the rotation that maximises the cosine between time-averaged chroma profiles (an "optimal transposition index"). Ablations: no rotation, and DTW for all 12 rotations. Convention: `rotate(X, k)[p] = X[(p - k) % 12]`.
- **Rationale:** Covers are often transposed. Profile selection costs 12 dot products instead of 12 alignments.
- **Evidence:** Serrà et al. 2008; `tests/test_transposition.py` recovers every one of the 12 shifts.
- **Rejected:** using the Da-TACOS `key_extractor` metadata, which would be annotation leakage; exhaustive DTW as the default (12x the cost).
- **Impact:** Rotation selection is nearly free. A wrong profile choice can hurt modulating songs (measured by the exhaustive-DTW ablation).

## D-005 — Uniform resampling to 96 frames (declared pilot constraint)
- **Date:** 2026-09-22
- **Decision:** Resample every full sequence to `n_frames = 96` with exact area averaging over equal-width bins (linear interpolation only when upsampling).
- **Rationale:** Bounds DTW cost at 96 x 96 per pair, which makes alignment-only retrieval over the full benchmark (1.95e8 pairs) feasible on a laptop CPU. Area averaging avoids the aliasing that point sampling causes when T is in the thousands.
- **Evidence:** A typical raw HPCP is about 10^4 frames. Batched DTW measured at about 0.03 ms/pair on 10 CPU threads for 96 x 96.
- **Rejected:** beat-synchronous features (would need madmom beats) and full-resolution DTW (infeasible at benchmark scale).
- **Impact:** Each frame averages roughly a hundred raw frames, so fine melodic and harmonic rhythm is blurred. This is a known accuracy limitation, reported rather than tuned away.

## D-006 — Subsequence DTW with slope-constrained, weighted steps
- **Date:** 2026-09-22
- **Decision:** Cost = cosine distance. Subsequence DTW (the query must be fully aligned, but may start and end anywhere in the candidate) with steps {(1,1), (2,1), (1,2)} and weights {1, 2, 1}. Normalised cost = accumulated cost / accumulated step weight (the weighted path length). Retrieval score = 1 − normalised cost.
- **Rationale:** Slopes are constrained to [1/2, 2], which forbids degenerate paths (for example, a whole query matched to one candidate frame, which classic {(1,0),(0,1),(1,1)} steps allow). With these weights, every admissible path has weighted length exactly N, so the dynamic program minimises the same quantity that is reported.
- **Evidence:** FMP §3.2 (DTW, step-size conditions) and subsequence DTW; `tests/test_dtw.py` (hand-calculated matrices, degenerate-path test, batched equals reference).
- **Rejected:** Serrà's Qmax local alignment (a different recurrence with binary similarity and gap penalties; kept as future work to avoid copying acoss) and Soft-DTW (not needed for a non-learned reranker).
- **Impact:** Asymmetric: score(q, c) ≠ score(c, q), which is fine for query-centric retrieval.

## D-007 — WID-level relevance and metric definitions
- **Date:** 2026-09-22
- **Decision:** A candidate is relevant iff it has the query's WID and is not the query itself. Report MAP, MRR, Recall@{1,10,100} (fraction of all relevant items in the top k), Hit@{1,10,100} (at least one relevant item in the top k), and the mean/median rank of the first relevant item. Ties are broken by PID order.
- **Rationale:** Matches the MIREX / Da-TACOS protocol. Recall@k and Hit@k differ once there are 12 relevant items per query (benchmark), so both are reported.
- **Evidence:** IR book ch. 8; MIREX CSI; `tests/test_metrics.py` (includes a multi-relevant AP example).
- **Note:** In the development and calibration pair protocols each query has exactly one relevant item, so AP = reciprocal rank and **MAP = MRR by construction**.

## D-008 — Seeded split order with lazy validity filtering
- **Date:** 2026-09-22
- **Decision:** Sort WIDs, permute with `default_rng(20260817)`, walk the permutation, skip WIDs that fail validation, and fill roles in the fixed order calibration → query → distractor → validation → train.
- **Rationale:** Deterministic, WID-disjoint, and equivalent to "filter, then take a seeded sample of the valid pool". It also means the data needed first can be fetched first.
- **Evidence:** `tests/test_splits.py` (determinism, disjointness, invalid/unavailable skipping, byte-identical manifest rebuilds).
- **Impact:** Manifests depend only on the seed, the sorted IDs and file validity.

## D-009 — Work-level paired bootstrap for significance
- **Date:** 2026-09-22
- **Decision:** 95% percentile CIs for ΔAP from 10,000 resamples of WIDs (all of a work's queries move together); paired across systems.
- **Rationale:** Recordings of the same work are not independent. Resampling tracks would overstate confidence.
- **Evidence:** Smucker, Allan & Carterette 2007.
- **Impact:** With only 20 development queries, intervals are wide. No development-set difference is claimed unless its interval excludes zero.

## D-010 — Data acquisition from Zenodo with selective member fetch; training budget of 1,500 works
- **Date:** 2026-09-22
- **Decision:** Download from Zenodo (the Google Drive IDs were dead). The link measured 0.3–0.45 MB/s, so instead of 17 GB of archives the project fetches individual H5 members by HTTP Range request in priority order: the dev roles, validation (150 works), train (1,500 works), then the whole Benchmark (all 15,000 files).
- **Rationale:** At the measured bandwidth the full Cover Analysis archive (7.1 GB) alone would take about 5–6 h. The benchmark must be complete, but training can use a subset.
- **Evidence:** Throughput logs in `data/external/fetch.log` (git-ignored); CRC-32 verification per member.
- **Rejected:** skipping the benchmark, or training on fewer than about 1,000 works.
- **Impact:** The encoder sees 3,000 of the 8,600 available Cover Analysis training tracks, so learned-retrieval numbers are a lower bound on what the full subset allows. `splits.n_train: all` plus a full download reproduces the full-data setting.

## D-011 — Encoder: compact TCN, SupCon, CPU-deterministic
- **Date:** 2026-09-22
- **Decision:** 6 residual dilated Conv1d blocks (128 channels, kernel 3; receptive field 253 frames at 256 input frames); mean+max pooling; 2-layer head to 128-d; L2-normalised. SupCon loss (τ = 0.1) on batches of 32 works x 2 recordings. Augmentation: random 60–100% time crop and random pitch rotation. AdamW (lr 1e-3, wd 1e-4) with cosine decay; checkpoint chosen by validation MAP; early stopping (patience 12). CPU with deterministic kernels and fixed threads.
- **Rationale:** Temporal convolutions are strong, cheap sequence baselines (Bai et al. 2018). SupCon uses every in-batch negative (Khosla et al. 2020). Pitch-rotation augmentation teaches key invariance without metadata.
- **Rejected:** Transformers and pretrained music models (premature at this data scale); a triplet loss as the default (it is implemented as `training.loss: triplet`).
- **Impact:** 645,504 parameters; about 0.36 s per 64-track step on 8 CPU threads (about 17 s per epoch on 3,000 tracks).

## D-012 — Hybrid score: per-query z-scored blend, alpha from calibration WIDs only
- **Date:** 2026-09-22
- **Decision:** Stage 1: top K = 30 by embedding cosine. Stage 2: `alpha * z(global) + (1 - alpha) * z(alignment)`, where z standardises within the query's shortlist. Alpha is grid-searched (step 0.05) on the calibration protocol, then locked in `reports/results/hybrid_calibration.json`. The final ranking is the reranked shortlist followed by the untouched Stage-1 remainder.
- **Rationale:** The two scores live on different scales. Per-query z-scoring is label-free and makes one blend weight transferable. Calibration queries and relevance come only from the 10 calibration works.
- **Deviation noted:** the calibration candidate pool also contains the 300 validation tracks as negatives. With only the 20 calibration tracks, a K = 30 shortlist would contain the entire pool and calibration would not reflect shortlist behaviour. Validation tracks never provide positives or queries, and no development or benchmark track is involved.
- **Rejected:** tuning alpha or K on development queries. The K sweep and a post-hoc alpha curve are reported as analysis only.
- **Impact:** The encoder checkpoint was selected on validation tracks, so calibration-time global scores are slightly optimistic. This biases alpha toward the global score, not toward the reranker.

## D-013 — Exact NumPy cosine retrieval by default; FAISS optional
- **Date:** 2026-09-22
- **Decision:** A dense inner product over L2-normalised embeddings. FAISS `IndexFlatIP` is used only if requested and installed, and falls back to NumPy with a warning.
- **Rationale:** At 15,000 x 128 an exact search is milliseconds per query, so approximate search would add error and a dependency for no gain.

## D-014 — Benchmark never drops tracks
- **Date:** 2026-09-22
- **Decision:** For the benchmark, non-finite HPCP values are replaced by 0 and counted, and short tracks are upsampled (`min_frames: 1`). Development splits instead reject such WIDs.
- **Rationale:** The official protocol evaluates all 13,000 queries against all 15,000 candidates. Dropping hard tracks would inflate results.

## D-015 — No derived model weights in Git
- **Date:** 2026-09-22
- **Decision:** Checkpoints stay in `runs/` (git-ignored). Training is reproducible from the committed configs and manifests.
- **Rationale:** The weights derive from CC BY-NC-SA data, which needs a licence review before redistribution. They are also binary artefacts.

## D-016 — Encoder hyperparameters selected on validation works only
- **Date:** 2026-09-22
- **Decision:** A first run with patience 12 stopped at epoch 24 (best validation MAP 0.134 at epoch 12) while the training loss was still falling. Four 60-epoch variants were then trained, each keeping its best epoch by validation MAP: base settings (0.1810), 64 works per batch (0.1804), 512 input frames (0.1682) and lr 3e-3 (0.1459). The **base settings without early stopping** were selected (`configs/hybrid_dev.yaml`, checkpoint `runs/sweep_base/encoder_best.pt`).
- **Rationale:** Validation MAP on 300 tracks is noisy (±0.02 epoch to epoch), so patience 12 stopped an underfit model. Only validation WIDs were used; calibration, development and benchmark data played no part.
- **Evidence:** `reports/results/encoder_sweep.csv`, `training_history_sweep_*.csv`, `figures/training_history_sweep_*.png`. The base run with 2 threads reproduced the 8-thread run's losses exactly.
- **Rejected:** choosing by development metrics; larger sweeps (the differences between the top two runs are within validation noise).
- **Impact:** Five training runs in total. The selected model is a validation-selected best-of-4, which is slightly optimistic on validation; that is harmless because evaluation happens on disjoint works.


## D-017 — Report the benchmark reversal as a negative result
- **Date:** 2026-09-23
- **Decision:** The final benchmark run shows classical alignment (MAP 0.084) beating the hybrid (0.018) and Stage 1 alone (0.010), reversing the development ordering. This is reported as the headline finding in the README and `reports/hybrid_results.md`, rather than leading with the favourable development numbers. Nothing was re-tuned afterwards and the benchmark was not re-run.
- **Rationale:** The benchmark is the protocol that matters and it was evaluated once, with every setting locked beforehand. Re-tuning after seeing it would destroy the only unbiased estimate this project has.
- **Evidence:** `reports/results/benchmark.json` — 13,000 queries; shortlist recall 0.021; Recall@100 identical (0.053) for global and hybrid; work-level bootstrap intervals exclude zero for every comparison.
- **Rejected:** raising K, retuning alpha, or retraining on the benchmark evidence and reporting only the improved run.
- **Impact:** The headline claim of the project changes from "hybrid retrieval works" to "hybrid retrieval is 137x cheaper than exhaustive alignment but, with this Stage 1, much less accurate; shortlist recall is the constraint".

## D-018 — Hubness correction with an inductive probe reference
- **Date:** 2026-09-23
- **Decision:** Penalise each candidate by its general score level: `corrected[q, c] = score[q, c] - lam * reference[c]`. The reference is estimated against a **fixed probe set of 200 training-split tracks** (`mean` or `topk` of the probe scores), never from the evaluation queries and never from labels. `lam` and the method are grid-searched on the calibration protocol only.
- **Rationale:** The development error analysis showed tonally static tracks dominating false positives (dispersion vs. hub count, Spearman −0.77). A probe-based reference keeps the definition identical at any protocol size and costs 200 x C alignments (about 3M pairs on the benchmark) instead of the C x C matrix (225M).
- **Evidence:** `reports/results/hubness_correction*.json`. At 96 frames, calibration MAP 0.206 → 0.379 (lam = 0.5, mean) and development MAP 0.248 → 0.289 (ΔAP +0.041, CI [−0.045, +0.131]). The probe reference correlates with tonal dispersion at ρ = −0.81 (96 frames) and −0.90 (384 frames).
- **Rejected:** the query-side CSLS term (constant within a query's ranking, so it cannot reorder it); a transductive column-mean reference over the evaluation queries (cheap on the cached benchmark matrix, but its meaning changes with the query set).
- **Cost:** 148 s at 96 frames and 2,220 s at 384 frames for the benchmark catalogue, cached under `runs/probe_reference/` because probes, candidates and resolution fully determine it. It is a one-off per catalogue, not per query.
- **Impact:** One extra scalar, calibrated on 10 works. At 384 frames the correction still wins on calibration (0.304 → 0.407) but *loses* on development (0.384 → 0.355), so the two mechanisms overlap; both numbers are reported.

## D-019 — Alignment resolution for the reranker chosen on calibration works
- **Date:** 2026-09-23
- **Decision:** Keep `features.n_frames: 96` as the declared default of the base configuration, and additionally evaluate a 384-frame variant whose selection is justified by calibration MAP (0.206 at 96 vs. 0.304 at 384). Only the shortlist reranker can afford it at benchmark scale.
- **Rationale:** The resolution sweep in `reports/classical_baseline.md` showed 96 frames to be the binding limit of the classical stage. Reranking 30 candidates per query at 384 frames costs about 16x more per pair but only touches 0.2% of the pairs.
- **Evidence:** Development (locked settings): classical alignment 0.248 → 0.384, hybrid 0.262 → 0.362, ΔAP for hybrid over Stage 1 +0.173 [+0.044, +0.324] — the first development comparison whose interval excludes zero by a clear margin.
- **Rejected:** alignment-only retrieval at 384 frames over the whole benchmark (about 42 h of compute); changing the base default, which would invalidate comparisons with the already-reported runs.


## D-020 — Retrain the encoder on every usable Cover Analysis work
- **Date:** 2026-09-23
- **Decision:** Once the full Cover Analysis subset finished downloading, rebuild the manifests with `splits.n_train: all` (4,780 works / 9,560 tracks) and retrain with unchanged hyperparameters (`configs/full_train.yaml`). The earlier 1,500-work checkpoint and its results are kept for comparison.
- **Rationale:** The first benchmark evaluation showed Stage-1 recall (0.021) to be the binding constraint, and the 1,500-work limit was a bandwidth artefact (D-010), not a design choice.
- **Evidence:** Validation MAP 0.181 → 0.319; development global MAP 0.189 → 0.394; benchmark global MAP 0.010 → 0.034 and hybrid 0.018 → 0.058.
- **Split integrity:** roles are filled along the same seeded permutation, so calibration/query/distractor/validation are unchanged and the old training works are a strict subset of the new ones (asserted before training).
- **Impact:** No leakage: evaluation and calibration works were never in any training split. The loss was still falling at epoch 60, so this is a lower bound on what the architecture supports.


## D-021 — Train the encoder to convergence (150 epochs)
- **Date:** 2026-09-23
- **Decision:** The 60-epoch full-data run stopped with a falling loss, so the same configuration was trained for 150 epochs, still selecting the checkpoint by validation MAP (`runs/encoder_full_long`). This checkpoint is used for the fourth and final benchmark evaluation.
- **Rationale:** Epoch count is a training hyperparameter, selected like any other on validation works. Stopping at 60 was an arbitrary budget, not a result.
- **Evidence:** Validation MAP 0.319 → 0.431 (best epoch 138); benchmark Stage-1 MAP 0.034 → 0.076; hybrid + hubness correction 0.068 → 0.122; shortlist recall 0.072 → 0.136.
- **Impact:** The hybrid's deficit against exhaustive corrected alignment fell from 2.0x to 1.11x. Validation MAP was still drifting upward at epoch 150, so this is still not a converged model.

## D-022 — Report the development protocol as unrepresentative
- **Date:** 2026-09-23
- **Decision:** State plainly in the reports that the 20-query, 120-candidate development protocol twice gave the wrong ordering: it preferred the hybrid over classical alignment (the benchmark reversed this), and with a strong Stage 1 it said reranking hurts (the benchmark showed it helps, +0.037 with an interval excluding zero).
- **Rationale:** A protocol with 119 candidates and a single relevant item cannot model a 15,000-candidate pool with 12 relevant items. Reporting only the agreeing results would misrepresent how much the small protocol can be trusted.
- **Impact:** Future work on this repository should treat development numbers as a smoke test for plumbing and a source of hypotheses, not as evidence of ranking quality. The calibration protocol has the same weakness, which is why alpha and lambda are the only things it selects.
