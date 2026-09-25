# Baseline specification (verified from code and configs)

Each item cites the file it was read from. Where documentation and code differ, the code wins, and the difference is noted.

## Features and preprocessing
- **Input.** Da-TACOS HPCP, a raw `(T, 12)` array per recording. No audio is used or distributed (`src/cover_retrieval/data/datacos.py`).
- **Two views** (`features/preprocessing.py`), each computed as frame-wise L2 normalisation → area resampling to a fixed length → frame-wise L2 normalisation:
  - *alignment view*: `(12, n_frames)`, with n_frames = 96 (runs 1–2, `configs/base.yaml`) or 384 (runs 3–4, set with `features.n_frames=384`);
  - *encoder view*: `(12, 512)` cached, then area-pooled to 256 input frames with the frames re-normalised (`models/tcn_encoder.py::prepare_input`).
- **Non-finite values.** Rejected for split selection; zeroed on the benchmark, where no track may be dropped (`data.nonfinite_policy`).

## Stage 1: TCN encoder (`models/tcn_encoder.py`, `configs/base.yaml`)
- 1×1 conv 12 → 128 channels.
- 6 residual blocks, kernel 3, dilations 1, 2, 4, 8, 16, 32. Each block is [conv → BN → GELU → dropout 0.1 → conv → BN] + skip → GELU. Convolutions are non-causal and length-preserving.
- Receptive field 1 + 2(k−1)(2⁶−1) = **253 frames**, out of 256 input frames (`training_summary_*.json`).
- Pooling: concat(mean over time, max over time) → 256-d. Head: Linear 256→128 → GELU → Linear 128→128, then L2 normalisation. **Embedding 128-d.**
- **645,504 trainable parameters** (`training_summary_encoder_full_long.json`).
- **Loss:** supervised contrastive, temperature 0.1 (`models/losses.py`). Each batch has 32 works × 2 recordings.
- **Augmentation:** random time crop of 60–100% of the length, plus a random cyclic pitch rotation of 0–11 semitones (`models/training.py::augment`).
- **Optimiser:** AdamW, lr 1e-3, weight decay 1e-4, cosine annealing to 0 over the scheduled epochs (`models/training.py`, lines 140–141). CPU training, bitwise-reproducible with seed 20260817.
- **Checkpoint selection:** best validation MAP (150 validation works), patience 12.

| checkpoint | training works | epochs | best epoch | best validation MAP |
|---|---|---|---|---|
| `sweep_base` (run 1) | 1,500 | 60 | 50 | 0.1810 |
| `encoder_full_train` (runs 2–3) | 4,780 | 60 | 58 | 0.3187 |
| `encoder_full_long` (run 4) | 4,780 | 150 | 138 | 0.4306 |

## Stage 1 retrieval
- Exact cosine search over all 15,000 embeddings (numpy backend) with self-matches excluded. The shortlist is the top K = 30 (`retrieval.shortlist_k`).
- Test-time rotations are off in every frozen run.

## Stage 2: alignment (`alignment/`, `retrieval/rank.py`)
1. **Key.** Choose the rotation k* = argmax_k cos(p̄_q, roll(p̄_c, k)), where p̄ is the time-averaged, L2-normalised chroma profile (`transposition.py`, `alignment.rotation_selection: profile_cosine`).
2. **Cost.** C[i, j] = 1 − cos(q_i, roll(c, k*)_j) (`csm.py`); frames with zero norm get similarity 0.
3. **Subsequence DTW** (`dtw.py`):
   - recurrence: D[i, j] = min over (di, dj) of D[i−di, j−dj] + w(di, dj)·C[i, j];
   - steps Σ = {(1,1), (2,1), (1,2)} with weights {1, 2, 1};
   - initialisation D[0, j] = C[0, j]; the path ends at argmin_j D[N−1, j].
4. **Normalisation.** Accumulated cost divided by the accumulated step weight. With these weights every admissible path has weighted length N, so minimising D and minimising the normalised cost are the same objective. The slope constraint (1/2 to 2) forbids degenerate paths that collapse the query onto one candidate frame.
5. **Alignment score.** a(q, c) = 1 − normalised cost (`rank.py`, line 119).

## Hybrid fusion (`retrieval/hybrid.py`)
- Within each query's shortlist: h = α·z(g) + (1−α)·z(a), where z is a z-score over the K shortlist items, g the cosine and a the alignment score.
- Items outside the shortlist keep their Stage-1 order below the shortlist.
- α is grid-searched in steps of 0.05 on the 10 calibration works by MAP (ties → lower mean first rank → α closer to 0.5). Locked values: 0.05 (run 1), 0.15 (runs 2–3), 0.10 (run 4).

## Hubness correction (`retrieval/normalization.py`, `hubness_correction*.json`)
- a′(q, c) = a(q, c) − λ·r(c), where r(c) is the mean alignment score of candidate c against **200 training-split probe queries**. This is inductive: no test query is used, and the correction costs 3 × 10⁶ extra pair alignments.
- λ is chosen on calibration works: 0.5 at 96 frames, 0.6 at 384 frames. The method is `mean`; `topk`, the candidate-side CSLS term, was also calibrated but not selected.
- The query-side CSLS term is omitted because it is constant within a query's ranking.

## Metrics and statistics (`evaluation/metrics.py`)
- MAP, MRR, Hit@k (at least one relevant item in the top k), Recall@k, first-relevant rank. The benchmark has 12 relevant items per query.
- Shortlist recall = mean over queries of (relevant items in the top K)/12. Coverage C(K) = share of queries with at least one relevant item in the top K (`research/`).
- **Bootstrap:** paired, resampling the 1,000 query works (cliques) with replacement, 10,000 resamples. The development protocol resamples 20 query works.

## Runtime (measured, from each run's `runtime` block)
- Embedding 15,000 tracks: 68–69 s. Exact search: 0.07–0.10 s. Shortlisting: about 9.4 s.
- K = 30 rerank: 69 s at 96 frames (5.3 ms/query); 494–495 s at 384 frames (38 ms/query).
- Exhaustive alignment at 96 frames, alignment only: **9,458 s (2.63 h)** for 1.95 × 10⁸ pairs (0.0485 ms/pair). The hub-corrected exhaustive system adds the probe reference (146 s at 96 frames).
- Exhaustive alignment at 384 frames: **estimated about 42 h, not run.**
