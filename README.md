# Structure-Aware Hybrid Retrieval for Cover Song Identification

A reproducible music-information-retrieval system that finds **cover versions** (other recordings of the same composition) on the [Da-TACOS](https://mtg.github.io/da-tacos/) dataset, using only pre-extracted **HPCP** (chroma) features.

It is a two-stage hybrid:

1. **Stage 1 — learned global retrieval.** A compact temporal-convolutional encoder maps a whole track to one 128-d unit vector; exact cosine search ranks every candidate and keeps a shortlist of K = 30.
2. **Stage 2 — structure-aware classical reranking.** For each shortlisted pair, choose the best of 12 key rotations, build a cosine-distance cross-similarity matrix, run slope-constrained **subsequence DTW**, and blend the alignment score with the global score. The blend weight is calibrated only on held-out calibration works.

Everything is deterministic from committed configs and ID-only manifests, tested with `pytest`, and runnable from a fresh clone.

> **Headline finding (please read before the architecture).** On the full Da-TACOS benchmark the **classical alignment baseline beats this hybrid by 4.6x** (MAP 0.084 vs. 0.018), reversing what the 20-query development protocol suggested. Reranking does beat embedding-only retrieval, but Stage-1 recall is the binding constraint: the K = 30 shortlist holds only 2.1% of relevant items. The hybrid's real win is cost — 5.3 ms per query versus 2.63 h of exhaustive alignment. The project is presented as a reproducible pipeline **and** a negative result about this two-stage design at scale, not as a state-of-the-art system. See [Results](#results).

---

## Problem

Cover song identification (CSI) is a **retrieval** problem: given a query recording, rank a collection so that other versions of the same work come first. Covers change key, tempo, structure, instrumentation and vocals, so neither raw spectral similarity nor exact matching works. Relevance is defined only by the SecondHandSongs *work* ID (WID); each recording has a performance ID (PID).

## System

```mermaid
flowchart LR
    subgraph Data["Da-TACOS (features only, CC BY-NC-SA)"]
        H5["HPCP (T, 12) per PID"] --> V["validate: shape / labels / finite / length"]
    end
    V --> P1["frame L2 -> area-resample -> frame L2"]
    P1 -->|"(12, 512) cache"| ENC["TCN encoder<br/>6 dilated residual blocks<br/>mean+max pool -> 128-d, L2"]
    P1 -->|"(12, 96)"| CLS["classical view"]
    ENC --> IDX["exact cosine index<br/>(NumPy; FAISS optional)"]
    Q["query"] --> IDX
    IDX -->|"Stage 1: rank all, keep top-K"| SL["shortlist (K = 30)"]
    SL --> ROT["12 key rotations<br/>(profile cosine)"]
    CLS --> ROT
    ROT --> CSM["cosine cross-similarity"] --> DTW["subsequence DTW<br/>steps (1,1),(2,1),(1,2)"]
    DTW --> BL["alpha * z(global) + (1 - alpha) * z(alignment)<br/>alpha from calibration WIDs only"]
    BL --> OUT["final ranking = reranked shortlist + untouched remainder"]
```

| Component | Classical or learned | Where |
|---|---|---|
| HPCP loading, validation, orientation, normalisation, resampling | classical | `src/cover_retrieval/data/datacos.py`, `features/` |
| 12-rotation key invariance | classical | `alignment/transposition.py` |
| Cross-similarity + DTW / subsequence DTW (reference with backtracking, plus a batched PyTorch version) | classical | `alignment/csm.py`, `alignment/dtw.py` |
| TCN encoder, SupCon / triplet losses, deterministic training | learned | `models/` |
| Exact embedding index, hybrid reranking, alpha calibration | glue | `retrieval/` |
| Metrics, work-level bootstrap, error analysis, plots | evaluation | `evaluation/` |

The alignment code is written from the textbook DTW recurrence (FMP §3.2). No cover-song library is imported, and no code comes from the AGPL-licensed `acoss`.

## Setup

Python 3.11 or newer (developed on 3.12).

```bash
git clone <this repo> && cd cover-version-retrieval
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ".[faiss]" for the optional FAISS backend
pytest                            # 112 tests, including the synthetic end-to-end pipeline
```

## Data

Da-TACOS has no audio. This project needs the metadata (for WID/PID identifiers) and the HPCP features, about 17 GB of ZIP archives in total. Everything lands in `data/external/`, which is git-ignored. Licensing and leakage rules are in [DATASET_CARD.md](DATASET_CARD.md).

```bash
# metadata (3.5 MB)
python scripts/download_datacos.py --what metadata

# fast link: whole HPCP archives (MD5-verified, needs >= 35 GB free while unpacking)
python scripts/download_datacos.py --what coveranalysis_hpcp benchmark_hpcp

# slow link: only the files the manifests need, via HTTP range requests (resumable, CRC-checked)
python scripts/download_datacos.py --fetch-members coveranalysis --config configs/base.yaml
python scripts/download_datacos.py --fetch-members benchmark --config configs/base.yaml
```

## Running

All commands run from the repository root.

```bash
python scripts/build_manifests.py        --config configs/base.yaml               # deterministic WID-disjoint splits
python scripts/profile_data.py           --config configs/base.yaml --include-benchmark
python scripts/run_classical_baseline.py --config configs/classical_baseline.yaml --resolution-sweep 48 192 384
python scripts/train_encoder.py          --config configs/hybrid_dev.yaml
python scripts/run_hybrid_retrieval.py   --config configs/hybrid_dev.yaml          # calibrates + locks alpha, dev evaluation
python scripts/run_benchmark.py          --config configs/benchmark.yaml --with-alignment-only   # FINAL test
```

Smoke test on **synthetic** data, which exercises every script. Its outputs go to `runs/smoke/`, never `reports/`:

```bash
PYTHON=python ./scripts/run_smoke.sh
```

Notebook: `notebooks/01_feature_and_alignment_inspection.ipynb`, regenerated by `scripts/build_notebook.py`. It imports library functions only.

## Project layout

```
configs/          base.yaml (+ classical_baseline, hybrid_dev, benchmark, smoke)
data/manifests/   committed ID-only manifests (splits, protocols, exclusions)
data/external/    Da-TACOS downloads (git-ignored)
src/cover_retrieval/
  data/           datacos.py, manifests.py, splits.py, remote_zip.py, synthetic.py
  features/       hpcp.py, preprocessing.py (validated, fingerprinted feature cache)
  alignment/      transposition.py, csm.py, dtw.py
  models/         tcn_encoder.py, losses.py, training.py
  retrieval/      index.py, rank.py, hybrid.py
  evaluation/     metrics.py, analysis.py
  utils/          io.py, seed.py
  pipeline.py     shared building blocks used by the scripts
scripts/          download, manifests, profiling, training, evaluation, smoke, notebook
tests/            unit, property and end-to-end tests
reports/          dataset_profile, classical_baseline, hybrid_results, error_analysis (+ results/, figures/)
notes/            decisions.md (D-001 onward), papers/
```

## Reproducibility controls

* **Fixed seed** `20260817` for split permutation, batch order, augmentation and initialisation (`utils/seed.py`). Training runs on CPU with `torch.use_deterministic_algorithms` and a fixed thread count; a test checks that two runs give bit-identical weights.
* **Frozen protocols.** Committed manifests list every selected WID/PID and role. Rebuilding them is byte-identical (tested). Candidates are sorted by PID, so score ties break deterministically.
* **Fingerprinted caches.** Preprocessed features are keyed by a hash of the track list and every preprocessing setting.
* **Locked evaluation settings.** The encoder checkpoint is chosen on validation works, and alpha/K on calibration works; both are written to `reports/results/hybrid_calibration.json` before any development or benchmark query is scored. `run_benchmark.py` refuses to run without that file and tunes nothing.
* **Provenance.** Every result JSON records the git commit and config path.

## Results

### Development protocol (Da-TACOS Cover Analysis, 20 queries x 120 candidates)

One relevant item per query, self excluded, so **AP = reciprocal rank and MAP = MRR by construction**. A random ranking scores about 0.045 MAP. Every setting was locked before the run; alpha and K come from calibration works only.

| system | MAP (= MRR) | Hit@1 | Hit@10 | mean first rank |
|---|---|---|---|---|
| global embedding (Stage 1 only) | 0.189 | 0.10 | 0.35 | 23.5 |
| global embedding + test-time rotations | 0.200 | 0.10 | 0.55 | 20.6 |
| classical alignment over all candidates | 0.248 | 0.15 | 0.45 | 31.3 |
| **hybrid (K = 30, alpha = 0.05)** | **0.262** | 0.15 | **0.55** | **20.8** |
| rerank only (K = 30, alpha = 0) | 0.265 | 0.15 | 0.55 | 20.6 |

Paired work-level bootstrap (10,000 resamples), ΔAP [95% CI]: hybrid − global **+0.073 [−0.000, +0.178]**; rerank-only − global **+0.076 [+0.003, +0.181]**; hybrid − classical +0.014 [−0.009, +0.042]. With 20 queries only the rerank-only gain over Stage 1 clears zero, and barely. Reranking 30 of 119 candidates matches full alignment while doing a quarter of the alignment work.

Classical ablations: no key handling 0.303, profile-selected rotation 0.248, exhaustive 12-rotation DTW 0.248, profile-only (no temporal model) 0.266 — all mutually indistinguishable at this sample size. Details and the resolution sweep (MAP rises to 0.384 at 384 frames) are in [reports/classical_baseline.md](reports/classical_baseline.md).

Encoder: 645,504 parameters, trained on 1,500 works, selected by validation MAP over 4 variants (best 0.181 on 150 disjoint validation works); see [reports/hybrid_results.md](reports/hybrid_results.md).

**Main error mechanism (hubness).** Tonally static tracks attract low DTW cost from everything. Across the 120 candidates, tonal dispersion vs. top-10 false-positive count gives Spearman ρ = −0.77 (classical) and −0.57 (hybrid); the worst hub is a false positive for 11 of 20 queries where 1.7 is expected. See [reports/error_analysis.md](reports/error_analysis.md).

### Final Da-TACOS benchmark (13,000 queries x 15,000 candidates)

Final run, all settings locked beforehand. 13,000 queries, 15,000 candidates, 12 relevant items per query, self excluded. Random ≈ 0.0008 MAP.

| system | MAP | MRR | Hit@1 | Hit@10 | median first rank |
|---|---|---|---|---|---|
| global embedding (Stage 1 only) | 0.010 | 0.052 | 0.025 | 0.096 | 185 |
| hybrid (K = 30, alpha = 0.05) | 0.018 | 0.124 | 0.102 | 0.154 | 185 |
| rerank only (K = 30, alpha = 0) | 0.019 | 0.125 | 0.103 | 0.154 | 185 |
| **classical alignment (all candidates)** | **0.084** | **0.282** | **0.244** | **0.352** | **77** |

Work-level bootstrap over 1,000 cliques: hybrid − global **+0.0084 [+0.0075, +0.0094]**, classical − global **+0.0735 [+0.0669, +0.0800]**. With 13,000 queries these are decisive.

**Negative result, reported as such.** The development ordering reverses at scale: classical alignment beats the hybrid **4.6x** (0.084 vs. 0.018). Reranking does beat Stage 1 alone, but Stage-1 recall is the binding constraint — the K = 30 shortlist contains only **2.1%** of relevant items, and Recall@100 is identical for global and hybrid because beyond the shortlist the hybrid ranking *is* the global ranking. A 120-candidate development protocol cannot predict behaviour at 15,000 candidates.

What the hybrid does buy is speed: 5.3 ms of reranking per query versus 2.63 h for exhaustive alignment over 1.95 x 10^8 pairs (**137x less alignment work**). All systems sit well below published Da-TACOS results, as expected from a 96-frame pilot resolution and an encoder trained on 1,500 works. Full tables, runtimes and analysis: [reports/hybrid_results.md](reports/hybrid_results.md).

### Synthetic smoke pipeline

`scripts/run_smoke.sh` exercises every script end to end on a synthetic fixture. Its numbers are **not** Da-TACOS results and are written to `runs/smoke/`, never to `reports/`.

### Reproducibility evidence

From a fresh clone with the same data, the manifests rebuild byte-identically, the classical rankings and metrics match exactly, and retraining the encoder reproduces the selected checkpoint **bit-for-bit** (epoch 50, validation MAP 0.18097, identical weights).

## Limitations

* **Absolute accuracy is low.** All four systems sit well below published CSI systems on this benchmark. The two causes are deliberate and documented: a 96-frame pilot resolution for alignment (D-005) and an encoder trained on 1,500 of about 4,800 usable works (D-010).
* **Stage 1 is too weak for the architecture to pay off.** Shortlist recall of 0.021 at K = 30 caps the hybrid regardless of how good the reranker is.
* **Pilot resolution.** The classical stage averages each track down to 96 frames (D-005), which blurs harmonic rhythm. The resolution sweep in `reports/classical_baseline.md` shows how much this costs: dev MAP rises from 0.248 at 96 frames to 0.384 at 384.
* **Small development protocol.** 20 queries against 120 candidates; bootstrap intervals are wide, and dev-set differences whose interval spans zero are not claimed as improvements. It also proved unrepresentative: its system ordering did not survive at benchmark scale.
* **Training data budget.** The encoder was trained on 1,500 of the roughly 4,800 usable Cover Analysis works, because of a 0.3–0.45 MB/s link (D-010). With two recordings per work, supervision is thin.
* **Asymmetric, whole-query alignment.** Subsequence DTW aligns the entire query; a cover that drops or reorders sections is penalised. Serrà-style local alignment (Qmax) is not implemented.
* **Dataset scope.** Da-TACOS is feature-only, from 2019, and Western-pop-centric. Nothing here establishes robustness to short or partial queries, live or user-generated recordings, or production-scale catalogues.

## Next research directions

Ordered by what the benchmark evidence actually demands:

1. **Fix Stage-1 recall**, the binding constraint: train on all Cover Analysis works, and measure shortlist recall against K directly. Reranking quality is irrelevant until covers reach the shortlist.
2. **Hubness correction** (per-candidate score centring, CSLS), calibrated on calibration works. Tonal dispersion predicts false positives at ρ = −0.77, and it also explains why classical alignment has the best median first rank (77) but the worst mean (528).
3. **Higher classical resolution** (or beat-synchronous frames), and whether the dev-set gain survives at benchmark scale.
4. Local alignment (Qmax-style) versus subsequence DTW for covers with changed structure.
5. CREMA/HPCP feature fusion, and error-stratified evaluation by rotation shift and length ratio.

## References

* Yesiler et al., *Da-TACOS*, ISMIR 2019. · Serrà et al., *Chroma binary similarity and local alignment applied to cover song identification*, IEEE TASLP 2008.
* Müller, *Fundamentals of Music Processing* (2nd ed.) and FMP notebooks: [music synchronization](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3_MusicSynchronization.html), [DTW](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html).
* [Essentia cover-song similarity tutorial](https://essentia.upf.edu/tutorial_similarity_cover.html). · Manning et al., [*Introduction to IR*, ch. 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html).
* Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020. · Bai, Kolter & Koltun, *TCNs for sequence modeling*, 2018. · Smucker, Allan & Carterette, CIKM 2007.

## Licence

Code: MIT (see `LICENSE`). Da-TACOS metadata and features: CC BY-NC-SA 4.0, © Music Technology Group, UPF. They are not redistributed here.
