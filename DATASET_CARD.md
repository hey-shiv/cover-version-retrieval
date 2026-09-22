# Dataset card — Da-TACOS as used in this project

## Source

| Item | Value |
|---|---|
| Dataset | Da-TACOS: *A Dataset for Cover Song Identification and Understanding* (Yesiler et al., ISMIR 2019) |
| Paper | https://archives.ismir.net/ismir2019/paper/000038.pdf |
| Homepage | https://mtg.github.io/da-tacos/ |
| Distribution used | Zenodo record [10.5281/zenodo.4717628](https://doi.org/10.5281/zenodo.4717628), v1.1.0 (the Google Drive links in the MTG README returned HTTP 404 in Sept 2026) |
| Files used | `da-tacos_metadata.zip` (3.5 MB), `da-tacos_coveranalysis_subset_hpcp.zip` (7.08 GB), `da-tacos_benchmark_subset_hpcp.zip` (10.09 GB) |
| Integrity | Whole archives: Zenodo MD5. Selective fetch: CRC-32 of every ZIP member (see `scripts/download_datacos.py`) |
| Metadata / features licence | **CC BY-NC-SA 4.0** (Music Technology Group, Universitat Pompeu Fabra) |
| Da-TACOS code licence | Apache-2.0 (not used here beyond the published Drive/Zenodo IDs) |

Da-TACOS contains **no audio**. It provides metadata and features pre-extracted from 44.1 kHz MP3s with the `acoss` framework (Essentia / librosa / madmom). This project uses **only the HPCP feature** and **only the WID/PID identifiers** from the metadata.

## Structure

| Subset | Tracks | Works (cliques) | Clique sizes | Role in this project |
|---|---|---|---|---|
| Cover Analysis | 10,000 | 5,000 | exactly 2 | training, validation, calibration, development evaluation |
| Benchmark | 15,000 | 3,000 | 1,000 x 13 + 2,000 x 1 (noise) | **final test only** |

Verified locally from the metadata: WID sets **and** PID sets of the two subsets are disjoint (0 shared WIDs, 0 shared PIDs), so training on Cover Analysis cannot leak into the benchmark.

* **WID** (work ID, `W_…`): the composition / cover clique. Two tracks are relevant to each other **iff** they share a WID.
* **PID** (performance ID, `P_…`): one recording. A query's own PID is always removed from its ranking.
* HPCP files: `<subset>_hpcp/<WID>_hpcp/<PID>_hpcp.h5`, written by `deepdish`: a `hpcp` dataset of shape `(T, 12)` float32 (per-frame max-normalised to [0, 1]), and `label` (WID) / `track_id` (PID) attributes. The loader cross-checks both attributes against the manifest.

## How the data is split (see `data/manifests/`, D-001, D-008, D-010)

1. Cover Analysis WIDs are sorted lexicographically and permuted with `numpy.random.default_rng(20260817)`.
2. Walking that order, a WID is **usable** only if it has exactly two PIDs and both HPCP files exist, are readable, carry matching labels, are finite, not silent, and have >= 96 frames. Unusable WIDs are logged in `coveranalysis_excluded.csv`.
3. Usable WIDs fill the roles in fixed order:

| Role | WIDs | Tracks | Use |
|---|---|---|---|
| calibration | 10 | 20 | the only data used to choose the hybrid blend weight |
| query | 20 | 40 | 20 development queries (lexicographically lowest PID per WID); the partner is the single relevant item |
| distractor | 40 | 80 | development negatives |
| validation | 150 | 300 | encoder checkpoint selection / early stopping; extra negatives for calibration |
| train | 1,500 | 3,000 | encoder training |

Development protocol: 20 queries ranked against 120 candidates (query + distractor tracks), self excluded, 119 ranked items per query. Every role is WID-disjoint from every other role; tests enforce this.

The Benchmark is never used for training, tuning, model selection or calibration. It is evaluated once, with settings locked beforehand: 13,000 clique tracks as queries, all 15,000 tracks as candidates (the 2,000 noise tracks stay in the pool), self excluded. No benchmark track is dropped: non-finite values would be zeroed and counted, and short tracks upsampled.

## Leakage rules

* Retrieval uses **only HPCP**. Titles, artists, release years, MusicBrainz tags, instrumental flags, and the Da-TACOS key/tempo/madmom features are never loaded into any model or score. `work_index()` returns identifiers only.
* Key invariance comes from the audio feature (12 cyclic rotations), never from key annotations; tempo invariance comes from DTW and resampling, never from tempo annotations.
* The encoder is trained on train WIDs only; validation WIDs select the checkpoint; the calibration WIDs choose alpha; development and benchmark queries choose nothing.

## What is committed and what is not

Committed: ID-only manifests (`W_…`, `P_…`, role), per-track shape statistics, aggregate metrics, per-query metric CSVs, plots, and configs.

Never committed (see `.gitignore`): metadata JSON, H5 features, ZIP archives, preprocessed caches, and model checkpoints. The checkpoints are derived from CC BY-NC-SA data, and redistributing them would need a licence review.

## Acquisition

```bash
python scripts/download_datacos.py --what metadata
# Everything (17 GB of ZIPs, needs >= 35 GB free while unpacking):
python scripts/download_datacos.py --what coveranalysis_hpcp benchmark_hpcp
# ...or, on slow links, only the files the manifests need (resumable, CRC-checked):
python scripts/download_datacos.py --fetch-members coveranalysis --config configs/base.yaml
python scripts/download_datacos.py --fetch-members benchmark --config configs/base.yaml
```

Attribution: *Furkan Yesiler, Chris Tralie, Albin Correya, Diego F. Silva, Philip Tovstogan, Emilia Gómez, Xavier Serra. Da-TACOS: A Dataset for Cover Song Identification and Understanding. ISMIR 2019, pp. 327–334.* Licensed CC BY-NC-SA 4.0. Any derivative data (for example, trained weights) inherits the NonCommercial and ShareAlike terms.

## Limitations

* **Feature-only, fixed extraction.** No audio means no re-extraction, no source separation, no lyrics/ASR, and no verification of feature quality. HPCP parameters are fixed by `acoss`.
* **Age and catalogue bias.** SecondHandSongs-derived metadata (2019) is dominated by Western popular music; it does not represent current catalogues, live/user-generated covers, or non-Western tonal systems well.
* **Two-track cliques.** Cover Analysis provides one positive per work, which limits metric-learning supervision.
* **Uncontrolled nuisances.** Recording quality, duration and arrangement differences are unannotated here (by design, since metadata is not used), so robustness to them cannot be measured directly.
* **No real-world robustness claim.** Results on Da-TACOS say nothing about noisy, short, or partial queries, streaming-scale catalogues, or adversarial edits.
