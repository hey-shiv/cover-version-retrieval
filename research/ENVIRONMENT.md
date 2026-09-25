# Execution environment and artifact inventory

Checked on 2026-09-24, inside the cloud session that produced `research/`. This file
decides what kind of evidence every experiment in this directory can be.

## What exists here

| artifact | present? | notes |
|---|---|---|
| Da-TACOS HPCP features (`data/external/`, ~17 GB of archives) | **no** | never committed (CC BY-NC-SA); not downloadable here |
| Preprocessed feature caches (`data/cache/`) | **no** | git-ignored |
| Encoder checkpoints, cached embeddings, alignment matrices (`runs/`) | **no** | git-ignored; derived from NC-SA data (D-015) |
| ID-only manifests (`data/manifests/`) | yes | splits, protocols |
| Result files (`reports/results/`, 65 files, 7.3 MB) | yes | see below |
| Report figures (`reports/figures/`) | yes | PNG plots |
| Synthetic Da-TACOS fixture generator (`scripts/make_synthetic_datacos.py`) | yes | exercises every script; not real data |

The only `.h5` files on disk are the synthetic fixtures `pytest` writes to `/tmp`.

What the committed result files contain, per experiment:

- **Benchmark, four runs** (`benchmark*.json`, `benchmark_per_query*.csv`): aggregate metrics, runtimes and bootstrap intervals. For each of the 13,000 queries and each system: AP and the rank of the *first* relevant item. Not included: full rankings, the ranks of the other 11 covers, Stage-1 scores or embeddings.
- **Development protocol, five configurations** (`hybrid_dev*.json`, per-query CSVs): metrics, a measured K sweep (K = 10, 30, 50, 100, 119), and the hybrid top 10 per query.
- **Classical development rankings** (`classical_dev_rankings.csv`): all 20 × 119 pairs, with alignment score and chosen key shift.
- **Hubness** (`hubness_dev.json`, `hubness_correction*.json`), **calibration grids**, **training histories**.

## Network

The environment's egress policy blocks `zenodo.org` (dataset download), `api.vercel.com`,
`arxiv.org`, `api.semanticscholar.org`, `api.crossref.org`, `music-ir.org` and most
publisher sites. Web *search* works; fetching most primary papers does not.

## Compute

4 CPU cores, 15 GB RAM, about 21 GB free disk, no GPU. Python 3.11.15, torch 2.14.0 (CPU),
numpy 2.4.6, matplotlib 3.11.2.

## Consequence: four kinds of evidence

Every result in `research/` carries exactly one of these labels.

| label | meaning |
|---|---|
| **ARTIFACT-ANALYSIS** | new analysis computed here from committed result files. Exact for what those files contain; runs no model. |
| **ARTIFACT-REPRODUCTION** | committed aggregate metrics recomputed from committed per-query files. Checks internal consistency; does **not** re-run the pipeline. |
| **SYNTHETIC-MECHANICS** | a new runner executed on the synthetic fixture to prove it works. Its numbers mean nothing about music. |
| **NOT-EXECUTED** | designed, registered, and (where possible) implemented, but not run: it needs Da-TACOS features or checkpoints that are not in this environment. |

No experiment here is a new run on Da-TACOS, and none is presented as one.

## Cost to execute the NOT-EXECUTED series elsewhere

These estimates use the runtimes measured in `benchmark_long384.json` on the author's laptop CPU:

- Features: about 17 GB of archives (35 GB free while unpacking), or selective fetch through `scripts/download_datacos.py`.
- Stage 1 for 15,000 tracks: about 69 s.
- Reranking at 384 frames: 38.1 ms per query at K = 30, so about 1.27 ms per aligned pair.
- **K sweep on the benchmark** at K ∈ {5, 10, 20, 30, 50, 100, 200, 500}, reusing one Stage-1 run and aligning each pair once at the largest K: 13,000 × 500 = 6.5 × 10⁶ pairs, about **2.3 h** at 384 frames (about 19 min at 96 frames, 0.177 ms per pair).
- Hubness probe reference: 200 × 15,000 pairs, about 37 min at 384 frames (one-off).
