# Literature review

**Scope and verification.** This environment's network policy blocks arXiv, Semantic Scholar, Crossref, MIREX and most publisher pages. So this review rests on three kinds of evidence, and every claim is tagged with one:

- **[S]**: title, venue, year and the stated claim confirmed by a web search in this session (2026-09-24/25). The full paper was not read.
- **[R]**: taken from `notes/literature/novelty_assessment.md`, which was verified against Crossref, arXiv, jmlr.org and ISMIR proceedings when it was written (commit 879c9e3), including the Da-TACOS benchmark numbers quoted there.
- **[U]**: supplied by the user and **not verifiable here**. Not used in any claim.

Numbers are only quoted where they carry [R] or [S]. A reader with network access should re-verify every [S] entry against the paper before submission (tracked in `reviewer_audit.md`).

The machine-readable version is [`literature_matrix.csv`](literature_matrix.csv).

## 1. Classical, alignment-based CSI

- **Serrà, Gómez, Herrera & Serra (2008)**, IEEE TASLP. Chroma binary similarity, optimal transposition index, and local alignment (Smith–Waterman style). [R]
- **Serrà, Serra & Andrzejak (2009)**, NJP: *Qmax*, cross-recurrence quantification. On the Da-TACOS benchmark with HPCP it reaches MAP **0.333** (Yesiler et al. 2019, Table 2). [R] This is the strongest classical point of comparison on this exact protocol. The repository's best classical system (0.136) is about 2.5× lower.
- Da-TACOS HPCP baselines (Yesiler et al. 2019): Dmax 0.292, SiMPle 0.165, FTM2D 0.126; EarlyFusion 0.426. [R]
- **Müller**, *Fundamentals of Music Processing*: the textbook DTW and subsequence-DTW formulation this repository implements. [R]

**Established:** key transposition by the optimal transposition index, alignment over cross-similarity, and local rather than global alignment for structural change. None of it is new here.

## 2. Scalability and indexing in CSI

- **Bertin-Mahieux & Ellis (2012)**, ISMIR: 2D Fourier transform magnitude of beat-synchronous chroma with PCA, as a fixed-length summary for million-track scale. It states explicitly that DTW-style comparisons do not scale to millions of items. [S]
- **Time-complexity evaluation of CSI algorithms** (a paper surfaced by search, not read). [S, title only]

**Established:** the accuracy/cost tension in CSI, and fixed-length summaries as the scalable step, are long-standing.

## 3. Learned global embeddings

- **MOVE** (Yesiler, Serrà & Gómez, ICASSP 2020), CREMA input: MAP 0.507 on Da-TACOS. **Re-MOVE** (ISMIR 2020, embedding distillation and pruning): 0.534, trained on a Da-TACOS training partition of 83,904 songs in 14,499 works (about 17× this project's training set). [R, S]
- **ByteCover** (Du et al., ICASSP 2021), CQT, multi-loss (classification + triplet), ResNet-IBN: MAP 0.714 on Da-TACOS. [R, S]
- **ByteCover2** (Du et al., ICASSP 2022): PCA-FC dimensionality reduction; reported to beat ByteCover even at 128 dimensions. [S] According to the Discogs-VI authors, ByteCover2 placed first in MIREX 2024 CSI and is not open source. [S] The MIREX 2024 mAP of 0.877 **[U]** could not be verified here.
- **CoverHunter** (Liu et al., ICME 2023): Conformer, attention-based time pooling, and a *coarse-to-fine training* scheme that aligns song chunks. [S]
- **Discogs-VI / Discogs-VINet** (Araz, Serra & Bogdanov, ISMIR 2024): about 493k versions in 98k cliques; a simple baseline competitive on SHS100K and Da-TACOS; 2nd in MIREX 2024. [S]
- **LIVI** (Affolter et al., ECIR 2026): lyrics-informed audio embeddings. [S]

**Established:** learned whole-track embeddings are the scalable Stage 1 of current CSI, and data scale is a first-order factor. The repository's encoder (645k parameters, 4,780 two-recording works, HPCP input) is small and data-poor by comparison.

## 4. Local, multi-vector and two-stage CSI

- **ByteCover3** (ICASSP 2023, the ByteCover line; author list not confirmed here): local features trained with a *local alignment loss*, and a **two-stage retrieval pipeline: ANN search, then re-ranking**, aimed at short queries. [S]
- **LIVI** (ECIR 2026): **two-stage**. It retrieves candidates with track-level embeddings, then reranks all of them with a **MaxSim operator over 30-second local embeddings**. [S]
- **ColBERT** (Khattab & Zaharia, SIGIR 2020) and ColBERTv2: late interaction and MaxSim over multi-vector representations; the origin of the MaxSim operator. [S]

**Established:** global retrieval followed by local or alignment-based reranking is the *standard architecture* of recent CSI, not a contribution. Windowed multi-vector scoring (experiment F1's `win_max`) is a MaxSim-style late interaction. It is new here only as a zero-retraining variant of *this* encoder, and it needs its own evidence.

## 5. Hubness and score normalisation

- **Aucouturier & Pachet (2008)**: the scale-free distribution of false positives in audio similarity ("hubs"). **Radovanović et al. (2010)**, JMLR: hubness arises in high dimensions. **Schnitzer et al. (2012)**, JMLR: mutual proximity and local scaling. **Flexer et al. (2012)**: hubness across MIREX systems. [R]
- CSI-specific: **Seo (2022)**, IEICE, frame-level hubness normalisation for OTI + Smith–Waterman. **Li & Chen (2018)**: mutual proximity after similarity fusion. **Hu & Chen (2019)**, TASLP: hub reduction in subsequence communities. [R, S]
- **CSLS** (Conneau et al., 2018) and **QB-Norm** (Bogolin et al., CVPR 2022): probe- or query-bank normalisation, which is the form of the repository's correction. [R]

**Established:** hubness correction for CSI, and probe-based normalisation. The repository's contribution here is evidence, not method: a held-out-calibrated measurement at Da-TACOS scale on a weak baseline (novelty_assessment.md).

## 6. Multi-stage retrieval, cascades and adaptive depth (information retrieval)

- **Wang, Lin & Metzler (2011)**, SIGIR: cascade ranking models that trade efficiency for effectiveness. [S]
- **Culpepper, Clarke & Lin (2016)**, ADCS: *Dynamic Cutoff Prediction in Multi-Stage Retrieval Systems*. It predicts, per query, the size of the candidate set passed to the next stage, noting that "optimal settings vary across queries". [S]
- **Efficient cost-aware cascade ranking** (SIGIR 2017); **Ranked List Truncation for LLM-based Re-Ranking** (SIGIR 2024; authors not confirmed here). [S]
- Query performance prediction from score distributions is a standard IR topic. [S]
- **Jacob, Lindgren, Zaharia, Carbin, Khattab & Drozdov (2024)**, arXiv:2411.11767: *Drowning in Documents: Consequences of Scaling Reranker Inference*. Strong text rerankers give diminishing returns as they score more candidates, and eventually degrade. Found during round 2 of the novelty audit (2026-09-25), after B1 showed reranker efficiency falling with K; verified by arXiv and dblp listings only. [S]

**Established:** reranker efficiency falling with candidate depth is reported for text rerankers (Jacob et al. 2024), so B1's version of it in CSI is a measurement in a new setting, not a new phenomenon. Per-query adaptive candidate depth, driven by first-stage evidence, is **prior art in IR**. Applying it to CSI with alignment reranking is a transfer, not a new method. It is worth publishing only if it measurably moves the CSI accuracy/computation frontier.

## 7. Benchmarks

- **Da-TACOS** (Yesiler et al., ISMIR 2019): 15,000-track benchmark with 1,000 cliques of 13 plus 2,000 noise tracks, pre-extracted features and no audio. [R]
- **MIREX CSI**: 2024 used an in-house set; 2025 a new in-house set (search snippet: 10,000 tracks, 80 works). [S] MIREX numbers are **not comparable** to Da-TACOS numbers (different data and protocol). The user-supplied values (ByteCover2 0.877 in 2024; MuCrossCover3 0.958 in 2025, flagged for a possible data issue) are **[U]** and are not cited.

## Where this project fits

| area | status in the literature | what this project can add |
|---|---|---|
| classical alignment, key rotation, DTW | established (2008–) | nothing new; implementation only |
| learned global embedding for CSI | established, far stronger systems exist | nothing new; a small, deterministic baseline |
| two-stage retrieve-then-rerank CSI | established (ByteCover3 2023, LIVI 2026) | not the architecture itself |
| hubness correction in CSI | established (Seo 2022 and others) | benchmark-scale, held-out-calibrated **evidence** (existing) |
| adaptive candidate depth | established in IR (2016–2024) | whether it helps **CSI with alignment reranking**: open, experiment E1 |
| **systematic measurement of the Stage-1 coverage bottleneck in two-stage CSI** | *not found by search* | **the most defensible empirical contribution**: exact coverage-vs-K ceilings, a failure decomposition, factorising Hit@1 into coverage × efficiency across encoders, and development-vs-benchmark transfer |

The last row is based on absence of search results, not on a systematic review. `novelty_audit.md` treats it accordingly.
