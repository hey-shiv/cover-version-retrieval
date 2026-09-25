# Related work (paper phase)

Verification levels, as in `citation_audit.md`: **C** = Crossref record stored in `notes/literature/sources/`; **G** = authors' GitHub README; **S** = web-search results only. In this environment the proxy blocked arXiv, dblp, Crossref, Semantic Scholar, ISMIR archives and MIREX, so no primary PDF could be read. Every S item must be read before submission.

## A. Classical cover song identification
- **Chroma / HPCP.** HPCP (Gómez 2006, S) is the pitch-class representation Da-TACOS distributes.
- **Key invariance and alignment.** Serrà et al. (2008, C) combine an optimal transposition index (OTI) with binary chroma similarity and local alignment. Qmax (Serrà, Serra & Andrzejak 2009, S) uses cross-recurrence quantification.
  - Our profile-cosine rotation is the OTI idea applied to time-averaged profiles.
  - Our subsequence DTW (Sakoe & Chiba 1978; Müller 2015; both S) is a whole-query alignment, weaker than Qmax's local alignment.
- **Scalability** has been a concern since at least Bertin-Mahieux & Ellis (2012, S).

## B. Learned version identification
- **MOVE** (ICASSP 2020, C) and **Re-MOVE** (ISMIR 2020, G; the brief's "TASLP 2021" is wrong) learn embeddings from CREMA-PCP.
- **ByteCover** (ICASSP 2021, C), **ByteCover2** (ICASSP 2022, S) and **CoverHunter** (ICME 2023, G/S) learn from CQT with metric or classification losses. CoverHunter adds coarse-to-fine chunk-alignment training.
- **Discogs-VI and Discogs-VINet** (ISMIR 2024, G/S): a 1.9M-version dataset and a CQT-Net model. Weights are public; per-query rankings are not (G).
- **CLEWS** (ICML 2025, S): supervised contrastive learning on weakly labelled segments, for segment-level version matching.
- **TCN encoders** (Bai et al. 2018, S) and **supervised contrastive loss** (Khosla et al. 2020, S) are the generic components our encoder uses. Neither is new here.

## C. Two-stage and local retrieval
- **ByteCover3** (ICASSP 2023, S): local features, a local alignment loss, and a **two-stage pipeline of ANN retrieval followed by re-ranking**.
- **LIVI** (ECIR 2026, S): track-level retrieval, then **MaxSim re-ranking over 30-second local embeddings** (the re-ranking may be only in the extended arXiv version).
- **ColBERT** (SIGIR 2020, S): multi-vector late interaction in text retrieval.
- Our fused global + window Stage 1 (A2) is an instance of this family.

**Consequence.** Two-stage retrieval, local features and late interaction are **prior art**. The paper claims none of them.

## D. Hubness
- **General:** Aucouturier & Pachet (2008, C) on scale-free false positives in audio similarity; Radovanović et al. (2010, S); Schnitzer et al. (2012, S) on mutual proximity and local scaling.
- **Retrieval corrections:** CSLS (Conneau et al. 2018, C) and QB-Norm (Bogolin et al. 2022, C).
- **In CSI:** Li & Chen (2018, C) use mutual proximity; Hu & Chen (2019, C) reduce hubness at feature summarisation; Seo (2022, C) normalises alignment similarity by a hubness score.
- Our correction is a probe-based, candidate-side CSLS-style term. It is **not new**; the evidence we add is its behaviour at 13,000-query scale and how it interacts with shortlist size.

## E. Multi-stage retrieval and evaluation
- **Cascades and per-query cutoffs:** Wang, Lin & Metzler (2011, S); Culpepper, Clarke & Lin (2016, S).
- **Reranker degradation with depth:** Jacob et al. (2024, S) report that text rerankers give diminishing and then negative returns as they score more candidates. Our B1 result, falling reranker efficiency as K grows, is the CSI analogue.
- **Statistics:** Efron & Tibshirani (1993, S); cluster bootstrap by Field & Welsh (2007, S). We resample query works (cliques).

## F. Datasets and benchmarks
- **Da-TACOS** (ISMIR 2019, C/G): the benchmark subset has 15,000 tracks (G). Its structure (1,000 cliques of 13 plus 2,000 noise tracks) matches `benchmark*.json → protocol` in this repository.
- **Discogs-VI** (2024) and **DiVers** (ISMIR 2026, S): DiVers has more than 1.1M versions, built on Discogs-VI plus more than 600k YouTube versions. Its splits are compatible with Da-TACOS and SHS100K. It studies robustness to in-the-wild audio, with CLEWS fine-tuning.
- **Robustness to real-world covers:** Hachmeier & Jäschke (2025, S) find large drops on YouTube covers. Araz et al. (2026, S) benchmark track and version identification jointly.
- **MIREX CSI.**
  - 2024: ByteCover2 ranked first and Discogs-VINet second (G, Discogs-VINet README). No mAP value could be verified.
  - 2025: uses an in-house collection of 10,000 tracks and 80 works (S). **No 2025 results page, "MuCrossCover3", 0.958 figure or data-issue caveat could be found** (U).
  - MIREX collections differ from Da-TACOS, so no MIREX number is compared with ours.

## Published Da-TACOS numbers
The repository's `notes/literature/novelty_assessment.md` records values from Yesiler et al. 2019, Table 2 (Qmax on HPCP 0.333), and from the MOVE, Re-MOVE and ByteCover papers (0.507, 0.534, 0.714). None could be re-read here. The paper uses only Qmax 0.333 (same HPCP input), cited to Da-TACOS, and `citation_audit.md` marks it as **must re-check against the PDF before submission**.

## What is not in the literature (as far as this search could see)
- A per-query decomposition of a two-stage CSI system into coverage and reranker efficiency across shortlist sizes.
- A controlled demonstration that a small development protocol reverses benchmark-scale conclusions.

Both are **absence of search results**, not proof of absence.
