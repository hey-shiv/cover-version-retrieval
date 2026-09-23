# Literature check: hubness in alignment-based cover song identification

**Purpose.** Decide whether the project's main finding — hubness correction of DTW alignment scores for cover song identification (CSI), motivated by tonal dispersion — is novel enough for a paper, and position our numbers against published Da-TACOS results.

**Date of search:** 2026-09-23. Author names, volumes and pages were re-checked against Crossref records; one first-author initial was corrected during that check. **Scope:** a focused novelty check, not a full systematic review; no PRISMA flow is claimed.

## Search log

| source | how | queries | raw output |
|---|---|---|---|
| Crossref | `query.bibliographic`, 12 rows each | Q01–Q12 (below) + targeted title lookups | `sources/Q*.json`, `sources/targeted_1.json` |
| arXiv | `export.arxiv.org` API, AND-joined terms | Q01–Q12 | `sources/Q*.json` |
| OpenAlex | polite pool, sorted by citations | Q01–Q12 | `sources/Q*.json` — **discarded**: citation sort surfaced off-topic high-citation papers |
| Semantic Scholar | DOI lookups for abstracts | 2 papers | inline |
| Web search | 4 targeted queries | hubness + CSI, mutual proximity + version ID, Aucouturier & Pachet, SuCo | inline |
| Full text read | PDFs | Seo 2022, Da-TACOS 2019, MOVE 2020, Re-MOVE 2020, ByteCover 2021 | — |

Queries Q01–Q12: hubness music similarity · mutual proximity hubness audio · hubness nearest neighbors high-dimensional data · cover song identification score normalization · cover song identification hubness · chroma alignment cover song identification dynamic time warping · cross-domain similarity local scaling word translation · Da-TACOS cover song identification benchmark · musical version identification embedding · cover song detection cross recurrence quantification Qmax · query bank normalization hubness retrieval · local scaling shared nearest neighbors hub reduction.

Every reference below was checked against Crossref (title match ratio 1.00 and DOI), arXiv, jmlr.org, or the ISMIR proceedings (`sources/verification.json`).

## Findings by question

### 1. Is hubness in music similarity established? — Yes, thoroughly.

- **Aucouturier & Pachet (2008)** found that audio similarity measures produce a scale-free distribution of false positives: the same "hub" songs recur regardless of the query. This is the origin of the hubness problem in MIR.
- **Radovanović, Nanopoulos & Ivanović (2010)** explained hubness as a consequence of high dimensionality: points close to the data mean become hubs.
- **Schnitzer, Flexer, Schedl & Widmer (2012)** introduced mutual proximity (MP) and local scaling as hub reduction methods.
- **Flexer, Schnitzer & Schlüter (2012)** meta-analysed hubness across 17 MIREX audio similarity algorithms (ISMIR best paper).
- Follow-ups: Schnitzer, Flexer & Tomašev (2014) on multimedia retrieval; Feldbauer & Flexer (2019) comparing reduction methods; `scikit-hubness` (Feldbauer, Rattei & Flexer, 2020); Flexer et al. (2018) framing hubness as algorithmic bias.

### 2. Has hubness been addressed specifically for CSI? — Yes, at both levels we would claim.

- **Frame level — Seo (2022), IEICE Trans.** Normalizes the frame-level binary similarity matrix of OTI + Smith-Waterman alignment (Serrà et al. 2008) by a per-query-frame hubness score. Evaluated on covers80 (160 songs) and an in-house 1,000-song set, with CLP and CREMA chroma. MAP gains of about 3–6 points. It reports that non-cover distance correlates with query hubness at −0.71. The normalization constant λ is swept on the evaluation sets themselves; no held-out calibration.
- **Song level — Li & Chen (2018), Applied Sciences.** Mutual proximity applied to fused track-by-track similarity graphs "to reduce the bad influence caused by the hubness phenomenon", after similarity-network fusion and diffusion. Covers80, Covers40, SHS.
- **Subsequence level — Hu & Chen (2019), IEEE/ACM TASLP (SuCo).** Hubness reduction on feature-subsequence communities, plus network enhancement.
- Related post-processing: Serrà, Zanin, Herrera & Serra (2012) exploit community structure in cover song networks.

### 3. Is "tonally static tracks become hubs" known? — The mechanism is; the song-level, benchmark-scale evidence is not.

Radovanović et al. (2010) give the general mechanism (proximity to the data mean). Seo (2022) states the frame-level version explicitly: chroma vectors near a cluster mean become hubs. Our result — that a song-level statistic (tonal dispersion) predicts how often a track is a top-10 false positive (Spearman −0.77 over 120 candidates), and that a probe-based reference recovers it (−0.81 / −0.90) — is a song-level instantiation. We found no prior quantification of this on Da-TACOS or any comparable scale.

### 4. Is our probe-based correction methodologically new? — No.

- **CSLS (Conneau et al., 2018)** penalizes each candidate by its mean similarity to its k nearest neighbours; our correction is its candidate-side term.
- **QB-Norm (Bogolin et al., 2022)** re-normalizes similarities using a *querybank* of training queries, explicitly "without concurrent access to any test set queries". Our fixed probe set of 200 training tracks is the same idea.
- Mutual proximity (Schnitzer et al. 2012) is the established MIR alternative.

### 5. Positioning on the Da-TACOS benchmark (13,000 queries x 15,000 candidates)

| system | input | MAP | MR1 | source |
|---|---|---|---|---|
| FTM2D | HPCP | 0.126 | 207 | Yesiler et al. 2019, Table 2 |
| SiMPle | HPCP | 0.165 | 358 | Yesiler et al. 2019 |
| Dmax | HPCP | 0.292 | 155 | Yesiler et al. 2019 |
| **Qmax** (Serrà et al. 2009) | **HPCP** | **0.333** | 119 | Yesiler et al. 2019 |
| Qmax | CREMA | 0.365 | 113 | Yesiler et al. 2019 |
| EarlyFusion | HPCP | 0.426 | 116 | Yesiler et al. 2019 |
| MOVE | CREMA | 0.507 | 40 | Yesiler, Serrà & Gómez 2020 |
| Re-MOVE | CREMA | 0.534 | 38 | Yesiler, Serrà & Gómez 2020 |
| ByteCover | CQT | 0.714 | 23 | Du et al. 2021 |
| *ours: classical 96 frames* | HPCP | 0.084 | 528 | this repo |
| *ours: classical 96 frames + hub corr.* | HPCP | 0.136 | 331 | this repo |
| *ours: hybrid + hub corr. (best learned)* | HPCP | 0.122 | 205 | this repo |

The protocol matches ours: 1,000 cliques of 13 plus 2,000 unqueried noise tracks. Caveats: ByteCover was trained on SHS100K after removing 8,373 overlapping recordings, and learned systems use far more training data. Re-MOVE released a Da-TACOS training set of 83,904 songs in 14,499 works (CREMA-PCP), about 17x what our encoder saw.

**Our uncorrected alignment baseline (0.084) is roughly 4x weaker than Qmax on the same HPCP input (0.333).** The cause is our 96-frame pilot resolution and whole-query subsequence DTW, not the data. A hubness gain measured on a weak baseline may shrink or vanish on a strong one; our own 384-frame development result already hints at this (D-018).

### 6. Are small development protocols known to mislead in CSI? — Acknowledged, not systematically shown.

MOVE (Yesiler et al. 2020) explicitly cautions that differences on the small YouTubeCovers set "may not be significant". The general survey (Yesiler, Doras, Bittner, Tralie & Serrà 2021) discusses evaluation practice. Our observation — two ordering reversals between a 20-query development protocol and the benchmark — is anecdotal evidence (n = 1 project), not a methodology contribution by itself.

## Novelty verdict

| claim | status |
|---|---|
| Hubness exists in music similarity | **known** (2008–2012) |
| Hubness hurts alignment-based CSI; reducing it helps | **known** (Seo 2022; Li & Chen 2018; Hu & Chen 2019) |
| Candidate-side / querybank normalization of retrieval scores | **known** (CSLS 2018, QB-Norm 2022, MP 2012) |
| Tonal dispersion predicts hubs under DTW, at song level | **weakly new** — a song-level quantification of a known mechanism |
| Hubness correction evaluated at Da-TACOS benchmark scale, with held-out calibration and work-level CIs | **new as evidence**, but on a baseline far below Qmax |
| Shortlist recall as the binding constraint of embedding + alignment hybrids | **known in IR generally**; our measurements are a case study |

**Bottom line.** The idea is not new. What is new is *evidence*: the first benchmark-scale, properly calibrated measurement of hubness correction for alignment-based CSI that we could find, plus a song-level explanatory variable. That supports a short paper only if the result survives on a **strong** baseline.

## What a publishable version needs

1. **A strong alignment baseline:** Qmax on HPCP at native resolution (Da-TACOS reports 0.333). The `acoss` implementation can be run as an external tool (not copied; AGPL-3.0).
2. **Head-to-head with the existing corrections** on that baseline: Seo's frame-level normalization, mutual proximity (via `scikit-hubness`), and CSLS/QB-Norm-style probe correction. The question becomes *which level of correction helps, and are they complementary?* — that comparison does not exist in the literature.
3. **The dispersion analysis on the full benchmark**, not only on 120 development candidates.
4. **One pre-registered benchmark evaluation** of the final comparison, with the four earlier exploratory runs disclosed.
5. Optionally, **a second dataset** (e.g. Covers80, which Seo used, for direct comparability).

Items 1–2 are the core. If the gain disappears on Qmax, that is itself a reportable negative result, but a weaker paper.

## References (verified)

- Aucouturier, J.-J., & Pachet, F. (2008). A scale-free distribution of false positives for a large class of audio similarity measures. *Pattern Recognition*, 41(1), 272–284. https://doi.org/10.1016/j.patcog.2007.04.012
- Bogolin, S.-V., Croitoru, I., Jin, H., Liu, Y., & Albanie, S. (2022). Cross modal retrieval with querybank normalisation. *CVPR*, 5184–5195. https://doi.org/10.1109/CVPR52688.2022.00513
- Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data. *ICLR*. arXiv:1710.04087
- Du, X., Yu, Z., Zhu, B., Chen, X., & Ma, Z. (2021). ByteCover: Cover song identification via multi-loss training. *ICASSP*, 551–555. https://doi.org/10.1109/ICASSP39728.2021.9414128
- Feldbauer, R., & Flexer, A. (2019). A comprehensive empirical comparison of hubness reduction in high-dimensional spaces. *Knowledge and Information Systems*. https://doi.org/10.1007/s10115-018-1205-y
- Feldbauer, R., Rattei, T., & Flexer, A. (2020). scikit-hubness: Hubness reduction and approximate neighbor search. *JOSS*. https://doi.org/10.21105/joss.01957
- Flexer, A., Dörfler, M., Schlüter, J., & Grill, T. (2018). Hubness as a case of technical algorithmic bias in music recommendation. *ICDMW*. https://doi.org/10.1109/ICDMW.2018.00154
- Flexer, A., Schnitzer, D., & Schlüter, J. (2012). A MIREX meta-analysis of hubness in audio music similarity. *Proc. ISMIR*. https://ismir2012.ismir.net/event/papers/175_ISMIR_2012.pdf
- Hu, J., & Chen, N. (2019). Enhanced feature summarizing for effective cover song identification. *IEEE/ACM TASLP*, 27(12), 2113–2126. https://doi.org/10.1109/TASLP.2019.2942157
- Khosla, P., et al. (2020). Supervised contrastive learning. *NeurIPS*. arXiv:2004.11362
- Li, M., & Chen, N. (2018). A robust cover song identification system with two-level similarity fusion and post-processing. *Applied Sciences*, 8(8), 1383. https://doi.org/10.3390/app8081383
- Müller, M. (2021). *Fundamentals of Music Processing* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-69808-9
- Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *JMLR*, 11, 2487–2531. https://jmlr.org/papers/v11/radovanovic10a.html
- Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). Local and global scaling reduce hubs in space. *JMLR*, 13, 2871–2902. https://jmlr.org/papers/v13/schnitzer12a.html
- Schnitzer, D., Flexer, A., & Tomašev, N. (2014). A case for hubness removal in high-dimensional multimedia retrieval. *ECIR*. https://doi.org/10.1007/978-3-319-06028-6_77
- Seo, J. S. (2022). Pairwise similarity normalization based on a hubness score for improving cover song retrieval accuracy. *IEICE Trans. Inf. & Syst.*, E105-D(5), 1130–1134. https://doi.org/10.1587/transinf.2021EDL8075
- Serrà, J., Gómez, E., Herrera, P., & Serra, X. (2008). Chroma binary similarity and local alignment applied to cover song identification. *IEEE TASLP*, 16(6), 1138–1151. https://doi.org/10.1109/TASL.2008.924595
- Serrà, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification. *New Journal of Physics*, 11, 093017. https://doi.org/10.1088/1367-2630/11/9/093017
- Serrà, J., Zanin, M., Herrera, P., & Serra, X. (2012). Characterization and exploitation of community structure in cover song networks. *Pattern Recognition Letters*. https://doi.org/10.1016/j.patrec.2012.02.013
- Smucker, M. D., Allan, J., & Carterette, B. (2007). A comparison of statistical significance tests for information retrieval evaluation. *CIKM*. https://doi.org/10.1145/1321440.1321528
- Yesiler, F., Doras, G., Bittner, R. M., Tralie, C. J., & Serrà, J. (2021). Audio-based musical version identification: Elements and challenges. *IEEE Signal Processing Magazine*. https://doi.org/10.1109/MSP.2021.3105941
- Yesiler, F., Miron, M., Serrà, J., & Gómez, E. (2022). Assessing algorithmic biases for musical version identification. *WSDM*. https://doi.org/10.1145/3488560.3498397
- Yesiler, F., Serrà, J., & Gómez, E. (2020). Accurate and scalable version identification using musically-motivated embeddings. *ICASSP*. https://doi.org/10.1109/ICASSP40776.2020.9053793
- Yesiler, F., Serrà, J., & Gómez, E. (2020). Less is more: Faster and better music version identification with embedding distillation. *Proc. ISMIR*. arXiv:2010.03284
- Yesiler, F., Tralie, C., Correya, A., Silva, D. F., Tovstogan, P., Gómez, E., & Serra, X. (2019). Da-TACOS: A dataset for cover song identification and understanding. *Proc. ISMIR*, 327–334. https://archives.ismir.net/ismir2019/paper/000038.pdf
