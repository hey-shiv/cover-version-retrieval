# Novelty audit (paper phase)

Builds on `research/novelty_audit.md` (rounds 1–3) and `related_work.md`. Categories:
1. genuinely methodological;
2. engineering implementation;
3. empirical finding;
4. evaluation methodology;
5. analysis / failure characterisation;
6. reproducibility contribution;
7. previously established by literature.

| candidate contribution | category | verdict | why |
|---|---|---|---|
| TCN global embedding for CSI | 2, 7 | not claimed | Generic TCN and SupCon. Learned global CSI embeddings are established (MOVE, ByteCover). |
| two-stage embedding → alignment retrieval | 7 | not claimed | ByteCover3 (ANN → re-ranking with local features), LIVI (global → MaxSim), IR cascades |
| profile-cosine key rotation, subsequence DTW | 2, 7 | not claimed | OTI (Serrà 2008); textbook DTW |
| probe-based hub correction | 7 | not claimed as a method | CSLS, QB-Norm; in CSI, Seo 2022 and Li & Chen 2018 |
| global + window fusion in Stage 1 | 7 (method), 3 (evidence) | claimed only as a registered empirical result | Late interaction (ColBERT, LIVI). The new part is the controlled, pre-registered end-to-end test on a CSI benchmark, and its K-dependence. |
| exact coverage ceilings and the Hit@1 = coverage × efficiency identity | 4, 5 | **claimed** | The identity is elementary. Applying it per query across encoders and K, with work-level intervals, is not reported for CSI in anything we could find. |
| attribution: about 80% of the Hit@1 log-gain came from coverage | 3 | **claimed** | From committed per-query files (X2) |
| bottleneck shift from exclusion to misranking with K | 3, 5 | **claimed as a CSI measurement** | Reranker degradation with depth is reported in text IR (Jacob et al. 2024). New setting, not a new phenomenon. |
| hub correction's value grows with K; rank-1 false positives concentrate in high-reference candidates | 3, 5 | **claimed** | B1 and P1, full benchmark |
| adaptive K does not beat fixed K at equal cost | 3 (negative) | **claimed as a negative result** | Adaptive depth is IR prior art (Culpepper 2016). The transfer to alignment-based CSI fails in a registered test. |
| a 20-query development protocol cannot observe the bottleneck and misleads in 3 of 4 paired comparisons | 4 | **claimed** | P2, X6. We found no systematic CSI demonstration. |
| reproducibility: locked calibration, WID-disjoint roles, deterministic training, exact reproduction of a frozen run | 6 | claimed as a supporting contribution | A1 reproduces run 4 bit for bit |

**Conclusion.** No methodological novelty. The contribution is empirical and evaluation-methodological: a measured account of the coarse-stage bottleneck in two-stage CSI, the variables that move it, and how evaluation scale changes conclusions.

**Main novelty risk.** A reader of ByteCover3 or LIVI may consider the architecture well understood. The paper must argue that *measuring* the bottleneck is the contribution, and must say plainly that the architecture itself is prior art.
