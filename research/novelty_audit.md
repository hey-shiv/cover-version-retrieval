# Novelty audit

Round 1 (before the local experiments) is kept unchanged below; round 2 (after A1, F1, H1 and their analyses) is at the end.

## Round 1

This is conservative by design. It will be repeated after A1, D1, E1 and F1 (round 2).
Sources: [`literature_review.md`](literature_review.md). Many entries are verified only by search results; see the verification tags there.

## 1. What is genuinely new?

Nothing at the level of **method**. At the level of **evidence**, possibly the following, on a public benchmark (Da-TACOS, 13,000 queries):

- an exact coverage-versus-K curve for the first stage of a two-stage CSI system, across three encoders, with work-level intervals;
- a complete per-query decomposition of outcomes into "not in the shortlist", "in the shortlist but misranked" and "rank 1", with the exact identity Hit@1 = coverage × reranker efficiency. It shows that most of a 3.6× Hit@1 gain came from coverage;
- a measured crossover between a learned embedding and exhaustive (hub-corrected) alignment used as candidate generators: alignment is better at small K, the two tie at K ≈ 50, and the embedding is better beyond (paired test in X1);
- a demonstration that a small development protocol cannot probe this bottleneck (shortlist recall 0.75–0.90 on development against 0.02–0.14 on the benchmark), with two sign reversals.

Our searches found no CSI paper reporting these quantities. That is **absence of search results, not proof of absence**: a full systematic search, with access to the ISMIR, ICASSP and TASLP proceedings, is required before claiming it.

## 2. Recombinations of existing ideas

- Two-stage embedding-then-alignment retrieval: ByteCover3, LIVI, and cascades in IR.
- Probe-based hubness correction: CSLS / QB-Norm, applied to CSI alignment scores, where Seo 2022 and others had already applied hubness correction.
- Adaptive shortlist size (E1, if it works): Culpepper et al. 2016 and ranked-list truncation, transferred to CSI.
- Windowed multi-vector MaxSim (F1): ColBERT / LIVI-style late interaction on a frozen encoder.

## 3. Engineering, not research

TCN encoder, SupCon training, batched subsequence DTW, the deterministic pipeline, the manifests, the reproducibility controls, and the website.

## 4. Empirical findings (the contribution, if any)

Items 1–6 of `research_question.md`, "What is already established", plus A1–F1 once run.

## 5. What a skeptical reviewer attacks

- **"The baselines are weak."** True. The best system (0.136 MAP) is about 2.5× below Qmax on the same input (0.333) and about 5× below ByteCover (0.714). The bottleneck may look different with a strong Stage 1. *Mitigation:* frame the result as a measurement method and a finding about *this regime*. Show that the finding held across three encoders of increasing strength, where coverage rose 0.19 → 0.52 while the conclusion held. Note that stronger systems with published per-query ranks could be re-analysed with the same scripts.
- **"This is obvious; everyone knows recall bounds reranking."** The bound is obvious. Its *size* at realistic K, how gains split between coverage and efficiency, and the embedding/alignment crossover are not reported for CSI. The paper must present them as measurements, not discoveries of the principle.
- **"Adaptive K is known in IR."** Yes. Any E1 result must be framed as "does it transfer to alignment-based CSI", with the IR prior art cited prominently.
- **"Single dataset, feature-only, 2019."** True; stated as a limitation.
- **"Cross-fitting on the benchmark is tuning on the test set."** It is out-of-fold and work-disjoint, pre-registered, and disclosed. The alternative, the 10-work calibration protocol, is shown to be unrepresentative (X6).

## 6. Prior papers making similar claims

- Culpepper, Clarke & Lin (2016): per-query candidate depth for multi-stage text retrieval.
- ByteCover3 and LIVI: two-stage CSI motivated by efficiency.
- Bertin-Mahieux & Ellis (2012): the scalability motivation.

None of these, as far as this search found, measures the coverage bottleneck in CSI.

## 7. What differentiates this project

Exactness and openness:
- Every number is reproducible from committed per-query files.
- The structural property is checked on every query.
- Development-versus-benchmark transfer is measured, not assumed.
- Negative results, including the original 4.5× reversal, are reported.

## Decision gate (round 1)

**Option C, an empirical analysis paper,** is supported by the evidence already in hand. Options B and A are **not yet** supported. Round 2 decides between C and B, based on:
- E1: adaptive K beating fixed K at equal cost, with the interval excluding zero;
- F1: a cheap Stage-1 change raising coverage@30, with the interval excluding zero.


## Round 2 (2026-09-25, after the local runs)

**New prior art found in this round.** Jacob et al. (2024, arXiv:2411.11767, [S]) report that text rerankers give diminishing and eventually negative returns as they score more candidates. B1's falling reranker efficiency (85.1% at K = 5 → 56.7% at K = 500) is the CSI version of that pattern. It is a **measurement in a new setting, not a new phenomenon**, and the paper now cites it.

**Claim by claim.**

| candidate claim | status | why |
|---|---|---|
| exact coverage ceilings and full K sweep for two-stage CSI, 13,000 queries | **supported (measurement)** | A1 reproduces the frozen benchmark exactly and sweeps K = 5–500; no CSI paper found reporting this |
| the bottleneck moves from coverage to reranking at K ≈ 100–200 | **supported (measurement)**; the mechanism has IR precedent | B1; Jacob et al. 2024 |
| hub correction's value grows with K | **supported (measurement)**; hub correction itself is prior art | B1, A1 |
| adaptive K for CSI | **negative result** | E1: 1/10 cells meet the criterion; all MAP gains cost Hit@1. Adaptive depth is IR prior art (Culpepper et al. 2016), so the value is the negative transfer result |
| global + window fusion raises Stage-1 coverage | **supported at the coverage level only** | F1 +3.8 [+3.2, +4.5] @30; multi-vector late interaction is prior art (ColBERT, LIVI); end-to-end effect unknown until A2 |
| two-stage hybrid matches exhaustive alignment at 1/150 of the alignments | **post-hoc, not claimable as a finding** | C1 was chosen after seeing A1 and mixes 96/384 frames and machines; reported as exploratory |
| a development protocol misleads design | **supported**, now with a third case (key handling) | X6, H1 |

**Decision gate (round 2).** **Option C, an empirical analysis paper,** is supported and stronger than in round 1. **Option B is not supported yet**: its only candidate, window fusion, is unproven end to end. A2 (one local run, ~80 min) decides whether B becomes available. No method novelty is claimed.
