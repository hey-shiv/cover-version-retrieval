# Research question

## Why the question moved

The obvious framing, "TCN embedding plus DTW reranking for cover song identification", is not a research gap:
- Two-stage retrieve-then-rerank CSI is established: ByteCover3 (2023) uses ANN plus reranking; LIVI (2026) uses global retrieval plus MaxSim over local embeddings.
- Hubness correction for CSI is prior art (Seo 2022; Li & Chen 2018; Hu & Chen 2019).
- Adaptive candidate depth is prior art in IR (Culpepper, Clarke & Lin 2016; ranked-list truncation, SIGIR 2024).

See [`literature_review.md`](literature_review.md).

What the repository has that those papers do not report is **a controlled, measured account of the bottleneck between the two stages**, on a public benchmark:
- per-query first-cover ranks for 13,000 queries under several encoders and rerankers;
- exact structural knowledge of what a K-candidate reranker can and cannot change.

So the question is about the **interface between the stages**, not about a new model.

## Central hypothesis

> In two-stage cover song retrieval, accuracy at the head of the ranking is bounded by the coverage of the first stage, and on a realistic benchmark that bound, not the quality of the structure-aware reranker, is the dominant limitation at any affordable shortlist size.

## Research questions

| RQ | question | status | evidence |
|---|---|---|---|
| **RQ1** | How does shortlist size K control coverage, recall and final quality, and at what cost in alignments per query? | coverage exact at every K (done); final quality needs reranking at K ≠ 30 | X1 (done); **A1** (local) |
| **RQ2** | Why do some queries need very large K? | partly answerable: the rank distribution is heavy-tailed (done); *why* needs per-query signals | X1, X4 (done); **A1 + D1** |
| **RQ3** | Can shortlist size be chosen per query, from Stage-1 evidence alone, and beat fixed K at equal cost? | upper bound known (done); realisable gain open | X4 bound (done); **D1, E1** |
| **RQ4** | Which failure mode dominates at each budget? | at K = 30, exact (done); across K needs reranking at every K | X2 (done); **B1** |
| **RQ5** | Can cheap Stage-1 changes (test-time rotations, windowed multi-vector MaxSim, fusion) raise coverage at fixed K without retraining? | open | **F1** |
| **RQ6** | How much error is attributable to hubness, and does hubness act in Stage 1 or only in reranking? | reranking side measured (done); Stage-1 side open | X2, X5 (done); **F1 (s1_hub), B1** |
| **RQ7** | What is the whole system's accuracy/computation trade-off? | measured at three operating points (done); full curve needs A1 | X3 (done); **A1** |
| **RQ8** | Does the small development protocol predict benchmark behaviour of a two-stage system? | answered: no, because its shortlist recall is near 1 (done) | X6 (done) |

## What is already established (ARTIFACT-ANALYSIS, 13,000 benchmark queries, work-level 95% CI)

These results come from committed per-query files and are exact up to bootstrap uncertainty; see `results/X*/`.

1. **Coverage is the binding constraint at K = 30.** With the best encoder (run 4), **47.8% [46.2, 49.5]** of queries have no cover among the 30 aligned candidates, so no reranker can help them.
2. **Hit@1 factorises exactly as coverage × reranker efficiency**, and the gains across the four runs came mainly from coverage:
   - coverage: 0.187 → 0.522;
   - efficiency, Hit@1 given at least one cover in the shortlist: 0.546 → 0.701;
   - about 80% of the log-improvement in Hit@1 came from coverage.
3. **Hubness correction acts only inside the shortlist.** It converts 24.9% [23.0, 26.8] of the "cover present, not at rank 1" failures into successes, and turns a success into a failure for 1.2% [1.0, 1.4] of all queries (about 3.6% of the uncorrected successes). By construction it cannot change the 47.8%.
4. **Learned embedding and exhaustive alignment are complementary as candidate generators.**
   - Corrected exhaustive alignment has higher coverage at small K: K = 10, 0.481 vs 0.380.
   - They tie at K = 50 (0.600 vs 0.599). The embedding has higher coverage beyond: K = 100, 0.706 vs 0.661; K = 1000, 0.954 vs 0.902. Paired work-level test in X1.
   - So the division of labour (embedding for recall, alignment for precision) is supported by the data, not only by cost.
5. **The first-cover rank is heavy-tailed.** 29% of queries need K > 100 and 9% need K > 500 (run 4). An oracle allocating K per query would match fixed-K = 30 coverage with **3.9** alignments per query on average. This is an upper bound, not a method.
6. **The development protocol cannot test this bottleneck.** Its shortlist recall at K = 30 is 0.75–0.90, against 0.02–0.14 on the benchmark. The sign of the reranking effect disagreed with the benchmark in 2 of 4 configurations.

## Refined thesis (provisional until A1/D1/E1/F1)

> *A scalable cover-song retrieval system is limited by the coverage of its first stage: structure-aware alignment cannot recover covers excluded from the shortlist. On Da-TACOS we measure this bottleneck exactly across shortlist sizes and encoders, decompose every query's outcome into coverage and reranking failures, show that first-stage training — not reranking — produced most of the observed improvement, and test whether first-stage evidence can allocate alignment work per query.*

The last clause stays in the thesis **only if E1 shows a gain over fixed K at equal cost**. Otherwise it becomes a reported negative result, and the thesis ends at the decomposition. If F1 shows that a cheap Stage-1 change raises coverage, that becomes a second positive claim.

## Contribution type (decision gate, see `novelty_audit.md`)

Current evidence supports **Option C, an empirical analysis paper** on the coverage bottleneck, hubness and computation in two-stage CSI. **Option B**, an empirical/system paper with a positive intervention, requires E1 or F1 to succeed. **Option A**, a new method, is not supported.
