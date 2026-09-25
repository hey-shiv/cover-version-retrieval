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

Status as of 2026-09-25, after the local runs (A1, F1, H1; LOCAL-FULL) and their cloud analyses (B1, D1, E1, F1s, H1s; C1 post-hoc).

| RQ | question | answer | evidence |
|---|---|---|---|
| **RQ1** | How does shortlist size K control coverage, recall and final quality, and at what cost? | MAP rises monotonically with K and has not saturated at 500 (hub-corrected hybrid 0.085 at K = 5, 0.122 at 30, 0.160 at 100, 0.212 at 500). Rerank cost is 0.72 ms per pair on the local laptop, so about 22 ms/query at K = 30 and 362 ms/query at K = 500. | A1 (LOCAL-FULL); X1 |
| **RQ2** | Why do some queries need very large K? | Only partly answered. The first-cover rank is heavy-tailed (X1, X4). Label-free Stage-1 signals predict "no cover in the top 30" with AUROC 0.76 [0.75, 0.77], so hardness is partly visible in the score distribution; the strongest single signal is top-100 entropy (0.73). The *cause* (e.g. structure, transposition, arrangement) is not tested. | X1, X4, D1 |
| **RQ3** | Can K be chosen per query and beat fixed K at equal cost? | **No** (registered negative). 1 of 10 cells meets the criterion; 5 have ΔAP > 0 with CI excluding zero (at most +0.0072), but every one of them lowers Hit@1 (up to −4.4 points across cells). The fitted policies are triage: fewer alignments for predicted-hard queries. A trade-off, not a Pareto gain. | D1, E1 |
| **RQ4** | Which failure mode dominates at each budget? | Class A (no cover in the shortlist) dominates up to K = 50 (40.0% vs 20.2%); the two are close at K = 100 (29.4% vs 26.9%); class B (misranked) dominates from K = 200 (33.1% vs 19.5%). Reranker efficiency falls from 85.1% at K = 5 to 56.7% at K = 500. | B1 |
| **RQ5** | Can cheap Stage-1 changes raise coverage at fixed K without retraining, and does that help end to end? | **Yes, for one variant, and the gain survives reranking.** Global + 3-window fusion: +3.8 [+3.2, +4.5] points coverage@30 (F1). In the registered end-to-end test (A2) hub-corrected MAP rises at every K: +0.017 [+0.015, +0.019] at K = 30 (0.122 → 0.139, +14%), +0.014 [+0.013, +0.016] at K = 100, +0.010 at K = 500; Hit@1 +3.0 points at K = 30. With fusion, K = 20 matches the old K = 30. Test-time rotations: +1.4 coverage points (not tested end to end). Windows alone: −1.5. Training scale matters far more: 60 epochs −13.2, 1,500 works −33.5 points. | F1, F1s, A2, A2s |
| **RQ6** | How much error is due to hubness, and where does it act? | Inside the shortlist, and increasingly with K: hub-corrected minus uncorrected MAP is +0.001 at K = 5, +0.009 at 30, +0.043 at 500; the uncorrected "hub at rank 1" class grows from 1.2% to 11.2% of queries, the corrected one from 0.6% to 2.8%. At Stage 1, calibration chose λ = 0 (no correction); weak evidence that hubness does not limit coverage. | X2, X5, B1, F1 |
| **RQ7** | What is the system's accuracy/computation trade-off? | Full curve in A1 (table t4, fig 8). Post-hoc (C1, not registered, 96 vs 384 frames, different machines): the hub-corrected hybrid ties exhaustive corrected alignment on MAP at K = 50 (+0.0018 [−0.0046, +0.0081]) and beats it from K = 100 (+0.0243 [+0.0179, +0.0304]) with 100 instead of 14,999 alignments per query; on Hit@1 it is ahead from K = 30. | A1, X3, C1 |
| **RQ8** | Does the development protocol predict benchmark behaviour? | No. Shortlist recall 0.75–0.90 vs 0.02–0.14; sign of the reranking effect wrong in 2 of 4 configurations; key handling looked useless on development (point estimate favoured none, CI included zero) and is essential on the benchmark (−0.039 AP without it, H1). | X6, H1 |

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

## Revised thesis (after the local runs)

> *In a two-stage cover song retrieval system, the first stage bounds accuracy at small shortlists and the reranker bounds it at large ones. On Da-TACOS we measure both bounds exactly across K from 5 to 500, locate where the bottleneck moves from coverage to reranking (K ≈ 100–200 for this system), attribute most past gains to the first stage, and show that per-query adaptive K — although difficulty is predictable — trades Hit@1 for MAP rather than improving both. Fusing global and window embeddings is the one cheap first-stage change that raised coverage, and in a registered end-to-end test it improved MAP and Hit@1 at every K, most where coverage is the bottleneck.*

What changed from the provisional thesis, and why:
- "Coverage is the dominant limitation at any affordable K" is **withdrawn**. It holds for K ≤ 50–100 only (B1). Whether K = 100–500 is "affordable" depends on the deployment; at 0.72 ms/pair it is 72–362 ms per query here.
- The adaptive-allocation clause is now a **reported negative result** (E1), as pre-committed.
- The fusion result is now an **end-to-end** result (A2, registered criterion met). It is a transfer of known late-interaction ideas, not a new method.

## Contribution type (decision gate, see `novelty_audit.md`)

Round 2 (after the local runs): **Option C, an empirical analysis paper**, remains the supported contribution, now with the full K sweep, the bottleneck shift and a registered negative result. Round 3 (after A2): **Option C with one registered positive intervention.** E1 failed; the F1 → A2 fusion result is supported end to end. It is too simple, and too close to prior late-interaction work, to carry a method paper (Option A), so it is presented as a supported intervention inside the analysis.
