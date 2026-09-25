# Research story (as the evidence supports it)

Each step names its evidence tier (see `repository_audit.md`).

1. **Initial hypothesis.** A compact learned global embedding can serve as a cheap coarse stage. Structure-aware alignment then reranks a short list of K = 30 candidates. At benchmark scale that is 390,000 alignments instead of 1.95 × 10⁸ (0.20%).

2. **Development result (DEV).** On 20 queries × 119 candidates with one relevant item each, the run-1 hybrid (0.262 MAP) edged out exhaustive alignment (0.248). The paired interval included zero: +0.014 [−0.009, +0.042].

3. **Benchmark reversal (FROZEN-RUN, run 1).** On 13,000 × 15,000 with 12 relevant each, exhaustive alignment scored 0.0835 and the hybrid 0.0185, a factor of 4.5 behind. Paired difference: −0.065 [−0.071, −0.059].

4. **Diagnosis.**
   - **Stage-1 shortlist recall.** Recall at K = 30 was 0.021 on the benchmark, against 0.75 on development. Only 18.7% of queries had any cover in the shortlist (X2), and no reranker can move a cover that isn't there. Hit@1 factorises exactly into coverage × reranker efficiency.
   - **Hubness.** Tonally static candidates match everything under DTW. Tonal dispersion correlates with top-10 false-positive counts (Spearman −0.77 for classical, −0.57 for hybrid, DEV). A probe reference computed from 200 training queries correlates with dispersion at −0.81 (96 frames) and −0.90 (384 frames).

5. **Interventions** (each locked on calibration or validation works, never on the benchmark):
   - **Training data**, 1,500 → 4,780 works: validation MAP 0.181 → 0.319, benchmark Stage 1 0.010 → 0.034 (run 2).
   - **Hub correction**: exhaustive classical 0.0835 → 0.1357 (run 2); hybrid gains at every run (X5).
   - **Alignment resolution**, 96 → 384 frames: DEV classical 0.248 → 0.384. On the benchmark at K = 30, +0.0047 [+0.0042, +0.0053] AP (run 3 vs run 2).
   - **Training length**, 60 → 150 epochs: validation 0.319 → 0.431, benchmark Stage 1 0.034 → 0.076 (run 4).

6. **Result.** The run-4 hybrid with hub correction reaches 0.1217 against 0.1357 for the corrected exhaustive 96-frame system, narrowing the ratio from 4.5× to 1.11×, at 38 ms per query of reranking against 2.63 h of alignment for all queries. About 80% of the log-gain in Hit@1 across the four runs came from coverage (X2).

7. **The remaining bottleneck at K = 30.** Shortlist recall is 0.136. 47.8% of queries have no cover among the 30 aligned candidates, against 15.6% misranked.

8. **Beyond K = 30 (LOCAL-FULL, A1/B1), which the frozen runs could not show.**
   - MAP keeps rising to K = 500 (0.212).
   - The dominant failure moves from *excluded* to *misranked* between K = 100 and K = 200.
   - Reranker efficiency falls from 85% to 57%.
   - Hub correction matters more as K grows. Uncorrected, 34% of rank-1 false positives at K = 500 come from the top decile of hub reference (P1).

9. **Two registered interventions on the coarse stage.**
   - **Per-query adaptive K fails** (E1), although difficulty is predictable (AUROC 0.76, D1).
   - **Fusing global and window embeddings succeeds end to end** (A2): +0.017 MAP at K = 30, and above zero at every K.

10. **Protocol lesson (P2).** In four paired comparisons the development protocol was inconclusive every time (every interval included zero), and in three its point estimate had the wrong sign. The benchmark was decisive every time.

**What remains unresolved.**
- The system is weak in absolute terms (Qmax reports 0.333 on the same input).
- Exhaustive alignment at 384 frames was never run.
- Every finding comes from one system on one dataset.
