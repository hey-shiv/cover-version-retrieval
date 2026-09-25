# Reviewer audit

## Round 1 (before the local experiments)

A simulated skeptical review of the work as it stands. Items marked **→** are actions; their status is tracked here.

**Novelty: "What is actually new?"**
Measurements, not methods (see `novelty_audit.md`). Acceptable for an analysis paper only if the measurements are framed as such.
**→** In the paper, lead with the decomposition and the crossover, not the architecture.

**Significance: "Why should MIR care?"**
Two-stage CSI is now the standard architecture (ByteCover3, LIVI), and practitioners choose K with no published guidance. The coverage curve and the decomposition tell them where the error is and what fixing it is worth.
**→** Add an explicit "how to choose K" takeaway once A1 exists.

**Method: "Clearly specified?"**
Yes. See `baseline_audit.md`: every equation and setting is traced to code.

**Evaluation: "Fair comparisons?"**
Runs differ in several settings. Paired comparisons state exactly what differs (X5). The best hybrid and the best classical system come from different runs (run 4 vs run 2).
**→** Never place them in one leaderboard row without the run label.

**Leakage: "Could benchmark tuning have leaked?"**
- Frozen runs: no. Every setting was locked from validation or calibration works (D-017–D-021), and no run was repeated.
- New fitted components (D1, E1): work-disjoint cross-fitting, pre-registered.
- K itself is an independent variable, reported across its whole range, not selected.

**Statistics: "Differences supported?"**
Work-level paired bootstrap. All X5 intervals exclude zero by wide margins. The dev protocol is excluded from every claim.

**Baselines: "Relevant prior methods represented?"**
**No.** No Qmax, ByteCover or MOVE was run here, only cited from their papers.
**→** Limitation. Future work: run the same coverage analysis on the ranked lists of a strong public system (for example Discogs-VINet, open and reproducible).

**Computation: "Scalability demonstrated?"**
Three measured operating points on one laptop. There is no ANN index (exact search at 15k is 7 µs/query), so "scalable" must be qualified.
**→** Report measured per-pair and per-query costs only. A1 adds per-K timing.

**Limitations: "What would a reviewer criticise?"**
Weak absolute accuracy, single dataset, HPCP only, two-recording training works, whole-query DTW (no local alignment), and a partly transductive k-occurrence signal. All listed in `research_question.md` and the paper.

**Reproducibility: "Could another researcher reproduce this?"**
- Artifact analyses: yes, from the repository alone, in 30 s.
- Local experiments: yes, given Da-TACOS and the checkpoints. The checkpoints are not redistributable (licence), but retraining reproduces them bit for bit (README).

**Literature verification.**
Many related-work entries are verified only by search results (tag [S]).
**→** Before submission, read every [S] paper in full and replace the tag with [V]. The MIREX numbers remain uncited (tag [U]).

**Honest state.** Promising, but the positive-intervention half (E1/F1) is unrun. See the decision in `PROGRESS.md`.


## Round 2 (2026-09-25, after the local runs)

Status of the round-1 actions:
- **"How to choose K" takeaway**: done. The paper now states that K = 30 was a cost choice, that MAP has not saturated at K = 500, and where the bottleneck moves (B1).
- **Per-K timing**: done, as measured ms per pair × K. The per-pair cost was measured once on the local laptop; the per-K cost is derived, and the paper says so.
- **Leaderboard hygiene**: the post-hoc hybrid-vs-exhaustive comparison (C1) is labelled POST-HOC in its result directory, its table caption and the paper, with its three caveats.

New attacks, and the answers the evidence supports:

1. **"Your hypothesis failed: coverage is not dominant at large K."** Correct, and the paper says so. H1 is restated as holding for K ≲ 100. The decomposition that shows the shift is the contribution.
2. **"The adaptive-K result is a tuning failure, not a finding."** The policy family, menus, budgets, direction search and criterion were registered before any local result. The difficulty model is good (AUROC 0.76). The failure is a Hit@1/MAP trade-off, consistent across cells. **Residual risk:** other policy families (continuous K, cost-sensitive objectives) were not tried; say so.
3. **"The E1 budget criterion is pedantic."** Four cells miss it by < 0.4% of the budget. Every one of them lowers Hit@1, so relaxing the criterion does not change the conclusion. Both facts are in the paper.
4. **"Window fusion might be tuned on the benchmark."** Window count, length, offsets and the 50/50 weight were fixed in the registry before F1 ran, and no other fusion setting was evaluated. **Residual risk:** one setting is not a study of fusion; A2 plus a calibration-only sweep would be.
5. **"Stage-1 hubness: λ = 0 does not show hubness is absent."** Agreed; the paper calls it weak evidence. A benchmark sweep of λ at Stage 1 would be tuning on the test set, so it is not proposed.
6. **"Local runs came from a dirty working tree."** Disclosed (`env.json`: `dirty: True` at e6c9c9e). A1's exact reproduction of the frozen benchmark bounds the effect on the shared pipeline. F1 and H1 have no frozen counterpart except H1's profile-cosine/384 row, which equals A1's rerank-only K = 30 exactly (MAP 0.11159). **→** Future local runs should start from a clean, committed tree (added to `LOCAL_EXPERIMENTS.md`).
7. **"Different machines for the frozen runs and the local runs."** It affects timing only. Results are deterministic and A1 reproduced the frozen numbers bit for bit.
8. **"Weak absolute accuracy."** Unchanged. Still the main threat. The hybrid at K = 500 (0.212 MAP) is below Qmax on HPCP (0.333).

**Honest state.** The analysis paper is supported by executed, validated experiments. The positive-intervention paper depends on A2.

## Round 3 (2026-09-25, after A2)

1. **"A2 was registered after you saw F1 — isn't that forking paths?"** A2 was registered after the coverage gain was known but before any end-to-end number. Its fusion setting is the only one ever evaluated, and its criterion (K = 30 and K = 100) was fixed in advance. The paper states the ordering. The residual risk is that one setting is not a study of fusion.
2. **"Is the gain just coverage?"** Mostly. ΔMAP is largest at small K (+0.017) and shrinks as coverage saturates (+0.010 at K = 500), which is what a coverage effect predicts. The fused score also enters the α blend, so a small in-shortlist effect cannot be excluded.
3. **"Dirty tree again."** `env.json` for A2 records `dirty: True` at `e3ff315`. The validator now checks that A2's Stage-1 coverage equals F1's `win_fuse` at every K (it does, exactly), and A1 reproduced the frozen run exactly. Disclosed in the paper.
4. **"Cost?"** Window embeddings and fused scores took 198 s for the whole benchmark, against 78 min of alignment at K = 500. Fusion at K = 20 matches the global Stage 1 at K = 30 in MAP.
