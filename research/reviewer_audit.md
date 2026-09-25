# Reviewer audit (round 1, before the local experiments)

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
