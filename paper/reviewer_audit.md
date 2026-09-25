# Reviewer audit (paper phase)

A simulated skeptical MIR review of `main.tex` version 1. For each criticism: the verdict, and what was changed in the paper. Earlier rounds, which cover the experiments rather than the manuscript, are in `research/reviewer_audit.md`.

| # | criticism | verdict | action taken |
|---|---|---|---|
| 1 | **Novelty.** "Two-stage embedding → alignment retrieval is ByteCover3/LIVI; hub correction is CSLS/Seo 2022. What is new?" | valid for method; answered for evidence | The abstract, introduction and §2 disclaim the architecture, correction and fusion as prior art. The contributions list is limited to measurements and one registered empirical result (`novelty_audit.md`). |
| 2 | **Related work.** "Is it current?" | was incomplete | Added DiVers (ISMIR 2026), CLEWS, unified track/version ID (2026), YouTube robustness (2025), LIVI (ECIR 2026). MIREX is described without numbers, because they could not be verified. |
| 3 | **Reproducibility.** "Could I reproduce this?" | yes | Every setting is in Table 14 (generated from config and result files). The method section writes out every equation. `REPRODUCIBILITY.md` gives commands. The checkpoints are licence-restricted but rebuild bit for bit. |
| 4 | **Leakage.** "Was anything tuned on the test set?" | no, now stated explicitly | Added the 0 shared WIDs/PIDs statement. The protocol says no setting was chosen on the benchmark and no run was repeated. Adaptive-K policies were cross-fitted by work. |
| 5 | **Statistics.** "Multiple comparisons?" | partly valid | The paper now says intervals are per comparison and unadjusted. The headline benchmark effects exclude zero by wide margins; development effects are described as inconclusive, never as significant. |
| 6 | **Baselines.** "No strong learned baseline was run." | valid, unfixable here | The absolute-accuracy gap (Qmax 0.333) is stated in the abstract and limitations. The first future-work item is a strong open encoder that accepts Da-TACOS features (MOVE or Re-MOVE on CREMA-PCP). Audio models such as Discogs-VINet cannot be run, because Da-TACOS has no audio. |
| 7 | **Computation.** "Is the scalability claim quantified?" | yes | Table 15 gives measured costs; 0.2% of pairs are aligned at K = 30. The unrun 384-frame exhaustive cost is labelled as an estimate everywhere, and the consistency check enforces this. |
| 8 | **Overclaiming: resolution mismatch.** "The hybrid is compared with a 96-frame exhaustive system." | valid | The abstract now says "exhaustive 96-frame alignment". §6.1 and §10 state the mismatch, and the frontier figure labels it. |
| 9 | **Overclaiming: "coverage accounts for 80% of the gain".** | wording fixed | Now "of the log-gain", with the factorisation spelled out in §6.2. |
| 10 | **Overclaiming: hub statistic scope.** | fixed | "rank-1 false positives in covered queries" |
| 11 | **Overclaiming: "not measured before".** | fixed | Now "we found no CSI study that reports these quantities". |
| 12 | **Post-hoc analysis.** "The hybrid beats exhaustive from K = 100: cherry-picked?" | valid concern | It is labelled post-hoc in the text and the appendix table, with three caveats, and is not in the contributions. |
| 13 | **A2 registration timing.** | valid | Stated in the limitations: A2 was registered after F1 showed the coverage gain, and only one fusion setting was tried. |
| 14 | **Training convergence.** "Is the 150-epoch model converged?" | handled | The curve's rise, the flat last 30 epochs, and the learning-rate decay are all stated. Neither convergence nor continued improvement is claimed (correcting a statement in `notes/decisions.md`). |
| 15 | **Provenance.** "Dirty working trees." | valid | Disclosed for both the frozen and the local runs. The exact reproduction and cross-run equality checks are cited as bounds. |
| 16 | **Development-set framing.** "Is 'the dev set lied' fair?" | reframed | The dev protocol was *inconclusive*, and its point estimates were misread. §11 explains the three structural reasons. |
| 17 | **Case studies as evidence.** | valid | They are labelled "chosen examples, not estimates of frequency". The frequencies come from the full-benchmark decomposition. |
| 18 | **Length.** | valid for ISMIR | The full draft is 21 pages. `paper_positioning.md` lists the 6-page cut. |

Remaining weaknesses that cannot be fixed without new work:
- one system;
- one dataset;
- low absolute accuracy;
- no strong external baseline;
- exhaustive 384-frame alignment never run;
- literature verified mostly from search results (`citation_audit.md`).
