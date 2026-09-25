# Paper: *Where Two-Stage Cover Song Retrieval Loses Its Covers*

**Status: PAPER READY WITH MINOR REVISIONS** as an empirical retrieval-analysis paper (see "Readiness" below). It is not a method paper.

- `main.pdf`: compiled draft, 21 pages including appendix.
- `main.tex`: every number is a macro from `tables/numbers*.tex`, every table is from `tables/`, and every figure is from `figures/`. All three are generated from result files.

## Files
| file | purpose |
|---|---|
| `repository_audit.md` | what exists, the evidence tiers, and inconsistencies found |
| `baseline_specification.md` | the system, verified line by line from code and configs |
| `result_verification.md` | every headline number traced to its file; corrections to the paper brief |
| `research_story.md` | the evidence-backed narrative |
| `related_work.md`, `literature_matrix.csv` | literature, with verification levels |
| `novelty_audit.md`, `paper_positioning.md` | what is and is not claimed; paper type; RQs |
| `reviewer_audit.md` | skeptical review of version 1 and the fixes made |
| `citation_audit.md` | per-reference verification; unverified numbers kept out of the paper |
| `consistency_report.md` | generated paper-vs-repository check (all ok) |
| `REPRODUCIBILITY.md` | commands, environment, data and checkpoint constraints |
| `tables/`, `figures/` | generated assets |

## Build
```bash
python research/scripts/make_paper_assets.py && python research/scripts/make_tables.py
python research/scripts/check_paper_consistency.py
cd paper && latexmk -pdf main.tex
```

## Readiness
**PAPER READY WITH MINOR REVISIONS.** The science is complete and locked, the draft compiles, and every number is generated and consistency-checked. Before submission:
1. **Read every reference tagged S** (`citation_audit.md`), especially Qmax 0.333 in Da-TACOS Table 2 and the ByteCover3 and LIVI pipeline descriptions. The cloud proxy blocked every primary source.
2. **Cut to venue length.** The draft is 21 pages; an ISMIR submission is 6 plus references. `paper_positioning.md` lists what stays.
3. **Proofread the generated tables** in the venue template.

**The smallest additions that would materially strengthen the paper** (a research cycle, not a revision):
- run the same decomposition with a strong open encoder that accepts Da-TACOS features: MOVE or Re-MOVE, on the CREMA-PCP features Da-TACOS distributes. Audio models such as Discogs-VINet cannot be run, because Da-TACOS has no audio;
- choose the fusion settings on calibration works.
