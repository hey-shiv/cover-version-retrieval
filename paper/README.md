# Paper (draft)

**Status: PAPER READY WITH MINOR ADDITIONS, as an empirical analysis paper with one registered positive intervention** (see [`research/PROGRESS.md`](../research/PROGRESS.md)).

- **Built on:**
  - the ARTIFACT analyses X0–X7 of the frozen Da-TACOS runs;
  - the local full-benchmark runs A1, F1 and H1 (LOCAL-FULL, 13,000 queries, run on the author's machine);
  - the cloud analyses of those runs: B1, D1, E1, and C1, which is post-hoc and labelled as such.
- **Also built on:** A2 (LOCAL-FULL), the registered end-to-end test of the fused Stage 1, and its analysis A2s. Nothing is pending.

## Sources of every number

| what | where it comes from |
|---|---|
| numbers in the prose | `generated/numbers.tex` (X*) and `generated/numbers_local.tex` (A1, A2, B1, C1, D1, E1, F1, H1), written by `research/scripts/make_tables.py` from `research/results/*/metrics.json` |
| tables | `generated/t*.tex`, same script |
| figures | `research/figures/*.pdf`, written by `research/scripts/make_figures.py` |
| references | `references.bib`. `[R]` entries were verified against Crossref / arXiv / ISMIR (repository literature notes); `[S]` entries only by web search. Read every `[S]` paper in full before submission. |

## Build

```bash
python research/scripts/validate_results.py && python research/scripts/analyze_local.py   # deterministic (fixed seed); rewrites its outputs
python research/scripts/artifact_analysis.py && python research/scripts/make_figures.py && python research/scripts/make_tables.py
cd paper && latexmk -pdf main.tex     # needs a LaTeX installation (not available in the cloud session)
```

The cloud session could not compile LaTeX. It checked mechanically that every macro, citation, `\input`, `\ref` and figure the draft uses exists, and that the generated tables contain no raw Unicode.
