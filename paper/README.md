# Paper (draft)

**Status: draft, not submittable.** The research state is *PROMISING BUT NEEDS ONE MORE RESEARCH CYCLE* (see [`research/PROGRESS.md`](../research/PROGRESS.md)).

- **Built on:** only the executed ARTIFACT analyses (X0–X7) of the committed Da-TACOS benchmark results.
- **Pending:** sections depending on the local experiments A1 (K sweep), D1/E1 (uncertainty, adaptive K) and F1 (Stage-1 variants) are marked `\pending{...}`. They must be filled only from the generated result files, never by hand.

## Sources of every number

| what | where it comes from |
|---|---|
| numbers in the prose | `generated/numbers.tex`, written by `research/scripts/make_tables.py` from `research/results/X*/metrics.json` |
| tables | `generated/t*.tex`, same script |
| figures | `research/figures/*.pdf`, written by `research/scripts/make_figures.py` |
| references | `references.bib`. `[R]` entries were verified against Crossref / arXiv / ISMIR (repository literature notes); `[S]` entries only by web search. Read every `[S]` paper in full before submission. |

## Build

```bash
python research/scripts/artifact_analysis.py && python research/scripts/make_figures.py && python research/scripts/make_tables.py
cd paper && latexmk -pdf main.tex     # needs a LaTeX installation (not available in the cloud session)
```

The cloud session could not compile LaTeX. It verified mechanically that every macro, citation, `\input` and figure the draft references exists.
