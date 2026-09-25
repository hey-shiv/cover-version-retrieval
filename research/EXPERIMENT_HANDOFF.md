# Experiment handoff: what the local machine sends back

This research runs in two places.

| where | does | never does |
|---|---|---|
| **Cloud session** | literature, design, runners, analysis, statistics, figures, paper, website | touch Da-TACOS data or checkpoints (it has neither) |
| **Local machine** | runs the three registered data-dependent experiments | make research decisions, tune on the benchmark, edit results |
| **GitHub** (branch `claude/compassionate-cray-q8bxmt`) | carries the small result files between the two | carry features, caches or checkpoints |

## What to produce locally

Run [`LOCAL_EXPERIMENTS.md`](LOCAL_EXPERIMENTS.md) steps 0–6. That creates exactly these directories:

| directory | files | approx. size | required? |
|---|---|---|---|
| `research/results/A1_k_sweep_long384/` | metrics.json, per_query.csv (13,000 rows), signals.csv (13,000 rows), candidates.csv (15,000 rows), runtime.json, config.yaml, env.json, README.md | ~10 MB | **yes** (tier 1) |
| `research/results/F1_stage1_variants/` | metrics.json, per_query.csv (13,000 rows), runtime.json, config.yaml, env.json, README.md | ~6 MB | **yes** (tier 1) |
| `research/results/H1_alignment_ablation/` | the same six files | ~2 MB | tier 2 |
| `research/results/A2_k_sweep_win_fuse/` | as A1 | ~10 MB | registered follow-up; see `LOCAL_EXPERIMENTS.md`, "Next cycle: A2" |

Every file is written by a runner. Do not create, rename or edit any of them by hand.

What each file is for, cloud-side:
- `per_query.csv` (A1): for every query, the ranks of all its covers under Stage 1, and for every K the first-cover rank, AP and rank-1 item under three rerankers. Gives the K sweep, the failure classes at every K, and the adaptive-K evaluation.
- `signals.csv` (A1): label-free Stage-1 difficulty signals. Inputs to the uncertainty model (D1) and the adaptive policy (E1).
- `candidates.csv` (A1): each candidate's k-occurrence and hubness reference. Defines the hub failure class.
- `metrics.json`: aggregates plus the **reproduction check** against the frozen run-4 file.
- `env.json` / `config.yaml` / `runtime.json`: commit, machine, resolved settings, measured time. These give the compute/quality frontier its real costs.

## What must NOT be pushed

`data/external/`, `data/cache/`, `runs/` (checkpoints, block caches, probe caches), and any `*.h5`, `*.npz`, `*.npy`, `*.pt`, `*.zip`.
Da-TACOS features are CC BY-NC-SA and derived weights inherit that licence (D-015).
`.gitignore` already blocks all of these.
Check with `git status` before committing: only `research/results/...` and `research/PROGRESS.md` should appear.

## Signalling what happened

Add one line per experiment to [`PROGRESS.md`](PROGRESS.md) under **Local runs**. Include:
- the experiment ID;
- whether it was completed, partial or failed;
- wall-clock time;
- anything that deviated from `LOCAL_EXPERIMENTS.md`, for example a dropped checkpoint or `--rotations` without exhaustive.

A failed or partial run is still worth pushing. The cloud reports it as such.

## What the cloud does next, automatically

When the results are on the branch, the cloud session runs:

```bash
python research/scripts/validate_results.py      # refuses missing / corrupt / inconsistent outputs
python research/scripts/analyze_local.py         # B1, D1, E1, C1 (post-hoc), F1s, H1s; A2s when A2 exists
python research/scripts/artifact_analysis.py     # X0-X7 (unchanged inputs; regenerated for consistency)
python research/scripts/make_figures.py          # every figure from result files; missing ones listed
python research/scripts/make_tables.py           # paper tables from result files
```

Then it updates the novelty and reviewer audits and the paper. It updates the website last.
If a hypothesis is not supported, adaptive K for example, the paper's thesis is revised around what is supported. Negative results are reported, not dropped.

## Received so far

| experiment | pushed in | validated | notes |
|---|---|---|---|
| A1 | `6cf9999` | yes; reproduction of `benchmark_long384.json` exact (\|Δ\| = 0) | run from a dirty tree at `e6c9c9e` (disclosed) |
| F1 | `6cf9999` | yes; F1's reference Stage 1 equals A1's exactly | all three checkpoints |
| H1 | `6cf9999` | yes; the locked variant equals A1's rerank-only K = 30 exactly | full rotation set including exhaustive |
| A2 | `ca67dea` | yes; Stage-1 coverage equals F1 `win_fuse` at every K; registered criterion met | run from a dirty tree at `e3ff315` (disclosed) |
