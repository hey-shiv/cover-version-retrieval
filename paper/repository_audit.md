# Repository audit (paper phase)

Audited 2026-09-25 on branch `claude/compassionate-cray-q8bxmt`, restarted from `main` at `1bf4107` (merge of PR #3).
Every statement below was checked against a file in the repository; file paths are given so each can be re-checked.

## 1. What the repository contains

| area | contents | status |
|---|---|---|
| `src/cover_retrieval/` | HPCP loading and preprocessing, TCN encoder, SupCon loss, exact cosine index, profile-cosine key rotation, cosine cross-similarity, batched subsequence DTW, z-scored hybrid fusion, hubness correction, metrics | tested |
| `scripts/` | manifests, download, training, classical/hybrid/benchmark runners, hubness calibration | used for runs 1–4 |
| `configs/` | `base.yaml` (seed 20260817), `hybrid_dev.yaml`, `full_train.yaml`, `benchmark.yaml`, `classical_baseline.yaml`, `smoke.yaml` | |
| `reports/results/` | four frozen benchmark runs (JSON and per-query CSV), development results, calibration, hubness, training histories, error cases | **frozen; primary evidence** |
| `reports/*.md`, `notes/decisions.md` | narrative reports and the decision log D-001–D-022 | prose, checked against the JSON files below |
| `research/` | registry, cloud analyses X0–X7, local runs A1/A2/F1/H1, analyses B1/C1/D1/E1/F1s/H1s/A2s, figures, tables | **primary evidence for the K-sweep results** |
| `paper/` | draft paper and generated tables | rewritten in this phase |
| `web/` | static companion site built from the same result files | not a source of truth |
| `tests/` | 142 tests (`pytest --collect-only`), all passing | README says 119 (stale; corrected) |

History: 22 commits on `main`, no tags, no other branches with unmerged research.

## 2. Evidence tiers

| tier | meaning | experiments |
|---|---|---|
| FROZEN-RUN | the four locked benchmark evaluations, run by the author, committed with per-query files | runs 1–4 (`reports/results/benchmark*.json`) |
| DEV | 20-query development protocol, 1 relevant item, 119 candidates per query | `reports/results/*dev*.json` |
| ARTIFACT-ANALYSIS | cloud analyses computed only from committed per-query files | X0–X7 |
| LOCAL-FULL | runs made on the author's laptop over all 13,000 benchmark queries, validated before use | A1 (K sweep), A2 (fused Stage 1), F1 (Stage-1 variants), H1 (alignment ablation) |
| ANALYSIS of LOCAL-FULL | cloud analyses of the LOCAL-FULL files | A2s, B1, D1, E1, F1s, H1s |
| POST-HOC | analyses not registered in advance | C1 |

**The paper brief does not mention the LOCAL-FULL tier.** It exists, is validated (71 checks, 0 failures, including exact cross-run equalities), and answers the brief's RQ1 (shortlist size vs. accuracy and cost) directly. It is used in the paper.

## 3. Datasets and roles

| role | works | recordings | source | used for |
|---|---|---|---|---|
| train | 4,780 (runs 2–4) / first 1,500 of them (run 1) | 9,560 / 3,000 | Da-TACOS Cover Analysis subset | encoder training |
| validation | 150 | 300 | Cover Analysis | checkpoint selection |
| calibration | 10 | 20 | Cover Analysis | α, λ, resolution; nothing else |
| development query + distractor | 20 + 40 | 40 + 80 | Cover Analysis | 20 × 120 development protocol |
| benchmark | 1,000 cliques of 13 + 2,000 noise | 15,000 | Da-TACOS benchmark subset | test only |

Source: `data/manifests/manifest_info.json`. All roles are WID-disjoint. The benchmark is evaluated with all 13,000 clique tracks as queries against all 15,000 tracks, self excluded, so each query has 12 relevant items (`benchmark*.json → protocol`).

## 4. Experiment inventory

| id | what | evidence file(s) | tier |
|---|---|---|---|
| run 1 | 1,500-work encoder, 60 epochs, 96 frames, α 0.05; plus exhaustive 96-frame classical alignment | `benchmark.json`, `benchmark_per_query.csv` | FROZEN-RUN |
| run 2 | 4,780-work encoder, 60 epochs, 96 frames, α 0.15, hub correction λ 0.5 (hybrid and exhaustive classical) | `benchmark_full96.json` | FROZEN-RUN |
| run 3 | same encoder, 384-frame rerank, α 0.15, λ 0.6 | `benchmark_full384.json` | FROZEN-RUN |
| run 4 | 4,780 works, 150 epochs, 384-frame rerank, α 0.10, λ 0.6 | `benchmark_long384.json` | FROZEN-RUN |
| dev | classical, hybrid, resolution sweep, hubness | `classical_dev.json`, `hybrid_dev*.json`, `hubness_dev.json`, `hubness_correction*.json` | DEV |
| A1 | run-4 pipeline at K = 5…500 from one K = 500 alignment pass; reproduces run 4 exactly at K = 30 | `research/results/A1_k_sweep_long384/` | LOCAL-FULL |
| A2 | A1 with the fused global + window Stage 1 | `research/results/A2_k_sweep_win_fuse/` | LOCAL-FULL |
| F1 | Stage-1 variants and checkpoints (coverage only) | `research/results/F1_stage1_variants/` | LOCAL-FULL |
| H1 | key handling × resolution at K = 30 | `research/results/H1_alignment_ablation/` | LOCAL-FULL |
| B1/D1/E1/A2s/F1s/H1s | failure classes by K, difficulty prediction, adaptive K, paired analyses | `research/results/*/` | ANALYSIS |
| C1 | hybrid at K vs. exhaustive run-2 system | `research/results/C1_hybrid_vs_exhaustive_posthoc/` | POST-HOC |

**Never run:** exhaustive classical alignment at 384 frames. It is estimated at about 42 h, as 16 × the measured 2.63 h at 96 frames (`reports/improvement_experiments.md`).

## 5. Inconsistencies found

1. **Validation-curve wording.** `notes/decisions.md` (D-021) says validation MAP "was still drifting upward at epoch 150". The history file (`training_history_encoder_full_long.csv`) shows:
   - the best epoch was 138 (0.4306);
   - the least-squares slope over epochs 121–150 is −0.00013 per epoch;
   - the mean rose from 0.409 (epochs 91–120) to 0.421 (epochs 121–150);
   - the cosine learning-rate schedule reaches 0 at epoch 150.

   Correct statement: the curve rose and then flattened as the learning rate decayed. Neither convergence nor continued improvement is shown. The decision log is append-only and was not edited; the paper uses the correct statement.
2. **Test count.** README says 119 tests; the collected count is 142. README corrected.
3. **Run-4 Stage-1 row.** The README and `improvement_experiments.md` once printed Stage-1 MRR 0.297, Hit@1 0.234 and Hit@10 0.373 for run 4. The result file gives 0.232, 0.155 and 0.380. The paper uses the file (`baseline_audit.md` lists this and four other documentation inconsistencies).
4. **Machine-dirty provenance.** Every frozen run records a `-dirty` git commit (e.g. `9a7bfe2-dirty`), and A1/A2/F1/H1 record `dirty: True`. A1 reproduces run 4 bit for bit, and cross-run equality checks pass exactly, which bounds the practical effect. This is disclosed.
5. **Hub correction on development.**
   - At 96 frames: +0.041 [−0.045, +0.131].
   - At 384 frames: −0.030 [−0.158, +0.106].

   Both intervals include zero, yet the benchmark effect is large and positive. The reports mention this only in passing; the paper treats it as a third development/benchmark disagreement.

## 6. Claims that needed verification

All checked in `result_verification.md`. Two items from the paper brief were **wrong or imprecise**:
- "validation was still drifting upward at epoch 150" (see 5.1);
- "the repository currently reports 119 Python tests" — it collects 142.
