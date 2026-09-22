# Week 1 Plan — Structure-Aware Hybrid Retrieval for Cover Song Identification

## Week 1 Mission
Build a reproducible, classical **HPCP-based cover-song retrieval baseline** on a fixed Da-TACOS development protocol: 12-key invariance → cross-similarity → subsequence DTW → ranked candidates → verified metrics. It is a controlled development baseline, not your final benchmark result and not a neural model.

The repository is currently empty and not initialized as Git, so Day 1 starts by creating the project properly.

## Week 1 end state
By Sunday, you should possess:

| Area | Measurable end state |
|---|---|
| Knowledge | You can explain WID/PID, cover cliques, HPCP, 12 chroma rotations, cross-similarity, DTW vs subsequence DTW, AP/MAP/MRR/Recall@K, and why the benchmark remains untouched. |
| Code | A command loads Da-TACOS HPCP features, builds a fixed manifest, scores query–candidate pairs, ranks candidates, and writes metrics/results. |
| Experiments | One fixed development experiment: 20 queries against 120 candidates, with self-matches excluded. |
| Metrics | `MAP`, `MRR`, `Recall@1`, `Recall@10`, `Recall@100`, and first-relevant rank, each validated first using hand-calculated toy rankings. |
| Plots/reports | HPCP plots, 12-rotation plot, two cross-similarity matrices with alignment paths, a retrieval metrics table, and an error gallery. |
| Research artifact | Dataset card, protocol note, decision log with seven decisions, and an error-analysis report. |
| GitHub | Clean repository, pushed `main` branch, no data/features committed, reproducible config and manifests committed. |
| Conceptual ability | You can defend every baseline decision: why HPCP, why rotations, why local alignment, why this development subset, and why each metric. |

A Week 1 result is valid only as: "a reproducible pilot baseline under this declared protocol." It is not evidence of state-of-the-art performance.

## Exact resources

### Must read deeply
1. [Da-TACOS: A Dataset for Cover Song Identification and Understanding — Yesiler et al., 2019](https://archives.ismir.net/ismir2019/paper/000038.pdf)
   - Read: §§1, 3, 5; Table 1 and Table 2.
   - Extract: dataset subsets, WID/PID meaning, features, distractors, metrics.
   - Unlocks: correct data protocol and evaluation.
2. [Chroma Binary Similarity and Local Alignment Applied to Cover Song Identification — Serrà et al., 2008](https://ieeexplore.ieee.org/document/4523006)
   - Read: abstract, §§1–4, figures, conclusion.
   - Extract: why tonal features, key invariance, cross-similarity, local alignment.
   - Unlocks: the baseline's reasoning.
3. [FMP Chapter 3: Music Synchronization — Meinard Müller](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3.html)
   - Read/run: music synchronization, chroma features, DTW, subsequence DTW.
   - Unlocks: implementation intuition.

### Skim
- [Essentia Cover Song Identification tutorial](https://essentia.upf.edu/tutorial_similarity_cover.html): read the four pipeline steps and inspect the output figures.
- [MIREX Cover Song Identification protocol](https://music-ir.org/mirex/wiki/2025%3ACover_Song_Identification): read data and evaluation only.
- [Stanford IR book, evaluation chapter](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html): AP, MAP, reciprocal rank.

### Reference only
- [Da-TACOS repository](https://github.com/MTG/da-tacos): H5 format, download script, licensing.
- [`acoss`](https://github.com/furkanyesiler/acoss): inspect its `Serra09` structure only. Do not copy it; it is AGPL-3.0.
- [FMP DTW notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html): return here while implementing.
- [FMP music synchronization notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3_MusicSynchronization.html): return here for chroma/transposition details.

## Week 1 project structure

```text
cover-version-retrieval/
├── README.md
├── DATASET_CARD.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── week1_baseline.yaml
├── data/
│   └── manifests/
│       ├── week1_calibration.csv
│       ├── week1_candidates.csv
│       └── week1_queries.csv
├── src/
│   └── cover_retrieval/
│       ├── data/datacos.py
│       ├── features/hpcp.py
│       ├── alignment/transposition.py
│       ├── alignment/dtw.py
│       ├── retrieval/rank.py
│       └── evaluation/metrics.py
├── scripts/
│   ├── build_week1_manifest.py
│   └── run_week1_baseline.py
├── tests/
│   ├── test_datacos.py
│   ├── test_transposition.py
│   ├── test_dtw.py
│   └── test_metrics.py
├── notebooks/
│   └── 01_week1_feature_inspection.ipynb
├── reports/
│   ├── dataset_profile.md
│   ├── week1_baseline.md
│   └── error_analysis.md
└── notes/
    ├── papers/
    └── decisions.md
```

Keep only this. No models, training code, FAISS, deployment, or feature fusion.

## Week 1 data protocol

Use only the **Da-TACOS Cover Analysis subset**.

It has 5,000 two-recording cliques. That is ideal for a small controlled baseline, while the 15,000-track Da-TACOS benchmark remains entirely untouched. [Dataset details](https://github.com/MTG/da-tacos)

### Fixed manifest strategy
1. Filter to WIDs that:
   - have exactly two PIDs;
   - have both HPCP files present and loadable;
   - contain valid finite arrays.
2. Sort WIDs lexicographically.
3. Use `numpy.random.default_rng(20260817)` to sample 70 valid WIDs without replacement.
4. Split them deterministically:
   - 10 WIDs → calibration set, 20 tracks.
   - 20 WIDs → evaluation query works, 40 tracks.
   - 40 WIDs → evaluation distractor works, 80 tracks.
5. Candidate set = all 120 tracks from the 60 evaluation WIDs.
6. Query set = one deterministic PID per query WID: lexicographically lowest PID from each of the 20 query WIDs.
7. Relevant item = the other PID sharing the query's WID.
8. Exclude the query PID itself from its candidate ranking.

### Fixed compute-control configuration
Use this only for Week 1:

```yaml
feature: hpcp
temporal_resample_frames: 96
normalization: l2_per_frame
key_offsets: [0, 1, ..., 11]
rotation_selection: highest_mean_feature_cosine
alignment: subsequence_dtw
local_cost: cosine_distance
score_normalization: divide_by_alignment_path_length
```

Uniformly resample each full sequence to 96 frames. This is a declared pilot compute control—not a claim that 96 frames are optimal.

Do not use titles, artists, release years, tags, metadata key, tempo, or instrumental labels for ranking. Metadata is for diagnostics only. Using it to choose a match would be leakage.

## Evaluation requirements
Implement all metrics this week, but interpret them correctly.

| Metric | Week 1 meaning |
|---|---|
| AP | With one relevant partner per query, AP equals reciprocal rank. Still implement it correctly for later multi-cover cliques. |
| MAP | Mean AP across 20 queries. |
| MRR | Mean reciprocal rank across 20 queries. It will equal MAP in this two-track-clique protocol; that is expected. |
| Recall@1 | Fraction of queries whose partner ranks first. |
| Recall@10 | Fraction whose partner is in the first ten. |
| Recall@100 | Meaningful here because each query ranks 119 non-self candidates. |
| First relevant rank | Exact rank of the counterpart recording. Report mean and median. |

Before touching Da-TACOS, write three toy rankings by hand and unit-test expected AP, MRR, Recall@K, and first rank.

## Research habit: decision log
Add this to `notes/decisions.md` every day:

```md
## D-00X — Short title
Date:
Decision:
Reason:
Evidence/resource:
Alternative rejected:
Effect on validity or compute:
Locked until:
```

Your Week 1 decisions should be:
- D-001: Cover Analysis subset, not benchmark.
- D-002: HPCP only.
- D-003: per-frame L2 normalization.
- D-004: 12 cyclic pitch shifts.
- D-005: uniform 96-frame resampling.
- D-006: subsequence DTW with cosine cost.
- D-007: WID-level relevance and ranking metrics.

## Day 1

### A. Study — 80 minutes
- Open [Da-TACOS paper](https://archives.ismir.net/ismir2019/paper/000038.pdf).
  - Read §§1 and 3, Table 1.
  - Focus: why CSI is retrieval, benchmark versus cover-analysis subsets, WID/PID, features.
  - Skip: §4 musicological analysis and detailed baseline results for now.
  - Time: 55 minutes.
- Open [Da-TACOS README](https://github.com/MTG/da-tacos).
  - Read: "Structure," "Metadata," "Pre-extracted features," "License."
  - Skip: download implementation details until coding.
  - Time: 25 minutes.

### B. Thinking — 25 minutes
Write answers:
1. Why is "same work" a better label than "same genre" for this task?
2. Why is the benchmark set not the correct place to tune a baseline?
3. What does one WID represent, and what does one PID represent?
4. Why can two recordings with different keys still be covers?

### C. Implementation — 2.5–3 hours
Create the repository and initialize Git.

Build:
- `README.md`: one paragraph project statement and Week 1 scope.
- `DATASET_CARD.md`: source, license, WID/PID, intended use, prohibited Git commits.
- `pyproject.toml`: minimal dependencies only: NumPy, SciPy, h5py/deepdish, PyYAML, Matplotlib, pytest.
- `configs/week1_baseline.yaml`.
- `src/cover_retrieval/data/datacos.py`:
  - `load_metadata(path) -> dict`
  - `iter_two_track_cliques(metadata) -> iterator`
- `scripts/build_week1_manifest.py`:
  - deterministic WID sampling with seed `20260817`;
  - writes calibration, candidates, and queries CSVs.

Test:
- metadata loader rejects malformed JSON;
- every selected WID has exactly two PIDs;
- no WID appears in more than one manifest role.

Do not implement H5 feature loading or any similarity method yet.

### D. Experiment — 45 minutes
Run the manifest builder on metadata only.

Output:
- total WIDs;
- valid two-PID WIDs;
- selected 10/20/40 WID counts;
- query count = 20;
- candidate count = 120.

### E. Documentation — 20 minutes
- Add D-001 to `notes/decisions.md`.
- In `DATASET_CARD.md`, document the exact split and that Da-TACOS benchmark is untouched.
- In `README.md`, state: "Week 1 uses only a fixed Cover Analysis development subset."

### F. GitHub checkpoint
Commit:

```text
chore: initialize reproducible week1 project scaffold
```

Push `main` after creating your remote repository.

### G. Definition of Done
This must work:

```bash
python scripts/build_week1_manifest.py --config configs/week1_baseline.yaml
```

It must create three committed CSV manifests, with 20 queries and 120 candidates, and no WID overlap between calibration and evaluation roles.

---

## Day 2

### A. Study — 75 minutes
- Reopen Da-TACOS README:
  - "Pre-extracted features," "Loading the data in Python."
  - Focus: H5 structure, `hpcp`, `label`, `track_id`.
  - Time: 35 minutes.
- Read Da-TACOS paper §3 and Table 1 again.
  - Focus: which features exist and why HPCP is your Week 1 choice.
  - Time: 20 minutes.
- Inspect [`acoss` dataset format](https://github.com/furkanyesiler/acoss).
  - Read only "Dataset structure required for acoss."
  - Skip all benchmark implementation code.
  - Time: 20 minutes.

### B. Thinking — 25 minutes
1. What is the expected shape and meaning of an HPCP sequence?
2. Why should your loader validate H5 metadata against the manifest?
3. Why is it useful to start with precomputed features rather than audio?
4. What would constitute a corrupt feature record?

### C. Implementation — 2.5–3 hours
Build `src/cover_retrieval/data/datacos.py`:
- `feature_path(root, wid, pid, feature="hpcp") -> Path`
- `load_hpcp(path) -> np.ndarray`
- `validate_hpcp(array) -> None`
  - 2D;
  - one dimension equals 12;
  - finite;
  - nonempty;
  - nonzero energy.
- `load_manifest(csv_path) -> list[Record]`

Build `scripts/profile_week1_data.py`:
- load every Week 1 record;
- print shape statistics;
- record unreadable/missing files;
- verify the H5 `label` matches WID and `track_id` matches PID.

Do not reshape, normalize, or compare features yet.

### D. Experiment — 45–60 minutes
Run the profiler on all 140 selected tracks, including calibration.

Produce:
- shape histogram;
- min/median/max frame length;
- number of invalid/missing feature files;
- a CSV of feature shapes.

If any chosen WID fails validation, rebuild manifests after filtering invalid WIDs. Do not manually substitute tracks.

### E. Documentation — 20 minutes
- Write `reports/dataset_profile.md`.
- Add D-002: "HPCP only for Week 1 because it directly represents harmonic pitch classes and avoids uncontrolled feature fusion."
- Add data-license reminder to `.gitignore` comments and README.

### F. GitHub checkpoint

```text
feat: add datacos hpcp loader and dataset validation
```

### G. Definition of Done
This must complete without silent skips:

```bash
python scripts/profile_week1_data.py --config configs/week1_baseline.yaml
```

It must either validate all 140 selected records or fail loudly and regenerate a new deterministic manifest from the filtered valid pool.

---

## Day 3

### A. Study — 80 minutes
- Open [FMP music synchronization](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3_MusicSynchronization.html).
  - Read: Introduction and "Chroma-Based Feature Representation."
  - Focus: robustness to timbre/instrumentation, feature normalization.
  - Skip: score/audio synchronization extensions.
  - Time: 45 minutes.
- Open [Essentia CSI tutorial](https://essentia.upf.edu/tutorial_similarity_cover.html).
  - Read: steps 1–2 only.
  - Focus: HPCP and key invariance.
  - Skip: code copying.
  - Time: 20 minutes.
- Read Serrà et al. abstract and introduction.
  - Focus: why covers may change key, tempo, and structure.
  - Time: 15 minutes.

### B. Thinking — 25 minutes
1. What information does HPCP preserve that MFCC does not?
2. What information does HPCP intentionally discard?
3. Why is per-frame L2 normalization reasonable?
4. In what case might HPCP create a false positive?

### C. Implementation — 2.5–3 hours
Build `src/cover_retrieval/features/hpcp.py`:
- `as_time_by_pitch(X) -> np.ndarray[T, 12]`
- `l2_normalize_frames(X, eps=...) -> np.ndarray[T, 12]`
- `uniform_resample_sequence(X, n_frames=96) -> np.ndarray[96, 12]`
- `plot_hpcp(X, title, output_path)`

Tests:
- each nonzero normalized frame has norm 1;
- resampling returns exactly 96 frames;
- no NaNs arise from silent/zero frames.

Do not derive or re-extract HPCP from raw audio.

### D. Experiment — 45–60 minutes
Select:
- two true cover pairs from calibration WIDs;
- two noncover pairs from different calibration WIDs.

For all four pairs:
- plot both normalized/resampled HPCP sequences;
- compare mean-pool cosine similarity;
- write one observation per pair.

### E. Documentation — 20 minutes
- Update `reports/dataset_profile.md` with feature orientation and resampling policy.
- Add D-003 and D-005.
- Add paper note: `notes/papers/serra2008.md`.

### F. GitHub checkpoint

```text
feat: normalize resample and visualize hpcp sequences
```

### G. Definition of Done
You can point to an HPCP plot and explain, without notes, why a cover may look related despite different instrumentation—and why that does not prove it is a cover.

---

## Day 4

### A. Study — 75 minutes
- Read Serrà et al. §§2–3.
  - Focus: pitch-class circular shifts, transposition invariance, cross-similarity.
  - Skip: historical related work details.
  - Time: 45 minutes.
- Revisit FMP Ch. 3 feature-normalization/transposition sections.
  - Time: 30 minutes.

### B. Thinking — 20 minutes
1. Why are exactly 12 cyclic shifts tested?
2. Why must the shift be applied only along the pitch-class dimension?
3. Why should metadata key estimates not choose the shift?
4. What is the difference between key invariance and claiming two tracks are covers?

### C. Implementation — 2.5–3 hours
Build `src/cover_retrieval/alignment/transposition.py`:
- `rotate_pitch_classes(X, semitones) -> X_rotated`
- `mean_feature_cosine(X, Y) -> float`
- `best_key_rotation(X, Y) -> RotationResult`
  - tests shifts 0–11;
  - returns best shift, 12 scores, rotated candidate sequence.

Tests:
- shift 0 changes nothing;
- applying shifts `k` then `12-k` restores the original;
- synthetic shifted sequence recovers its known rotation.

Do not add key estimation, tempo normalization, or feature fusion.

### D. Experiment — 45–60 minutes
On the ten calibration WIDs:
- for every true pair, calculate all 12 rotation scores;
- for ten matched noncover comparisons, do the same;
- produce one 12-bar score plot for two true pairs and two noncover pairs.

### E. Documentation — 20 minutes
- Add D-004.
- In `notes/decisions.md`, explicitly state: "Rotation is selected from HPCP only, not metadata key."
- Add a short "What key invariance does not solve" section to your Serrà paper note.

### F. GitHub checkpoint

```text
feat: add transposition invariant hpcp matching
```

### G. Definition of Done
A synthetic HPCP sequence rolled by `k` pitch bins must recover `k` or its equivalent inverse convention exactly. Your tests must make the convention explicit.

---

## Day 5

### A. Study — 90 minutes
- Read the [FMP DTW notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html).
  - Read: "Basic Idea," "Warping Path," recurrence, backtracking.
  - Focus: boundary, monotonicity, step-size constraints.
  - Skip: variants until your basic code works.
  - Time: 60 minutes.
- Read FMP's explanation of [subsequence DTW](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3.html).
  - Focus: why it relaxes full start/end matching.
  - Time: 30 minutes.

### B. Thinking — 25 minutes
1. Why does mean-pool cosine ignore musical order?
2. What does DTW gain over framewise comparison?
3. Why can full start-to-end DTW still fail for cover versions?
4. What constraints prevent an alignment path from becoming nonsensical?

### C. Implementation — 2.5–3 hours
Build `src/cover_retrieval/alignment/dtw.py`:
- `cosine_cost_matrix(X, Y) -> np.ndarray[T_x, T_y]`
- `dtw(cost_matrix) -> DTWResult`
  - total cost;
  - accumulated cost matrix;
  - backtracked path.
- `subsequence_dtw(cost_matrix) -> DTWResult`
  - path may start/end within candidate sequence;
  - score normalized by path length.
- `plot_alignment(cost_matrix, path, output_path)`.

Tests:
- hand-calculated tiny cost matrix;
- a time-stretched synthetic sequence;
- path validity: monotonicity and legal steps.

Do not optimize with numba, use Soft-DTW, or use a third-party DTW package yet.

### D. Experiment — 45–60 minutes
Use:
- one true pair after best key rotation;
- one noncover pair after best key rotation.

For each:
- save cross-similarity/cost matrix;
- overlay DTW and subsequence-DTW paths;
- compare normalized scores.

### E. Documentation — 20 minutes
- Add D-006.
- Write `notes/papers/fmp_dtw.md`.
- Add a one-paragraph explanation to `reports/week1_baseline.md`: "Why subsequence DTW is the Week 1 local-alignment approximation."

### F. GitHub checkpoint

```text
feat: implement dtw and subsequence alignment baseline
```

### G. Definition of Done
For a toy cost matrix, your code returns the manually expected path and normalized cost. The test must fail if you break boundary, monotonicity, or path-length normalization.

---

## Day 6

### A. Study — 75 minutes
- Read the remainder of the [Essentia CSI tutorial](https://essentia.upf.edu/tutorial_similarity_cover.html).
  - Focus: cross-similarity and local subsequence alignment.
  - Skip: treating its implementation as code to copy.
  - Time: 45 minutes.
- Read Serrà et al. §4 and conclusion.
  - Focus: limitations and false-match risks.
  - Time: 30 minutes.

### B. Thinking — 25 minutes
1. Why should a retrieval system return a ranking rather than only a binary cover/noncover label?
2. Why use a normalized path cost?
3. What does the 96-frame resampling simplify, and what information might it lose?
4. Why is Week 1's system a baseline rather than a final method?

### C. Implementation — 2.5–3 hours
Build:

`src/cover_retrieval/retrieval/rank.py`
- `score_pair(query_hpcp, candidate_hpcp, config) -> PairScore`
  - normalize;
  - resample;
  - select best of 12 rotations by mean feature cosine;
  - compute subsequence-DTW score using selected rotation;
  - retain shift, path length, and score.
- `rank_candidates(query, candidates, config) -> list[RankedCandidate]`
  - excludes self PID;
  - sorts ascending distance;
  - adds `is_relevant` from WID equality.

`src/cover_retrieval/evaluation/metrics.py`
- Only function signatures today; full metric implementation tomorrow.

Do not add parallelism, indexes, learned embeddings, or benchmark-scale evaluation.

### D. Experiment — 45–60 minutes
Run 5 fixed calibration queries against the 20 calibration candidates.

For each query, save:
- ranked top 10 candidate CSV;
- selected pitch shift;
- relevant partner rank;
- best nonrelevant candidate;
- runtime per pair.

### E. Documentation — 20 minutes
- Add a "Baseline algorithm" section to README containing the six exact steps.
- Add D-007 draft: relevance is determined only by WID equality.
- Create `reports/error_analysis.md` using this template:

```md
## Query
PID / WID:
Relevant PID:
Relevant rank and score:
Top false positive PID / WID / score:
Selected shift:
Alignment plot:
Hypothesis:
- Why did the true match succeed or fail?
- What musical or representation factor may explain it?
- What should be tested later?
```

### F. GitHub checkpoint

```text
feat: rank candidates with transposition aware subsequence dtw
```

### G. Definition of Done
For each of five queries, one command writes a complete top-10 ranking with no self-match and exactly one WID-relevant item.

---

## Day 7

### A. Study — 75 minutes
- Read [Stanford IR evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html).
  - Focus: AP, MAP, reciprocal rank.
  - Skip: web-search-engine material unrelated to rankings.
  - Time: 45 minutes.
- Read [MIREX CSI evaluation](https://music-ir.org/mirex/wiki/2025%3ACover_Song_Identification).
  - Focus: each track as query, ranking, mean AP, first relevant rank.
  - Time: 30 minutes.

### B. Thinking — 25 minutes
1. Why are MAP and MRR identical in this Week 1 two-track setup?
2. Why will they not be identical on the final Da-TACOS benchmark?
3. Why is Recall@100 meaningful here but not if you had only 40 candidates?
4. What would make a comparison between two baselines unfair?
5. Why is it wrong to claim final performance from this development result?

### C. Implementation — 2.5–3 hours
Complete `src/cover_retrieval/evaluation/metrics.py`:
- `average_precision(relevance: list[bool])`
- `reciprocal_rank(relevance)`
- `recall_at_k(relevance, k)`
- `first_relevant_rank(relevance)`
- `evaluate_rankings(rankings) -> MetricsSummary`

Write `tests/test_metrics.py` with three hand-derived examples:
1. relevant at rank 1;
2. relevant at rank 5;
3. multiple relevant items, for future MAP correctness.

Build `scripts/run_week1_baseline.py`:
- reads frozen candidate/query manifests;
- runs baseline;
- writes:
  - `reports/week1/rankings.csv`
  - `reports/week1/metrics.json`
  - `reports/week1/metrics.md`
  - `reports/week1/runtime.csv`

### D. Experiment — 60 minutes
Run the frozen Week 1 evaluation:
- queries: 20 fixed query PIDs;
- candidates: 120 fixed evaluation tracks;
- one relevant partner per query;
- self excluded;
- no calibration WIDs included.

Report:
- MAP;
- MRR;
- Recall@1, @10, @100;
- mean/median first relevant rank;
- mean pair-scoring time;
- total runtime.

Error analysis:
- choose up to three successful cases: relevant rank ≤10;
- choose three false positives: nonrelevant rank 1;
- choose three false negatives: relevant rank >10.

If the system has fewer than three successes, document that honestly. Do not cherry-pick.

### E. Documentation — 30 minutes
Complete:
- `reports/week1_baseline.md`
  - protocol;
  - config;
  - metrics;
  - runtime;
  - limitations;
  - no final-benchmark claim.
- `reports/error_analysis.md`
- `notes/decisions.md` with final D-007.
- README quickstart command and Week 1 result link.

### F. GitHub checkpoint

```text
feat: evaluate reproducible week1 classical retrieval baseline
```

Tag it:

```text
week1-classical-baseline-v0
```

Push `main` and the tag.

### G. Definition of Done
This single command must regenerate all Week 1 outputs:

```bash
python scripts/run_week1_baseline.py --config configs/week1_baseline.yaml
```

It must produce metrics, rankings, and runtime files from committed manifests. All metric unit tests must pass.

## Week 1 deliverables
- Deterministic manifests and locked development protocol.
- Da-TACOS feature loader with validation.
- HPCP feature plots.
- Transposition-invariance tests and plots.
- DTW and subsequence-DTW implementation with toy tests.
- Classical ranked retrieval command.
- Verified metric functions.
- Week 1 metrics report.
- Error-analysis gallery.
- Seven documented design decisions.

## Week 1 GitHub state
Your repository should contain code, manifests, tests, configurations, reports, and notes.

It should not contain:
- Da-TACOS H5 features;
- audio;
- raw metadata dumps;
- downloaded archives;
- model checkpoints;
- copied `acoss` code.

Your README should clearly say:

> "This repository provides a reproducible Week 1 classical HPCP/subsequence-DTW development baseline. It does not report final Da-TACOS benchmark performance."

## Week 1 knowledge check
Answer these without notes:
1. What is the difference between a WID and a PID?
2. Why is CSI a ranking problem rather than ordinary multiclass classification?
3. What does HPCP preserve, and what does it discard?
4. Why are 12 cyclic pitch rotations tested?
5. Why is metadata key not allowed to select the winning rotation?
6. What does a cross-similarity matrix contain?
7. How does subsequence DTW differ from full DTW?
8. Why normalize alignment cost by path length?
9. Why are MAP and MRR equal in this Week 1 protocol?
10. Why may you not use the Da-TACOS benchmark set for Week 1 tuning?
11. What is a false positive in cover retrieval?
12. Why is the 96-frame representation acceptable only as a declared pilot constraint?

## Week 1 graduation criteria
Do not move to learned metric embeddings until all are true:
- You can run the baseline from scratch with one command.
- All tests pass, including toy metric and DTW tests.
- Manifests are deterministic and WID-disjoint by role.
- The benchmark subset has not been read, used, or tuned on.
- You have a saved metrics report and runtime report.
- You have inspected at least three success/failure cases.
- You can explain every baseline component without notes.
- You can state two concrete limitations of the Week 1 system.

## What I should do tomorrow at 9 AM
Open the Da-TACOS paper to §3, spend 55 minutes extracting the dataset protocol into `DATASET_CARD.md`, then initialize this repository with Git and create the deterministic Week 1 manifest builder.
