# Local experiments: execution order

Run these on the machine that holds the Da-TACOS features and the encoder checkpoints.
Everything else (analysis, statistics, figures, paper, website) happens in the cloud from
the small files these steps produce. No step needs a research decision: every setting is
fixed in [`experiments/registry.yaml`](experiments/registry.yaml).

**Rules:**
- Never edit a result file.
- Never re-run into an existing output directory. The runners refuse to overwrite; use a new `--out` name and record why in `research/PROGRESS.md`.
- Never tune anything on the benchmark.

Total for tier 1 is about 3 h, most of it one alignment pass. Tier 2 adds 10 min to 2 h.

## 0. Environment (2 min)

```bash
git fetch origin && git checkout claude/compassionate-cray-q8bxmt && git pull
source .venv/bin/activate            # the environment that trained the encoders
pip install -e ".[dev]"
pytest                               # expect 142 passed (includes tests/test_research_lib.py)
```

## 1. Preflight (seconds)

```bash
python research/scripts/local/preflight.py
```

This must end in `READY for tier 1`. It needs:
- the benchmark HPCP files or their preprocessed caches under `data/cache/`;
- `runs/encoder_full_long/encoder_best.pt`;
- three committed locked-settings files.

If `runs/probe_reference/probe200_n384_benchmark.npy` exists it is reused, which saves about 37 min.

## 2. Sanity checks on 200 queries (~5 min)

These write to `/tmp`, never to `research/results/`, and are labelled `-SANITY`.

```bash
python research/scripts/local/run_shortlist_sweep.py --config configs/full_train.yaml --tag long384 \
  --set features.n_frames=384 --hub-correction reports/results/hubness_correction_n384.json \
  --max-queries 200 --out /tmp/sanity_A1

python research/scripts/local/run_stage1_variants.py --config configs/full_train.yaml \
  --checkpoint full_long=runs/encoder_full_long/encoder_best.pt --primary full_long \
  --max-queries 200 --out /tmp/sanity_F1
```

Proceed only if both finish and `/tmp/sanity_A1/per_query.csv` has 200 rows.

If the probe cache is missing, the A1 sanity run computes the full probe reference once, taking about 37 min. It is then cached and reused by step 3.

The full-run commands below repeat the sanity-check flags and settings unchanged; only `--out` changes, plus the extra checkpoints in step 4.

## 3. A1: shortlist-size sweep, the main experiment (~2.3 h, resumable)

```bash
python research/scripts/local/run_shortlist_sweep.py --config configs/full_train.yaml --tag long384 \
  --set features.n_frames=384 \
  --hub-correction reports/results/hubness_correction_n384.json \
  --frozen reports/results/benchmark_long384.json \
  --ks 5 10 20 30 50 100 200 500 \
  --out research/results/A1_k_sweep_long384
```

- The final line must read `reproduction of reports/results/benchmark_long384.json: PASS`. That means that at K = 30 the run reproduced the frozen run-4 metrics to within 1e-6. If it says FAIL, **stop**: push only `metrics.json` and `env.json` and report the mismatch in `research/PROGRESS.md`.
- If interrupted, rerun the same command unchanged. An interrupted run has no `metrics.json`, so the runner resumes, and finished 500-query blocks are reused from `runs/research/`, which is git-ignored.

## 4. F1: Stage-1 variants and training scale (~15–30 min)

```bash
python research/scripts/local/run_stage1_variants.py --config configs/full_train.yaml \
  --checkpoint full_long=runs/encoder_full_long/encoder_best.pt \
  --checkpoint full_60=runs/encoder_full_train/encoder_best.pt \
  --checkpoint base_1500=runs/sweep_base/encoder_best.pt \
  --primary full_long \
  --out research/results/F1_stage1_variants
```

If a secondary checkpoint is missing, drop its `--checkpoint` line and note it in `PROGRESS.md`. Keep `full_long`.

## 5. H1: alignment ablation (tier 2; ~10 min, or ~2 h with exhaustive)

```bash
python research/scripts/local/run_alignment_ablation.py --config configs/full_train.yaml \
  --checkpoint runs/encoder_full_long/encoder_best.pt --k 30 \
  --frames 96 384 --rotations none profile_cosine exhaustive_dtw \
  --out research/results/H1_alignment_ablation
```

Short on time? Use `--rotations none profile_cosine`, which takes about 10 min. Record the choice in `PROGRESS.md`.

## 6. Validate, then push only the small files

```bash
python research/scripts/validate_results.py        # must end "0 failed"
du -sh research/results/A1_* research/results/F1_* research/results/H1_*   # expect < 20 MB each
git add research/results/A1_k_sweep_long384 research/results/F1_stage1_variants research/results/H1_alignment_ablation research/PROGRESS.md
git commit -m "results: local runs A1, F1, H1 (LOCAL-FULL)"
git push origin claude/compassionate-cray-q8bxmt
```

Never `git add` any of these: `data/`, `runs/`, `*.npz`, `*.npy`, `*.pt`, `*.h5`. `.gitignore` already blocks them.

## Next cycle: A2, the K sweep with the fused Stage 1 (~80 min, resumable)

Registered 2026-09-25 in [`experiments/registry.yaml`](experiments/registry.yaml), after F1 showed a +3.8-point coverage gain for `win_fuse` and before any end-to-end result. Success criterion: paired ΔMAP versus A1 with intervals excluding zero at K = 30 **and** K = 100.

Start from a **clean, committed tree** (the tier-1 runs were made from a dirty tree; see `reviewer_audit.md`, round 2):

```bash
git fetch origin && git checkout claude/compassionate-cray-q8bxmt && git pull
git status --short                                   # must print nothing (untracked data/, runs/ are ignored)
pytest -q                                            # all pass
python research/scripts/local/run_shortlist_sweep.py --config configs/full_train.yaml --tag long384 \
  --set features.n_frames=384 --hub-correction reports/results/hubness_correction_n384.json \
  --stage1 win_fuse --max-queries 200 --out /tmp/sanity_A2          # ~5 min sanity; 200 rows expected
python research/scripts/local/run_shortlist_sweep.py --config configs/full_train.yaml --tag long384 \
  --set features.n_frames=384 --hub-correction reports/results/hubness_correction_n384.json \
  --stage1 win_fuse --ks 5 10 20 30 50 100 200 500 \
  --out research/results/A2_k_sweep_win_fuse
python research/scripts/validate_results.py                        # must end "0 failed"
git add research/results/A2_k_sweep_win_fuse research/PROGRESS.md
git commit -m "results: local run A2 (LOCAL-FULL)" && git push origin claude/compassionate-cray-q8bxmt
```

- There is no reproduction check: the frozen run used the global Stage 1, so `--frozen` is not passed.
- The fused score replaces the global score everywhere downstream, including the global term of the α blend. That is what "the system with a fused Stage 1" means. α and λ stay locked at the A1 values.
- The alignment cache key includes the Stage-1 scorer, so A1's cached blocks are not reused by mistake. The probe reference is reused.

## Optional, not recommended now

- **I2: 300-epoch training** (~4.6 h). The registry records why it is deprioritised: validation MAP was flat over epochs 121–150.
