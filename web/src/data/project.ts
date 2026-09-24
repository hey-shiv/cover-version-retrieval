/**
 * Project facts shown outside the result tables. Sources named per item.
 */
import { results } from './results'

/** `pytest --collect-only` at commit 879c9e3: 119 tests collected, 119 passed (the README still says 112). */
export const TEST_COUNT = 119

export const SEED = results.dataset.seed // data/manifests/manifest_info.json

/** data/manifests/manifest_info.json → role_counts (Cover Analysis, WID-disjoint roles). */
export const ROLES = results.dataset.role_counts

/** DATASET_CARD.md */
export const DATASET = {
  name: 'Da-TACOS',
  citation:
    'Yesiler, Tralie, Correya, Silva, Tovstogan, Gómez, Serra. Da-TACOS: A Dataset for Cover Song Identification and Understanding. ISMIR 2019, pp. 327–334.',
  license: 'CC BY-NC-SA 4.0 · © Music Technology Group, Universitat Pompeu Fabra',
  zenodo: 'https://doi.org/10.5281/zenodo.4717628',
  homepage: 'https://mtg.github.io/da-tacos/',
  coverAnalysis: { tracks: 10000, works: 5000, cliqueSize: 2 },
  benchmark: { tracks: 15000, works: 3000, cliques: 1000, cliqueSize: 13, noise: 2000 },
  archivesGB: 17,
} as const

/** README "Reproducibility controls" and D-008/D-015/D-016. */
export const REPRO = [
  {
    key: 'SEED',
    value: String(SEED),
    body: 'One seed for split permutation, batch order, augmentation and initialisation (utils/seed.py). Training runs on CPU with torch.use_deterministic_algorithms and a fixed thread count; a test checks two runs give bit-identical weights.',
    where: 'configs/base.yaml · src/cover_retrieval/utils/seed.py',
  },
  {
    key: 'MANIFEST',
    value: 'ID-only CSV',
    body: 'Every selected WID/PID and its role is committed. Rebuilding is byte-identical (tested). Candidates are sorted by PID, so score ties break deterministically.',
    where: 'data/manifests/',
  },
  {
    key: 'CACHE',
    value: 'fingerprinted',
    body: 'Preprocessed features are keyed by a hash of the track list and every preprocessing setting, so a changed setting can never silently reuse a stale cache.',
    where: 'src/cover_retrieval/features/preprocessing.py',
  },
  {
    key: 'MODEL',
    value: 'selected on validation',
    body: 'The encoder checkpoint is chosen by validation MAP on 150 disjoint works. Retraining from a fresh clone reproduced the selected checkpoint bit-for-bit (epoch 50, validation MAP 0.18097).',
    where: 'reports/results/training_summary_*.json',
  },
  {
    key: 'CALIBRATION',
    value: 'locked to file',
    body: 'α, K, λ and resolution come from 10 calibration works and are written to hybrid_calibration*.json before any development or benchmark query is scored. run_benchmark.py refuses to start without it and tunes nothing.',
    where: 'reports/results/hybrid_calibration*.json',
  },
  {
    key: 'EVALUATION',
    value: 'provenance in every file',
    body: 'Each result JSON records the git commit and config path. Every benchmark run was evaluated once; no run was repeated after its result was seen.',
    where: 'reports/results/benchmark*.json',
  },
] as const

/** README → "Next research directions", in the README's order (by what the evidence demands). */
export const FUTURE = [
  { id: 'recall', title: 'Keep pushing Stage-1 recall', body: 'Validation MAP was still drifting upward at epoch 150; shortlist recall (0.136) still caps the hybrid. More epochs, larger K, multi-vector (per-section) embeddings.', target: 'Stage 1' },
  { id: 'classical384', title: 'Classical 384 frames + hubness correction', body: 'Over the full benchmark, about 42 h of CPU. The most likely strongest system; never run.', target: 'Stage 2' },
  { id: 'ttr', title: 'Test-time rotation matching at scale', body: 'Best development system (0.436 with the 60-epoch encoder), 12× the query-embedding cost. Never evaluated on the benchmark.', target: 'Stage 1' },
  { id: 'local', title: 'Local alignment (Qmax-style)', body: 'Versus whole-query subsequence DTW, for covers that drop, add or reorder sections.', target: 'Stage 2' },
  { id: 'fusion', title: 'CREMA / HPCP feature fusion', body: 'And error-stratified evaluation by rotation shift and length ratio.', target: 'Representation' },
] as const
