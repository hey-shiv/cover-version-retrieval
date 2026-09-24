/**
 * The research log, in order. Dates and commit hashes from `git log`; decisions from
 * notes/decisions.md; numbers from the result files named in each entry (read through
 * benchmark-data.ts / retrieval-cases.ts wherever a number is shown).
 */
export interface LogEntry {
  date: string
  commit: string
  kind: 'build' | 'evaluate' | 'negative' | 'diagnose' | 'experiment' | 'lesson'
  title: string
  body: string
  decisions?: string[]
  evidence: string[]
}

export const LOG: LogEntry[] = [
  {
    date: '2026-09-22',
    commit: '55e914a',
    kind: 'build',
    title: 'Loaders, classical alignment, TCN encoder, hybrid retrieval',
    body: 'Everything written from textbook recurrences: 12-rotation key handling, cosine cross-similarity, slope-constrained subsequence DTW, a 645,504-parameter TCN trained with SupCon. No cover-song library imported.',
    decisions: ['D-002', 'D-003', 'D-004', 'D-005', 'D-006', 'D-011'],
    evidence: ['src/cover_retrieval/', 'tests/'],
  },
  {
    date: '2026-09-22',
    commit: '5a54711',
    kind: 'evaluate',
    title: 'Development protocol: the hybrid looks best',
    body: 'On 20 queries against 120 candidates the hybrid (0.262) edges out classical alignment (0.248) and Stage 1 (0.189). Only rerank-only over Stage 1 clears zero in the bootstrap, and barely.',
    decisions: ['D-012', 'D-016'],
    evidence: ['reports/results/hybrid_dev.json', 'reports/hybrid_results.md'],
  },
  {
    date: '2026-09-23',
    commit: '98b66e7',
    kind: 'negative',
    title: 'Benchmark run 1: the ordering reverses',
    body: 'At 13,000 × 15,000, exhaustive classical alignment (0.084) beats the hybrid (0.018). Reported as the headline, not buried. Nothing re-tuned, the run not repeated.',
    decisions: ['D-017'],
    evidence: ['reports/results/benchmark.json'],
  },
  {
    date: '2026-09-23',
    commit: '98b66e7',
    kind: 'diagnose',
    title: 'Why: two measured mechanisms',
    body: 'Shortlist recall is 0.021: only 2.1% of relevant items ever reach the reranker. And tonally static tracks act as hubs — dispersion vs. top-10 false-positive count, Spearman ρ = −0.77.',
    evidence: ['reports/error_analysis.md', 'reports/results/hubness_dev.json'],
  },
  {
    date: '2026-09-23',
    commit: 'c55f0bf',
    kind: 'experiment',
    title: 'Hubness correction · 384-frame reranking',
    body: 'Penalise each candidate by its mean score against 200 fixed training-split probes, λ chosen on 10 calibration works. Separately, align the shortlist at 384 frames — affordable because only 0.2% of pairs are aligned.',
    decisions: ['D-018', 'D-019'],
    evidence: ['reports/results/hubness_correction.json', 'reports/results/hybrid_dev_n384.json'],
  },
  {
    date: '2026-09-23',
    commit: 'ecd00e9',
    kind: 'experiment',
    title: 'Encoder on all 4,780 training works → run 2',
    body: 'The 1,500-work limit had been a bandwidth artefact. Validation MAP 0.181 → 0.319 from data alone. Run 2 also measures classical + hubness correction: 0.136, the best system in the project.',
    decisions: ['D-020'],
    evidence: ['reports/results/benchmark_full96.json'],
  },
  {
    date: '2026-09-23',
    commit: '00e1b39',
    kind: 'evaluate',
    title: 'Run 3: 384-frame reranking at scale',
    body: 'Hybrid + hubness correction at 384 frames: 0.068. Still half the corrected exhaustive alignment. Shortlist recall 0.072 is the ceiling.',
    evidence: ['reports/results/benchmark_full384.json'],
  },
  {
    date: '2026-09-23',
    commit: '922de70',
    kind: 'experiment',
    title: 'Train to convergence (150 epochs) → run 4',
    body: 'Validation MAP 0.319 → 0.431. Stage 1 doubles; shortlist recall reaches 0.136; the hybrid reaches 0.122 — within 1.11× of corrected exhaustive alignment at 38 ms of reranking per query instead of 2.63 h for the whole catalogue.',
    decisions: ['D-021'],
    evidence: ['reports/results/benchmark_long384.json', 'reports/results/training_summary_encoder_full_long.json'],
  },
  {
    date: '2026-09-23',
    commit: '922de70',
    kind: 'lesson',
    title: 'The small protocol misled twice',
    body: 'It preferred the hybrid over classical alignment (reversed at scale), and with a strong Stage 1 it said reranking hurts (−0.157) while the benchmark shows it helps (+0.037, interval excluding zero).',
    decisions: ['D-022'],
    evidence: ['reports/results/hybrid_dev_long384.json', 'reports/improvement_experiments.md'],
  },
  {
    date: '2026-09-23',
    commit: '879c9e3',
    kind: 'lesson',
    title: 'Literature check: the idea is not new',
    body: 'Hubness correction for cover song identification is prior art (Seo 2022; Li & Chen 2018; Hu & Chen 2019). What is new is evidence at Da-TACOS scale with held-out calibration — on a baseline about 4× weaker than Qmax.',
    evidence: ['notes/literature/novelty_assessment.md'],
  },
]
