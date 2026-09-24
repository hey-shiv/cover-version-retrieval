/**
 * Final Da-TACOS benchmark: 13,000 queries x 15,000 candidates, 12 relevant items per query.
 *
 * Four separate evaluations, each with every setting locked beforehand (encoder from
 * validation works; alpha, lambda and resolution from calibration works). They are
 * kept apart here on purpose: numbers from different runs are never merged into one table.
 *
 * Sources (via generated/results.json):
 *   run1  reports/results/benchmark.json         + benchmark_per_query.csv
 *   run2  reports/results/benchmark_full96.json  + benchmark_per_query_full96.csv
 *   run3  reports/results/benchmark_full384.json + benchmark_per_query_full384.csv
 *   run4  reports/results/benchmark_long384.json + benchmark_per_query_long384.csv
 *
 * Note on run 4, Stage 1: benchmark_long384.json and its per-query CSV give
 * MRR 0.232, Hit@1 0.155, Hit@10 0.380, median first rank 26. The README and
 * reports/improvement_experiments.md print 0.297 / 0.234 / 0.373 / 23 for that row.
 * This site shows the result file's values.
 */
import { results, type BenchmarkRun, type Metrics } from './results'

export const benchmarkRuns: BenchmarkRun[] = results.benchmark.runs
export const rankGrid: number[] = results.benchmark.rank_grid
/** edges of the first-rank histogram bins [e_i, e_{i+1}); 31 is an edge (K = 30) */
export const rankEdges: number[] = results.benchmark.rank_edges

export const RUN_LABEL: Record<BenchmarkRun['id'], string> = {
  run1: 'Run 1',
  run2: 'Run 2',
  run3: 'Run 3',
  run4: 'Run 4',
}

/** Short encoder/alignment descriptors for each run (restating each JSON's locked block). */
export const RUN_SETUP: Record<BenchmarkRun['id'], { encoder: string; frames: string; hub: string }> = {
  run1: { encoder: '1,500 works · 60 ep', frames: '96', hub: '—' },
  run2: { encoder: '4,780 works · 60 ep', frames: '96', hub: 'λ 0.5' },
  run3: { encoder: '4,780 works · 60 ep', frames: '384', hub: 'λ 0.6' },
  run4: { encoder: '4,780 works · 150 ep', frames: '384', hub: 'λ 0.6' },
}

export type SystemKind = 'stage1' | 'hybrid' | 'rerank' | 'hybrid-hub' | 'classical' | 'classical-hub' | 'stage1-rot'

export function systemKind(key: string): SystemKind {
  if (key.startsWith('hybrid_hubcorr')) return 'hybrid-hub'
  if (key.startsWith('hybrid')) return 'hybrid'
  if (key.startsWith('rerank_only')) return 'rerank'
  if (key === 'classical_alignment_hubcorr') return 'classical-hub'
  if (key.startsWith('classical')) return 'classical'
  if (key === 'global_embedding_test_time_rotations') return 'stage1-rot'
  return 'stage1'
}

export const SYSTEM_LABEL: Record<SystemKind, string> = {
  stage1: 'Stage 1 · global embedding',
  'stage1-rot': 'Stage 1 + test-time rotations',
  hybrid: 'Hybrid · K = 30',
  rerank: 'Rerank only · α = 0',
  'hybrid-hub': 'Hybrid + hubness correction',
  classical: 'Classical alignment · all pairs',
  'classical-hub': 'Classical + hubness correction',
}

export function systemLabel(key: string): string {
  const kind = systemKind(key)
  const alpha = key.match(/alpha(\d\.\d+)/)?.[1]
  return alpha && kind !== 'rerank' ? `${SYSTEM_LABEL[kind]} · α = ${Number(alpha)}` : SYSTEM_LABEL[kind]
}

export const METRICS = ['MAP', 'MRR', 'Hit@1', 'Hit@10', 'median_first_rank'] as const
export type MetricKey = (typeof METRICS)[number]
export const METRIC_LABEL: Record<MetricKey, string> = {
  MAP: 'MAP',
  MRR: 'MRR',
  'Hit@1': 'Hit@1',
  'Hit@10': 'Hit@10',
  median_first_rank: 'Median first rank',
}

export function metric(m: Metrics, k: MetricKey): number {
  return m[k] ?? NaN
}

export const run = (id: BenchmarkRun['id']) => benchmarkRuns.find((r) => r.id === id)!

/** Stage-1 system key of a run (always "global_embedding"). */
export const STAGE1 = 'global_embedding'

/** Best hybrid MAP of a run (any hybrid or rerank variant). */
export function bestHybrid(r: BenchmarkRun): number {
  return Math.max(...Object.entries(r.metrics).filter(([k]) => /hybrid|rerank/.test(k)).map(([, m]) => m.MAP ?? 0))
}

/**
 * Best system in the benchmark at the time of each run. Run 1 = its own classical alignment;
 * runs 2-4 = classical + hubness correction from run 2 (0.136), the only evaluation of it.
 */
export function bestOverall(r: BenchmarkRun): { map: number; from: string } {
  if (r.id === 'run1') return { map: r.metrics.classical_alignment.MAP!, from: 'run 1 · classical alignment' }
  return { map: run('run2').metrics.classical_alignment_hubcorr.MAP!, from: 'run 2 · classical + hubness correction' }
}

/**
 * Published Da-TACOS benchmark results for context, as tabulated in
 * notes/literature/novelty_assessment.md (Yesiler et al. 2019, Table 2; later papers as cited there).
 * Same protocol: 1,000 cliques of 13 plus 2,000 noise tracks.
 */
export const PUBLISHED = [
  { system: 'Qmax (Serrà et al. 2009)', input: 'HPCP', MAP: 0.333, source: 'Yesiler et al. 2019' },
  { system: 'Dmax', input: 'HPCP', MAP: 0.292, source: 'Yesiler et al. 2019' },
  { system: 'EarlyFusion', input: 'HPCP', MAP: 0.426, source: 'Yesiler et al. 2019' },
  { system: 'MOVE', input: 'CREMA', MAP: 0.507, source: 'Yesiler, Serrà & Gómez 2020' },
  { system: 'ByteCover', input: 'CQT', MAP: 0.714, source: 'Du et al. 2021' },
] as const
