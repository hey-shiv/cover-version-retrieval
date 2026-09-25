/**
 * Typed access to web/src/data/generated/results.json, which web/scripts/build_data.py
 * copies from reports/results/ (every block carries its `source`). Nothing in it is typed by hand.
 */
import raw from './generated/results.json'

export type Metrics = Partial<
  Record<'MAP' | 'MRR' | 'Hit@1' | 'Hit@10' | 'Hit@100' | 'Recall@10' | 'Recall@100' | 'mean_first_rank' | 'median_first_rank', number>
>
export interface Interval { delta: number; lo: number; hi: number }

export interface FirstRank { cdf: number[]; log_bins: number[]; in_shortlist_30: number; median: number }

export interface BenchmarkRun {
  id: 'run1' | 'run2' | 'run3' | 'run4'
  source: string
  per_query_source: string
  context: string
  locked: { alpha: number; shortlist_k: number; checkpoint: string; n_frames?: number; hub_correction?: { lambda: number; method: string; source: string } }
  protocol: { queries: number; candidates: number; noise_tracks: number }
  metrics: Record<string, Metrics>
  shortlist_recall: number
  runtime: Record<string, number>
  bootstrap_vs_global: Record<string, Interval>
  git_commit: string
  first_rank: Record<string, FirstRank>
}

export interface DevCase {
  case: 'success' | 'false_positive' | 'false_negative'
  query: string; partner: string
  rank_hybrid: number; rank_global: number; rank_classical: number
  partner_cos: number; partner_align: number; partner_shift: number; in_shortlist: boolean
  fp: string; fp_rank: number; fp_cos: number; fp_align: number; fp_shift: number
  figure: string
}
export interface DevRun {
  id: 'base' | 'n384' | 'full96' | 'full384' | 'long384'
  context: string
  source: string
  locked: { alpha: number; shortlist_k: number }
  shortlist_recall: number
  metrics: Record<string, Metrics>
  systems: string[]
  queries: { query: string; wid: string; ranks: Record<string, number> }[]
  k_sweep: { K: number; recall: number; MAP: number }[]
  bootstrap: Record<string, Interval>
  cases: DevCase[]
  top10: { query: string; rank: number; candidate: string; cos: number; relevant: boolean }[]
}

export interface KPoint {
  K: number; coverage: number; recall: number; ms_per_query: number
  hyb_MAP: number; 'hyb_Hit@1': number; hub_MAP: number; 'hub_Hit@1': number; rr_MAP: number; 'rr_Hit@1': number
}
export interface ClassPoint { K: number; A_no_cover: number; B_hub_top1: number; B_other: number; R_rank1: number; efficiency: number }
export interface AdaptiveCell {
  menu: string; budget: number; mean_K: number
  dAP: number; dAP_lo: number; dAP_hi: number; dHit1: number; dHit1_lo: number; dHit1_hi: number; meets: boolean
}
export interface Research {
  source: string
  evidence: Record<'a1' | 'b1' | 'e1' | 'f1', string>
  reproduction_pass: boolean
  n_queries: number
  ms_per_pair: number
  k_sweep: KPoint[]
  classes: Record<'hyb' | 'hub', ClassPoint[]>
  adaptive: AdaptiveCell[]
  stage1_coverage30: { variant: string; delta: number; lo: number; hi: number }[]
  a2: {
    source: string
    evidence: string
    meets_registered_criterion: boolean
    k_sweep: { K: number; coverage: number; hub_MAP: number; 'hub_Hit@1': number; dAP: number; dAP_lo: number; dAP_hi: number; dHit1: number }[]
  } | null
}

interface ResultsFile {
  research: Research | null
  benchmark: { rank_grid: number[]; rank_edges: number[]; runs: BenchmarkRun[] }
  dev: { protocol: { n_queries: number; n_candidates: number }; runs: DevRun[] }
  classical: {
    source: string
    metrics: Record<string, Metrics>
    resolution_sweep: { protocol: string; n_frames: number; MAP: number; ms_per_pair: number }[]
    rankings_source: string
    pairs: Record<string, Record<string, { rank: number; score: number; shift: number; relevant: boolean }>>
    shift_histogram: number[]
  }
  hubness: {
    source: string
    summary: {
      n_candidates: number
      expected_top10_fp_count_if_uniform: number
      dispersion_median: number
      classical: { spearman_dispersion_vs_top10_fp: number; top_hubs: { pid: string; top10_fp_count: number; dispersion: number }[] }
      hybrid: { spearman_dispersion_vs_top10_fp: number; top_hubs: { pid: string; top10_fp_count: number; dispersion: number }[] }
    }
    edges_classical: [string, string][]
    edges_hybrid: [string, string][]
    edges_source: string[]
    corrections: {
      source: string
      n_frames: number
      locked: { method: string; k: number; lambda: number }
      grid: { method: string; k: number; lambda: number; MAP: number }[]
      dev_metrics: Record<string, Metrics>
      dev_bootstrap: { delta: number; ci_low: number; ci_high: number }
      reference_vs_dispersion_spearman: number
    }[]
  }
  training: {
    runs: Record<string, {
      source: string
      summary: { best_epoch: number; best_val_map: number; epochs_run: number; n_parameters: number; receptive_field_frames: number; train_wids: number; total_seconds: number }
      encoder: Record<string, number | string>
      training: Record<string, number | string | boolean>
      epochs: [number, number, number][]
    }>
    sweep: Record<string, string>[]
    sweep_source: string
  }
  calibration: Record<string, { source: string; alpha: number; shortlist_k: number; shortlist_recall: number; table: [number, number][] }>
  dataset: {
    source: string
    seed: number
    n_metadata_wids: number
    role_counts: Record<string, { wids: number; tracks: number }>
    n_excluded_wids: number
  }
}

export const results = raw as unknown as ResultsFile
