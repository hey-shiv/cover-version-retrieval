/**
 * Development protocol (Da-TACOS Cover Analysis): 20 queries x 120 candidates, one relevant
 * item per query, self excluded — so AP = reciprocal rank and MAP = MRR by construction.
 *
 * Sources (via generated/results.json):
 *   reports/results/hybrid_dev{,_n384,_full96,_full384,_long384}.json   metrics, K sweep, bootstrap
 *   reports/results/hybrid_dev_per_query*.csv                           rank of the partner per system
 *   reports/results/error_cases*.csv                                    deterministically selected cases
 *   reports/results/hybrid_dev_rankings_top10*.csv                      hybrid top 10 (score = global cosine)
 *   reports/results/classical_dev_rankings.csv                          alignment score + key shift, every pair
 *   reports/error_analysis.md                                           tonal dispersion quoted for the 9 base cases
 */
import { results, type DevRun, type DevCase } from './results'

export const devRuns: DevRun[] = results.dev.runs
export const devRun = (id: DevRun['id']) => devRuns.find((r) => r.id === id)!

export const CASE_LABEL: Record<DevCase['case'], string> = {
  success: 'Success',
  false_positive: 'False positive',
  false_negative: 'False negative',
}

/**
 * Tonal dispersion (mean cosine distance of a track's 96 frames from its own mean chroma)
 * of [query, partner, top false positive] for the six non-trivial base-run cases, quoted from the
 * tables in reports/error_analysis.md. Low = harmonically static. Dev-pool median 0.053.
 */
export const DISPERSION_MEDIAN = 0.053
export const DISPERSION: Record<string, [number, number, number]> = {
  P_310762: [0.047, 0.031, 0.02],
  P_419983: [0.045, 0.062, 0.031],
  P_207372: [0.055, 0.143, 0.041],
  P_29344: [0.011, 0.042, 0.024],
  P_411001: [0.04, 0.055, 0.024],
  P_385477: [0.073, 0.067, 0.05],
}

/** Short hypotheses quoted/condensed from reports/error_analysis.md (base run only). */
export const CASE_NOTES: Record<string, string> = {
  P_797406:
    'The clearest case of reranking fixing Stage 1: the embedding put the partner 15th; alignment gives it the best score in the shortlist. Same key, almost the same length (ratio 0.98).',
  P_36279: 'Already rank 1 everywhere. The partner was found at a key shift of 8 semitones and is about 1.9× longer in raw frames.',
  P_495015: 'Rank 1 in all systems, with a key shift of 5 semitones.',
  P_310762:
    'Stage-1 recall failure: classical alignment ranks the partner 11th, but the embedding scored it too low (cos 0.84) to reach the shortlist.',
  P_419983: 'Stage-1 recall failure: alignment alone ranks the partner 15th; the shortlist never sees it.',
  P_207372:
    'The reranker hurt: partner at global rank 5 fell to 22. It is the most tonally varied track of the nine cases (dispersion 0.143), and DTW at 96 frames favours static material.',
  P_29344:
    'A static query (dispersion 0.011, least varied in the pool) meets a static hub. Its cost matrix against the non-cover is almost uniformly minimal, so any path is cheap.',
  P_411001: 'Both systems rank the same hub (P_576649) first. Hypothesis: an arrangement or structure change whole-query DTW cannot model.',
  P_385477: 'Partner is 1.37× longer in raw frames. Hypothesis: added sections force whole-query alignment through unrelated material.',
}

/* ---------------------------------------------------------------- the case study */

/** P_797406 (W_203093) → true partner P_797408, base run (1,500-work encoder, 96 frames, K = 30, α = 0.05). */
export const CASE_QUERY = 'P_797406'
export const CASE_PARTNER = 'P_797408'

export function caseStudy() {
  const base = devRun('base')
  const c = base.cases.find((x) => x.query === CASE_QUERY)!
  const pairs = results.classical.pairs[CASE_QUERY]
  // The hybrid's final top 10: global cosine from hybrid_dev_rankings_top10.csv,
  // alignment score + shift for the same pair from classical_dev_rankings.csv
  // (the reranker uses the same 96-frame alignment function).
  const top = base.top10
    .filter((t) => t.query === CASE_QUERY)
    .map((t) => ({ ...t, align: pairs[t.candidate].score, shift: pairs[t.candidate].shift }))
  return { case: c, top, alpha: base.locked.alpha, k: base.locked.shortlist_k, nCandidates: 119 }
}
