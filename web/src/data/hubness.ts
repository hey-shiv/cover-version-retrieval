/**
 * Hubness (reports/error_analysis.md, reports/hybrid_results.md §5, D-018).
 *
 * Sources (via generated/results.json):
 *   reports/results/hubness_dev.json                    Spearman, expected count, top hubs + dispersion
 *   reports/results/classical_dev_rankings.csv          classical top-10 false positives per query (graph edges)
 *   reports/results/hybrid_dev_rankings_top10.csv       hybrid top-10 false positives per query
 *   reports/results/hubness_correction{,_n384}.json     λ grid on calibration works, dev effect
 */
import { results } from './results'

export const hub = results.hubness

/** Hub graph: one node per candidate that appears as a top-10 false positive, with the queries it hit. */
export function hubGraph(system: 'classical' | 'hybrid') {
  const edges = system === 'classical' ? hub.edges_classical : hub.edges_hybrid
  const byCandidate = new Map<string, string[]>()
  for (const [q, c] of edges) byCandidate.set(c, [...(byCandidate.get(c) ?? []), q])
  const queries = [...new Set(results.dev.runs[0].queries.map((q) => q.query))]
  const candidates = [...byCandidate.entries()].map(([pid, qs]) => ({ pid, queries: qs })).sort((a, b) => b.queries.length - a.queries.length)
  const known = new Map(hub.summary[system].top_hubs.map((h) => [h.pid, h.dispersion]))
  return { queries, candidates, dispersion: known, edges }
}
