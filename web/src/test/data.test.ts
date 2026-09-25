import { describe, expect, it } from 'vitest'
import { PARAMS, receptiveField, trainingRuns } from '../data/architecture'
import { benchmarkRuns, bestHybrid, run } from '../data/benchmark-data'
import { caseStudy } from '../data/retrieval-cases'
import { hpcp, rotationPairs } from '../data/figures'
import { rotationScores } from '../lib/chroma'
import { results } from '../data/results'

describe('architecture', () => {
  it('parameter count and receptive field match the training summary', () => {
    const s = trainingRuns.encoder_full_long.summary
    expect(PARAMS.total).toBe(s.n_parameters)
    expect(PARAMS.total).toBe(645504)
    expect(receptiveField(6)).toBe(s.receptive_field_frames)
  })
})

describe('benchmark numbers quoted in the README', () => {
  const r3 = (x: number) => Math.round(x * 1000) / 1000
  it('run 1 negative result', () => {
    const r = run('run1')
    expect(r3(r.metrics.classical_alignment.MAP!)).toBe(0.084)
    expect(r3(r.metrics['hybrid_K30_alpha0.05'].MAP!)).toBe(0.018)
    expect(r3(r.shortlist_recall)).toBe(0.021)
  })
  it('best systems after the experiments', () => {
    expect(r3(run('run2').metrics.classical_alignment_hubcorr.MAP!)).toBe(0.136)
    expect(r3(bestHybrid(run('run4')))).toBe(0.122)
    expect(r3(run('run4').shortlist_recall)).toBe(0.136)
    expect(run('run2').metrics.classical_alignment_hubcorr.median_first_rank).toBe(14)
  })
  it('every run covers the full protocol', () => {
    for (const r of benchmarkRuns) {
      expect(r.protocol.queries).toBe(13000)
      expect(r.protocol.candidates).toBe(15000)
    }
  })
})

describe('case study P_797406', () => {
  it('matches reports/error_analysis.md', () => {
    const cs = caseStudy()
    expect(cs.case.rank_global).toBe(15)
    expect(cs.case.rank_hybrid).toBe(1)
    expect(cs.case.rank_classical).toBe(1)
    expect(cs.case.partner).toBe('P_797408')
    expect(cs.top[0].candidate).toBe('P_797408')
    expect(cs.top[0].align).toBeCloseTo(0.979, 3)
  })
})

describe('decoded HPCP', () => {
  it('recovers the published best key shift for every pair', () => {
    for (const p of rotationPairs) {
      const s = rotationScores(hpcp[p.query].values, hpcp[p.candidate].values)
      expect(s.indexOf(Math.max(...s)), `${p.query}/${p.candidate}`).toBe(p.best_shift)
    }
  })
})

describe('research runs quoted in the README', () => {
  const research = results.research!
  const r3 = (x: number) => Math.round(x * 1000) / 1000
  it('the K sweep reproduces run 4 at K = 30 and reaches 0.212 at K = 500', () => {
    expect(research.reproduction_pass).toBe(true)
    const at = (k: number) => research.k_sweep.find((p) => p.K === k)!
    expect(at(30).hub_MAP).toBeCloseTo(bestHybrid(run('run4')), 9)
    expect(at(30).recall).toBeCloseTo(run('run4').shortlist_recall, 5)
    expect(r3(at(500).hub_MAP)).toBe(0.212)
  })
  it('every query is in exactly one failure class at every K', () => {
    for (const rows of Object.values(research.classes))
      for (const c of rows) expect(c.A_no_cover + c.B_hub_top1 + c.B_other + c.R_rank1).toBeCloseTo(1, 5)
  })
})

describe('A2 quoted in the README', () => {
  it('fused Stage 1: MAP 0.139 at K = 30 and the registered criterion is met', () => {
    const a2 = results.research!.a2!
    expect(a2.meets_registered_criterion).toBe(true)
    expect(Math.round(a2.k_sweep.find((p) => p.K === 30)!.hub_MAP * 1000) / 1000).toBe(0.139)
    for (const p of a2.k_sweep) expect(p.dAP_lo).toBeGreaterThan(0)
  })
})
