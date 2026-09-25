import { useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { results, type ClassPoint, type KPoint, type Research } from '../data/results'
import { benchmarkRuns } from '../data/benchmark-data'
import { useWidth } from '../lib/hooks'
import { PAPER_URL } from '../components/Nav'
import { linScale, logScale } from '../lib/format'

/**
 * Research runs after the four frozen evaluations: the shortlist-size sweep (A1), failure classes by K (B1),
 * adaptive K (E1) and Stage-1 variants (F1). All from research/results/ via generated/results.json.
 */
export function Beyond() {
  const res = results.research
  if (!res) return null
  const at = (k: number) => res.k_sweep.find((p) => p.K === k)!
  const hub = res.classes.hub
  const cls = (k: number) => hub.find((c) => c.K === k)!
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`
  const failing = res.adaptive.filter((c) => c.dAP_lo > 0)
  return (
    <Chapter id="beyond" no="11 · Beyond thirty" title={<>More candidates, <em>different failures</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            K = 30 was chosen for cost, not quality. Aligning the top {at(500).K} instead raises MAP from {at(30).hub_MAP.toFixed(3)} to{' '}
            <strong>{at(500).hub_MAP.toFixed(3)}</strong>, and it has not levelled off.
          </p>
          <p>
            Every candidate list from 5 to 500 comes from one alignment pass over each query's top 500. At K = 30 that pass reproduces the frozen run 4
            exactly{res.reproduction_pass ? '' : ' (reproduction check failed)'}. As K grows, the kind of failure changes. Up to K = 50 most queries fail
            because no cover reached the shortlist ({pct(cls(50).A_no_cover)} of them at K = 50). From K = 200, most fail because a cover is in the
            list but a non-cover outranks it ({pct(cls(200).B_hub_top1 + cls(200).B_other)} vs {pct(cls(200).A_no_cover)}).
          </p>
        </div>
        <aside className="aside">
          Research runs on all {res.n_queries.toLocaleString('en-US')} benchmark queries, made on the author's laptop after the four frozen
          evaluations. K is swept, not selected: no setting was tuned on these numbers. Alignment cost {res.ms_per_pair.toFixed(2)} ms per pair,
          measured.
        </aside>
      </div>
      <Sweep res={res} />
      <Classes res={res} />
      <div className="figure two-up">
        <div className="prose">
          <h3 className="sub">Choosing K per query did not help</h3>
          <p>
            Stage-1 score statistics predict which queries will have no cover in the top 30 (AUROC 0.76). But a policy that spends alignments where
            they seem most useful, at the same average cost as a fixed K, never improved both metrics. In {failing.length} of {res.adaptive.length} settings it raised MAP a little, and in every one of those it
            lowered Hit@1. The criterion was registered before the run, so this is reported as a negative result.
          </p>
        </div>
        <Adaptive res={res} />
      </div>
      <Stage1 res={res} />
      <div className="split">
        <div className="body prose">
          <h3 className="sub">The argument, in one paragraph</h3>
          <p>
            In coarse-to-fine cover retrieval, the shortlist is a measurable bottleneck. Alignment can only reorder covers the first stage admits. At
            K = 30 exclusion is the main failure, and improving the first stage produced most of the gains. Enlarging the shortlist keeps paying but
            hands the bottleneck to the reranker. A 20-query development set could not have shown any of this.{' '}
            <a href={PAPER_URL}>Read the paper ↗</a>
          </p>
        </div>
        <aside className="aside">
          The paper is a draft. Every number in it is generated from the same result files as this page, and its audits list what is still
          unverified.
        </aside>
      </div>
    </Chapter>
  )
}

type SweepMetric = 'MAP' | 'Hit@1' | 'coverage'

function Sweep({ res }: { res: Research }) {
  const [m, setM] = useState<SweepMetric>('MAP')
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const H = 300
  const L = 48
  const R = width < 560 ? 16 : 190
  const pts = res.k_sweep
  const x = logScale([pts[0].K, pts[pts.length - 1].K], [L, width - R])
  const run2 = benchmarkRuns[1].metrics.classical_alignment_hubcorr
  const fused = new Map((res.a2?.k_sweep ?? []).map((p) => [p.K, p]))
  const hasFused = pts.every((p) => fused.has(p.K))
  const series: { key: string; label: string; color: string; dash?: string; get: (p: KPoint) => number }[] =
    m === 'coverage'
      ? [
          { key: 'cov', label: 'coverage (≥ 1 cover in top K)', color: 'var(--query)', get: (p) => p.coverage },
          { key: 'rec', label: 'recall (all 12 covers)', color: 'var(--query)', dash: '5 3', get: (p) => p.recall },
          ...(hasFused ? [{ key: 'fcov', label: 'coverage, fused', color: 'var(--cover)', get: (p: KPoint) => fused.get(p.K)!.coverage }] : []),
        ]
      : [
          { key: 'hub', label: 'hybrid + hub corr.', color: 'var(--path)', get: (p) => (m === 'MAP' ? p.hub_MAP : p['hub_Hit@1']) },
          { key: 'hyb', label: 'hybrid', color: 'var(--path)', dash: '5 3', get: (p) => (m === 'MAP' ? p.hyb_MAP : p['hyb_Hit@1']) },
          { key: 'rr', label: 'rerank only', color: 'var(--faint)', get: (p) => (m === 'MAP' ? p.rr_MAP : p['rr_Hit@1']) },
          ...(hasFused
            ? [{ key: 'fuse', label: 'fused + hub corr.', color: 'var(--cover)', get: (p: KPoint) => (m === 'MAP' ? fused.get(p.K)!.hub_MAP : fused.get(p.K)!['hub_Hit@1']) }]
            : []),
        ]
  const refVal = m === 'MAP' ? run2.MAP : m === 'Hit@1' ? run2['Hit@1'] : undefined
  const vals = series.flatMap((s) => pts.map(s.get)).concat(refVal ?? [])
  const lo = m === 'coverage' ? 0 : m === 'MAP' ? 0.06 : 0.2
  const top = m === 'coverage' ? 1 : Math.max(...vals) * 1.05
  const y = linScale([lo, top], [H - 44, 12])
  const ticks = m === 'coverage' ? [0, 0.25, 0.5, 0.75, 1] : m === 'MAP' ? [0.08, 0.12, 0.16, 0.2] : [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
  // end labels: keep at least 13 px apart
  const ends = series.map((s) => y(s.get(pts[pts.length - 1])))
  const order = ends.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0])
  for (let j = 1; j < order.length; j++) if (order[j][0] - order[j - 1][0] < 13) order[j][0] = order[j - 1][0] + 13
  const labelY = Object.fromEntries(order.map(([v, i]) => [i, v]))
  return (
    <div className="figure">
      <div className="controls">
        <Seg label="Metric" value={m} onChange={setM} options={[{ value: 'MAP', label: 'MAP' }, { value: 'Hit@1', label: 'Hit@1' }, { value: 'coverage', label: 'Stage-1 coverage' }]} />
        <span className="label">log axis · second row: rerank time per query</span>
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`${m} against shortlist size K from 5 to 500`}>
          {ticks.map((v) => (
            <g key={v}>
              <line x1={L} x2={width - R} y1={y(v)} y2={y(v)} stroke="var(--rule)" strokeDasharray={v ? '2 3' : undefined} />
              <text x={L - 6} y={y(v) + 4} textAnchor="end" className="axis-t">
                {m === 'MAP' ? v.toFixed(2) : `${Math.round(v * 100)}%`}
              </text>
            </g>
          ))}
          {pts.map((p) => (
            <g key={p.K}>
              <text x={x(p.K)} y={H - 26} textAnchor="middle" className="axis-t" fontWeight={p.K === 30 ? 600 : undefined}>
                {p.K}
              </text>
              {(width >= 560 || [5, 30, 100, 500].includes(p.K)) && (
                <text x={x(p.K)} y={H - 10} textAnchor="middle" className="axis-t" fill="var(--mute)">
                  {p.ms_per_query < 10 ? p.ms_per_query.toFixed(0) : Math.round(p.ms_per_query)} ms
                </text>
              )}
            </g>
          ))}
          <line x1={x(30)} x2={x(30)} y1={12} y2={H - 44} stroke="var(--ink)" />
          {refVal !== undefined && (
            <g>
              <line x1={L} x2={width - R} y1={y(refVal)} y2={y(refVal)} stroke="var(--hub)" strokeWidth={1.2} strokeDasharray="1 3" />
              <text x={L + 6} y={y(refVal) - 6} className="axis-t" fill="var(--hub)">
                {width < 560 ? 'exhaustive + hub corr. (run 2)' : 'exhaustive alignment + hub corr. (run 2)'}
              </text>
            </g>
          )}
          {series.map((s, si) => {
            const d = pts.map((p, j) => `${j ? 'L' : 'M'}${x(p.K).toFixed(1)},${y(s.get(p)).toFixed(1)}`).join(' ')
            const last = pts[pts.length - 1]
            return (
              <g key={s.key}>
                <path d={d} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dash} />
                {pts.map((p) => (
                  <circle key={p.K} cx={x(p.K)} cy={y(s.get(p))} r={2.6} fill={s.color} />
                ))}
                {width >= 560 && (
                  <text x={x(last.K) + 8} y={labelY[si] + 4} className="axis-t" fill={s.color}>
                    {s.label} {m === 'MAP' ? s.get(last).toFixed(3) : `${(s.get(last) * 100).toFixed(1)}%`}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
      {width < 560 && (
        <ul className="ecdf-legend">
          {series.map((s) => (
            <li key={s.key}>
              <svg width="22" height="8" aria-hidden="true">
                <line x1="0" x2="22" y1="4" y2="4" stroke={s.color} strokeWidth={2.4} strokeDasharray={s.dash} />
              </svg>
              {s.label}
            </li>
          ))}
        </ul>
      )}
      <Caption label="Figure 17" source={[`research/results/A1_k_sweep_long384 (${res.evidence.a1})`, ...(res.a2 ? [`${res.a2.source} (${res.a2.evidence})`] : []), 'reports/results/benchmark_full96.json (reference line)']}>
        Quality against shortlist size, with every other setting as locked for run 4. Vertical line: the original K = 30. Dotted line: exhaustive alignment of all 14,999 candidates. Rerank time is the measured
        cost per alignment times K. Green: the same system with the fused global + window Stage 1 (registered run A2). The dotted reference is a different run with 96-frame alignment, so the crossing is indicative, not a paired test.
      </Caption>
    </div>
  )
}

const CLASS_PARTS: { key: keyof ClassPoint; label: string; color: string }[] = [
  { key: 'A_no_cover', label: 'no cover in top K', color: 'var(--query)' },
  { key: 'B_hub_top1', label: 'cover present, a hub at rank 1', color: 'var(--hub)' },
  { key: 'B_other', label: 'cover present, other non-cover at rank 1', color: 'var(--path)' },
  { key: 'R_rank1', label: 'cover at rank 1', color: 'var(--cover)' },
]

function Classes({ res }: { res: Research }) {
  const [sys, setSys] = useState<'hub' | 'hyb'>('hub')
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const rows = res.classes[sys]
  const H = 240
  const L = 44
  const bw = Math.min(56, ((width - L - 16) / rows.length) * 0.7)
  const step = (width - L - 16) / rows.length
  const y = linScale([0, 1], [H - 26, 8])
  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">Where the queries fail, as K grows</h3>
          <p>
            Blue shrinks: more queries get a cover into the list. Red and ochre grow: more of those covers are then outranked. Of queries with a cover
            in the list, the share put first falls from {(rows[0].efficiency * 100).toFixed(0)}% at K = {rows[0].K} to{' '}
            {(rows[rows.length - 1].efficiency * 100).toFixed(0)}% at K = {rows[rows.length - 1].K}. Switch off the hub correction to see the ochre band
            swell.
          </p>
        </div>
      </div>
      <div className="controls">
        <Seg label="Reranker" value={sys} onChange={setSys} options={[{ value: 'hub', label: 'with hub correction' }, { value: 'hyb', label: 'without' }]} />
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Share of queries in each outcome class for each K">
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <text key={v} x={L - 6} y={y(v) + 4} textAnchor="end" className="axis-t">
              {v * 100}%
            </text>
          ))}
          {rows.map((r, i) => {
            let acc = 0
            const cx = L + step * i + step / 2
            return (
              <g key={r.K}>
                {CLASS_PARTS.map((p) => {
                  const v = r[p.key] as number
                  const y0 = y(acc)
                  acc += v
                  return <rect key={p.key} x={cx - bw / 2} y={y(acc) + 1} width={bw} height={Math.max(0, y0 - y(acc) - 2)} fill={p.color} />
                })}
                <text x={cx} y={H - 8} textAnchor="middle" className="axis-t" fontWeight={r.K === 30 ? 600 : undefined}>
                  {width < 560 ? r.K : `K ${r.K}`}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <ul className="ecdf-legend">
        {CLASS_PARTS.map((p) => (
          <li key={p.key}>
            <svg width="12" height="12" aria-hidden="true">
              <rect width="12" height="12" fill={p.color} />
            </svg>
            {p.label}
          </li>
        ))}
      </ul>
      <Caption label="Figure 18" source={`research/results/B1_failure_by_K (${res.evidence.b1})`}>
        Every benchmark query in exactly one class at each K. A hub is a rank-1 non-cover whose mean score against 200 fixed probe tracks is above the
        95th percentile of the catalogue, a threshold fixed before the run.
      </Caption>
    </div>
  )
}

function Adaptive({ res }: { res: Research }) {
  const sign = (v: number, d: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`
  return (
    <div className="scroll-x">
      <table className="data">
        <thead>
          <tr>
            <th>menu of K</th>
            <th className="n">budget</th>
            <th className="n">Δ MAP</th>
            <th className="n">Δ Hit@1 (points)</th>
          </tr>
        </thead>
        <tbody>
          {res.adaptive
            .filter((c) => c.budget !== 10)
            .map((c) => (
              <tr key={`${c.menu}-${c.budget}`}>
                <td className="mono">{c.menu}</td>
                <td className="n">{c.budget}</td>
                <td className="n">
                  {sign(c.dAP, 4)} <span className="dim">[{sign(c.dAP_lo, 4)}, {sign(c.dAP_hi, 4)}]</span>
                </td>
                <td className="n">
                  {sign(c.dHit1 * 100, 1)} <span className="dim">[{sign(c.dHit1_lo * 100, 1)}, {sign(c.dHit1_hi * 100, 1)}]</span>
                </td>
              </tr>
            ))}
        </tbody>
      </table>
      <div className="source">research/results/E1_adaptive_k ({res.evidence.e1}) · budget 10 omitted: the only menu entry at or below it is 10</div>
    </div>
  )
}

const VARIANT: Record<string, string> = {
  win_fuse: 'global + windows',
  ttr: 'key rotations',
  s1_hub: 'hub corr. (λ = 0)',
  win_max: 'windows only',
  'global:full_60': '60 epochs',
  'global:base_1500': '1,500 works',
}

function Stage1({ res }: { res: Research }) {
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const rows = res.stage1_coverage30
  const labelW = width < 560 ? 150 : 230
  const x = linScale([Math.min(...rows.map((r) => r.lo)) * 100 - 2, Math.max(...rows.map((r) => r.hi)) * 100 + 4], [labelW, width - 60])
  const H = rows.length * 30 + 30
  const fuse = rows.find((r) => r.variant === 'win_fuse')
  const a2 = res.a2?.meets_registered_criterion ? res.a2 : null
  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">What moves Stage-1 coverage</h3>
          <p>
            Training data and training time dominate. Among changes that need no retraining, one stands out: adding the best match between three
            overlapping windows of each recording to the global score. It puts a cover into the top 30 for {fuse ? (fuse.delta * 100).toFixed(1) : '?'} more
            queries in a hundred.{' '}
            {a2 && (
              <>
                A test registered before any end-to-end number showed the gain survives reranking: MAP at K = 30 rose from{' '}
                {res.k_sweep.find((p) => p.K === 30)!.hub_MAP.toFixed(3)} to {a2.k_sweep.find((p) => p.K === 30)!.hub_MAP.toFixed(3)}, and it rose at every
                K from 5 to 500 (green line in Figure 17).
              </>
            )}
          </p>
        </div>
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Change in coverage at K = 30 for each Stage-1 variant, with 95% intervals">
          <line x1={x(0)} x2={x(0)} y1={0} y2={H - 24} stroke="var(--ink)" />
          {[-30, -20, -10, 0].filter((t) => x(t) >= labelW).map((t) => (
            <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="axis-t">
              {t > 0 ? `+${t}` : t}
            </text>
          ))}
          {rows.map((r, i) => {
            const cy = i * 30 + 14
            const c = r.lo > 0 ? 'var(--cover)' : r.hi < 0 ? 'var(--path)' : 'var(--mute)'
            return (
              <g key={r.variant}>
                <text x={labelW - 10} y={cy + 4} textAnchor="end" className="axis-t">
                  {VARIANT[r.variant] ?? r.variant}
                </text>
                <line x1={x(r.lo * 100)} x2={x(r.hi * 100)} y1={cy} y2={cy} stroke={c} strokeWidth={3} />
                <circle cx={x(r.delta * 100)} cy={cy} r={4.5} fill={c} />
                <text x={x(Math.max(r.hi, 0) * 100) + 8} y={cy + 4} className="axis-t num">
                  {r.delta >= 0 ? '+' : '−'}
                  {Math.abs(r.delta * 100).toFixed(1)}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <Caption label="Figure 19" source={`research/results/F1s_stage1_summary (${res.evidence.f1})`}>
        Change in coverage at K = 30, in points, against the reference Stage 1 (150 epochs, 4,780 training works) on the same queries. The last two
        rows are weaker encoders. Paired work-level 95% intervals. Window sizes and
        the 50/50 fusion weight were fixed before the run; the hub-correction strength was chosen on calibration works and came out as zero.
      </Caption>
    </div>
  )
}
