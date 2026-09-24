import { useMemo, useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { CASE_LABEL, CASE_NOTES, DISPERSION, DISPERSION_MEDIAN, devRuns } from '../data/retrieval-cases'
import type { DevRun } from '../data/results'
import { useWidth } from '../lib/hooks'
import { logScale } from '../lib/format'

const K = 30

export function Explorer() {
  const [runId, setRunId] = useState<DevRun['id']>('base')
  const run = devRuns.find((r) => r.id === runId)!
  const hybridKey = run.systems.find((s) => s.startsWith('hybrid'))!
  const [sel, setSel] = useState('P_797406')
  const q = run.queries.find((x) => x.query === sel)!
  const cs = run.cases.find((c) => c.query === sel)
  const caseByQuery = useMemo(() => new Map(run.cases.map((c) => [c.query, c])), [run])
  const figure = cs ? `${import.meta.env.BASE_URL}figures/${cs.figure}` : null

  return (
    <Chapter id="explorer" no="07 · Inspect" title={<>Twenty queries, <em>one at a time</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">The development protocol is small enough to look at every query.</p>
          <p>
            Each row is one query; the axis is where its single true partner lands among 119 candidates, under three systems. The line is the shortlist
            boundary. Switch configurations to watch the same twenty queries under a better encoder or a finer alignment. Error cases are chosen by
            rule in each configuration: the three best-ranked, the worst three with a wrong top result, and the three worst-ranked.
          </p>
        </div>
        <aside className="aside">
          With one relevant item per query, AP equals reciprocal rank, so MAP = MRR. Twenty queries give bootstrap intervals of about ±0.15 MAP. Treat
          this as an inspection tool, not evidence: the project found this protocol misleading twice (D-022).
        </aside>
      </div>

      <div className="controls">
        <Seg label="Configuration" value={runId} onChange={setRunId} options={devRuns.map((r) => ({ value: r.id, label: r.context }))} />
      </div>

      <div className="explorer">
        <div className="explorer-list">
          <Legend />
          <RankRows run={run} hybridKey={hybridKey} sel={sel} onSel={setSel} caseByQuery={caseByQuery} />
        </div>
        <div className="explorer-detail" aria-live="polite">
          <div className="label">Query</div>
          <div className="detail-pid">
            <span className="q pid">{q.query}</span> <span className="mono dim">{q.wid}</span>
          </div>
          {cs && <span className={`tag ${cs.case}`}>{CASE_LABEL[cs.case]}</span>}
          <table className="data detail-table">
            <tbody>
              <tr>
                <th>Stage 1</th>
                <td className="n">{q.ranks.global_embedding}</td>
              </tr>
              <tr>
                <th>Classical, all pairs</th>
                <td className="n">{q.ranks.classical_alignment}</td>
              </tr>
              <tr>
                <th>Hybrid K = 30, α = {run.locked.alpha}</th>
                <td className="n">
                  <b>{q.ranks[hybridKey]}</b>
                </td>
              </tr>
              <tr>
                <th>Stage 1 + test-time rotations</th>
                <td className="n">{q.ranks.global_embedding_test_time_rotations}</td>
              </tr>
            </tbody>
          </table>
          {cs ? (
            <>
              <div className="readout detail-scores">
                <div>
                  <span className="k">partner </span>
                  <span className="c">{cs.partner}</span> · cos {cs.partner_cos.toFixed(3)} · align {cs.partner_align.toFixed(3)} · shift {cs.partner_shift}{' '}
                  · {cs.in_shortlist ? 'in shortlist' : <span className="h">not in shortlist</span>}
                </div>
                <div>
                  <span className="k">top non-cover </span>
                  <span className="h">{cs.fp}</span> (rank {cs.fp_rank}) · cos {cs.fp_cos.toFixed(3)} · align {cs.fp_align.toFixed(3)} · shift {cs.fp_shift}
                </div>
                {runId === 'base' && DISPERSION[q.query] && (
                  <div>
                    <span className="k">dispersion q / partner / FP </span>
                    {DISPERSION[q.query].map((d) => d.toFixed(3)).join(' / ')} <span className="k">(median {DISPERSION_MEDIAN})</span>
                  </div>
                )}
              </div>
              {runId === 'base' && CASE_NOTES[q.query] && <p className="detail-note">{CASE_NOTES[q.query]}</p>}
              {figure && (
                <figure className="detail-fig">
                  <img src={figure} alt={`Cost matrices and DTW paths: ${q.query} against its partner and against the top non-cover`} loading="lazy" />
                  <figcaption className="source">reports/figures/{cs.figure.replace('.webp', '.png')}</figcaption>
                </figure>
              )}
            </>
          ) : (
            <p className="detail-note dim">Not one of this configuration's nine rule-selected error cases, so only ranks are recorded for it.</p>
          )}
        </div>
      </div>
      <Caption label="Figure 10" source={[run.source, run.source.replace('.json', '').replace('hybrid_dev', 'hybrid_dev_per_query') + '.csv', 'reports/results/error_cases*.csv']}>
        Rank of the true partner (1 = top, of 119; log axis). Metrics for this configuration: Stage 1 MAP {run.metrics.global_embedding.MAP?.toFixed(3)} ·
        classical {run.metrics.classical_alignment.MAP?.toFixed(3)} · hybrid {run.metrics[hybridKey].MAP?.toFixed(3)} · shortlist recall{' '}
        {run.shortlist_recall.toFixed(2)}.
      </Caption>
    </Chapter>
  )
}

function Legend() {
  return (
    <div className="legend">
      <span>
        <svg width="12" height="12">
          <circle cx="6" cy="6" r="4" fill="none" stroke="var(--query)" strokeWidth="1.5" />
        </svg>
        Stage 1
      </span>
      <span>
        <svg width="12" height="12">
          <rect x="2" y="2" width="8" height="8" fill="none" stroke="var(--mute)" strokeWidth="1.5" transform="rotate(45 6 6)" />
        </svg>
        Classical, all pairs
      </span>
      <span>
        <svg width="12" height="12">
          <circle cx="6" cy="6" r="4.5" fill="var(--path)" />
        </svg>
        Hybrid
      </span>
    </div>
  )
}

function RankRows({
  run,
  hybridKey,
  sel,
  onSel,
  caseByQuery,
}: {
  run: DevRun
  hybridKey: string
  sel: string
  onSel: (q: string) => void
  caseByQuery: Map<string, DevRun['cases'][number]>
}) {
  const [ref, width] = useWidth<HTMLDivElement>(600)
  const labelW = width < 480 ? 70 : 96
  const x = logScale([1, 119], [labelW + 18, width - 16])
  const rows = [...run.queries].sort((a, b) => a.ranks[hybridKey] - b.ranks[hybridKey])
  const rowH = 24
  return (
    <div ref={ref}>
      <svg width={width} height={rows.length * rowH + 28} role="group" aria-label="Partner rank per query">
        <line x1={x(K + 0.5)} x2={x(K + 0.5)} y1={0} y2={rows.length * rowH} stroke="var(--ink)" strokeDasharray="3 3" />
        {[1, 3, 10, 30, 100].map((t) => (
          <text key={t} x={x(t)} y={rows.length * rowH + 18} textAnchor="middle" className="axis-t">
            {t}
          </text>
        ))}
        {rows.map((r, i) => {
          const y = i * rowH + rowH / 2
          const g = r.ranks.global_embedding
          const c = r.ranks.classical_alignment
          const h = r.ranks[hybridKey]
          const cs = caseByQuery.get(r.query)
          const active = r.query === sel
          return (
            <g
              key={r.query}
              className="rank-row"
              onClick={() => onSel(r.query)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onSel(r.query))}
              tabIndex={0}
              role="button"
              aria-pressed={active}
              aria-label={`${r.query}: Stage 1 rank ${g}, classical ${c}, hybrid ${h}`}
            >
              <rect x={0} y={i * rowH} width={width} height={rowH} fill={active ? 'var(--paper-2)' : 'transparent'} />
              <text x={4} y={y + 4} className="axis-t" fill={active ? 'var(--ink)' : undefined}>
                {r.query}
              </text>
              {cs && <circle cx={labelW} cy={y} r={3} fill={cs.case === 'success' ? 'var(--cover)' : cs.case === 'false_positive' ? 'var(--hub)' : 'var(--path)'} />}
              <line x1={x(Math.min(g, c, h))} x2={x(Math.max(g, c, h))} y1={y} y2={y} stroke="var(--rule)" />
              <circle cx={x(g)} cy={y} r={4.5} fill="var(--paper)" stroke="var(--query)" strokeWidth={1.5} />
              <rect x={x(c) - 3.5} y={y - 3.5} width={7} height={7} fill="var(--paper)" stroke="var(--mute)" strokeWidth={1.5} transform={`rotate(45 ${x(c)} ${y})`} />
              <circle cx={x(h)} cy={y} r={4.5} fill="var(--path)" />
            </g>
          )
        })}
      </svg>
    </div>
  )
}
