import { useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import {
  benchmarkRuns,
  METRICS,
  METRIC_LABEL,
  PUBLISHED,
  RUN_LABEL,
  RUN_SETUP,
  metric,
  rankGrid,
  systemKind,
  systemLabel,
  type MetricKey,
  type SystemKind,
} from '../data/benchmark-data'
import { devRun } from '../data/retrieval-cases'
import { useWidth } from '../lib/hooks'
import { linScale, logScale } from '../lib/format'

const KIND_COLOR: Record<SystemKind, string> = {
  stage1: 'var(--query)',
  'stage1-rot': 'var(--query)',
  hybrid: 'var(--path)',
  rerank: 'var(--faint)',
  'hybrid-hub': 'var(--path)',
  classical: 'var(--mute)',
  'classical-hub': 'var(--hub)',
}
const SHORT: Record<SystemKind, string> = {
  stage1: 'Stage 1',
  'stage1-rot': 'Stage 1 + rotations',
  hybrid: 'Hybrid',
  rerank: 'Rerank only',
  'hybrid-hub': 'Hybrid + hub corr.',
  classical: 'Classical',
  'classical-hub': 'Classical + hub corr.',
}
const hollow = (k: SystemKind) => k === 'hybrid' || k === 'classical'

export function Benchmark() {
  return (
    <Chapter id="results" no="09 · Results" title={<>Thirteen thousand queries, <em>four evaluations</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            The final test: every clique track of the Da-TACOS benchmark as a query (13,000), against all 15,000 tracks, with 12 true covers per
            query. Random ranking scores about 0.0008 MAP.
          </p>
          <p>
            It was evaluated four times, once per configuration, each time with every setting locked beforehand: the encoder from validation works;
            α, λ and resolution from calibration works. No benchmark number selected anything, and no run was repeated after its result was seen.
            The runs are shown side by side, never merged. Two of them measure systems the others do not.
          </p>
        </div>
        <aside className="aside">
          Paired work-level bootstrap over the 1,000 query cliques: every interval reported against the matching Stage 1 excludes zero. At this scale
          even small differences are real. The question is what they mean.
        </aside>
      </div>
      <RunPanels />
      <FirstRankCurves />
      <Reversal />
      <Published />
    </Chapter>
  )
}

function RunPanels() {
  const [m, setM] = useState<MetricKey>('MAP')
  const all = benchmarkRuns.flatMap((r) => Object.values(r.metrics).map((x) => metric(x, m)))
  const isRank = m === 'median_first_rank'
  const [ref, width] = useWidth<HTMLDivElement>(1000)
  const cols = width > 980 ? 4 : width > 560 ? 2 : 1
  const pw = (width - (cols - 1) * 20) / cols
  const scale = isRank ? logScale([1, 300], [pw - 12, 12]) : linScale([0, Math.max(...all) * 1.08], [12, pw - 12])

  return (
    <div className="figure">
      <div className="controls">
        <Seg label="Metric" value={m} onChange={setM} options={METRICS.map((k) => ({ value: k, label: METRIC_LABEL[k] }))} />
        <span className="label">{isRank ? 'lower is better · log axis' : 'higher is better'} · same axis in every panel</span>
      </div>
      <div ref={ref} className="runs" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {benchmarkRuns.map((r) => (
          <article key={r.id} className="run-panel">
            <header>
              <div className="run-name">{RUN_LABEL[r.id]}</div>
              <div className="mono dim" title="git commit recorded in the result file">code @ {r.git_commit.replace('-dirty', '')}</div>
            </header>
            <dl className="run-setup">
              <dt>encoder</dt>
              <dd>{RUN_SETUP[r.id].encoder}</dd>
              <dt>rerank frames</dt>
              <dd>{RUN_SETUP[r.id].frames}</dd>
              <dt>hubness</dt>
              <dd>{RUN_SETUP[r.id].hub}</dd>
              <dt>α · K</dt>
              <dd>
                {r.locked.alpha} · {r.locked.shortlist_k}
              </dd>
            </dl>
            <ul className="run-systems">
              {Object.entries(r.metrics).map(([key, mm]) => {
                const k = systemKind(key)
                const v = metric(mm, m)
                return (
                  <li key={key}>
                    <div className="sys-label">
                      {systemLabel(key).split(' · α')[0]}
                      {r.id === 'run2' && key === 'classical_alignment' && <span className="dim"> (same scores as run 1)</span>}
                    </div>
                    <svg width={pw} height={16} aria-label={`${systemLabel(key)}: ${METRIC_LABEL[m]} ${v}`}>
                      <line x1={12} x2={pw - 12} y1={8} y2={8} stroke="var(--rule)" />
                      <line x1={isRank ? pw - 12 : 12} x2={scale(v)} y1={8} y2={8} stroke={KIND_COLOR[k]} strokeWidth={1.2} opacity={0.5} />
                      <circle cx={scale(v)} cy={8} r={4.5} fill={hollow(k) ? 'var(--paper)' : KIND_COLOR[k]} stroke={KIND_COLOR[k]} strokeWidth={1.6} />
                    </svg>
                    <div className="sys-val num">{isRank ? v : v.toFixed(3)}</div>
                  </li>
                )
              })}
            </ul>
            <footer className="run-foot">
              <span className="label">shortlist recall</span> <span className="num">{r.shortlist_recall.toFixed(3)}</span>
              <div className="source">{r.source}</div>
            </footer>
          </article>
        ))}
      </div>
      <Caption label="Figure 13">
        Every system each run evaluated, straight from its result file. Hollow marks are the uncorrected variants. "Rerank only" is the hybrid with α = 0
        (alignment score alone inside the shortlist). Run 2 reused run 1's cached 96-frame alignment matrix, so its uncorrected classical row repeats
        run 1's scores. Classical alignment at 384 frames over all pairs (about 42 h of CPU) was never run.
      </Caption>
    </div>
  )
}

function FirstRankCurves() {
  const [id, setId] = useState(benchmarkRuns[1].id)
  const r = benchmarkRuns.find((x) => x.id === id)!
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const H = 300
  const x = logScale([1, 15000], [44, width - (width < 640 ? 16 : 250)])
  const y = linScale([0, 1], [H - 26, 10])
  const keys = Object.keys(r.first_rank).filter((k) => systemKind(k) !== 'rerank')
  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">Where the first cover lands</h3>
          <p>
            Averages hide the shape. For each system, the share of queries whose first true cover is within rank <i>r</i>. Past rank 30 every hybrid
            curve merges into its Stage-1 curve, by construction: the hybrid only reorders the head.
          </p>
        </div>
      </div>
      <div className="controls">
        <Seg label="Run" value={id} onChange={setId} options={benchmarkRuns.map((b) => ({ value: b.id, label: RUN_LABEL[b.id] }))} />
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`First-cover rank distribution per system, ${RUN_LABEL[r.id]}`}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={44} x2={x(15000)} y1={y(v)} y2={y(v)} stroke="var(--rule)" strokeDasharray={v ? '2 3' : undefined} />
              <text x={38} y={y(v) + 4} textAnchor="end" className="axis-t">
                {v * 100}%
              </text>
            </g>
          ))}
          {[1, 10, 30, 100, 1000, 10000].map((t) => (
            <text key={t} x={x(t)} y={H - 8} textAnchor="middle" className="axis-t" fontWeight={t === 30 ? 600 : undefined}>
              {t.toLocaleString('en-US')}
            </text>
          ))}
          <line x1={x(30.5)} x2={x(30.5)} y1={10} y2={H - 26} stroke="var(--ink)" />
          {keys.map((k, i) => {
            const kind = systemKind(k)
            const cdf = r.first_rank[k].cdf
            const d = cdf.map((v, j) => `${j ? 'L' : 'M'}${x(rankGrid[j]).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
            const at = cdf[rankGrid.findIndex((g) => g >= 10)]
            return (
              <g key={k}>
                <path d={d} fill="none" stroke={KIND_COLOR[kind]} strokeWidth={hollow(kind) ? 1.4 : 2.2} strokeDasharray={hollow(kind) ? '5 3' : undefined} />
                {width >= 640 && (
                  <text x={x(15000) + 8} y={24 + i * 16} className="axis-t" fill={KIND_COLOR[kind]}>
                    {SHORT[kind]} · {(at * 100).toFixed(0)}% in top 10
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
      <Caption label="Figure 14" source={r.per_query_source}>
        {r.context}. Cumulative share of the 13,000 queries by rank of the first correct cover (log axis). Dashed = uncorrected variants. Vertical
        line: the K = 30 shortlist.
      </Caption>
    </div>
  )
}

function Reversal() {
  const dev = devRun('base')
  const r1 = benchmarkRuns[0]
  const rows = [
    { key: 'hybrid', label: 'Hybrid', dev: dev.metrics['hybrid_K30_alpha0.05'].MAP!, bench: r1.metrics['hybrid_K30_alpha0.05'].MAP!, color: 'var(--path)' },
    { key: 'classical', label: 'Classical', dev: dev.metrics.classical_alignment.MAP!, bench: r1.metrics.classical_alignment.MAP!, color: 'var(--mute)' },
    { key: 'stage1', label: 'Stage 1', dev: dev.metrics.global_embedding.MAP!, bench: r1.metrics.global_embedding.MAP!, color: 'var(--query)' },
  ]
  const byDev = [...rows].sort((a, b) => b.dev - a.dev)
  const byBench = [...rows].sort((a, b) => b.bench - a.bench)
  const [ref, width] = useWidth<HTMLDivElement>(600)
  const H = 170
  const narrow = width < 520
  const xa = narrow ? width * 0.36 : Math.min(170, width * 0.3)
  const xb = narrow ? width * 0.64 : width - Math.min(170, width * 0.3)
  const yy = (i: number) => 40 + i * 48
  return (
    <div className="figure two-up reversal">
      <div className="prose">
        <h3 className="sub">The reversal</h3>
        <p>
          The same three systems, same code, same locked settings. On 20 development queries the hybrid ranked first. On the benchmark it ranked
          second, a factor of {(r1.metrics.classical_alignment.MAP! / r1.metrics['hybrid_K30_alpha0.05'].MAP!).toFixed(1)} behind exhaustive alignment.
          It was reported as the headline result (D-017), and it is what every later experiment was aimed at.
        </p>
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Ordering of three systems on the development protocol versus benchmark run 1">
          <text x={xa} y={16} textAnchor="middle" className="axis-t">
            {narrow ? 'DEV' : 'DEVELOPMENT · 20 × 120'}
          </text>
          <text x={xb} y={16} textAnchor="middle" className="axis-t">
            {narrow ? 'BENCHMARK' : 'BENCHMARK RUN 1 · 13,000 × 15,000'}
          </text>
          {rows.map((r) => {
            const i = byDev.indexOf(r)
            const j = byBench.indexOf(r)
            return (
              <g key={r.key}>
                <line x1={xa + 8} x2={xb - 8} y1={yy(i)} y2={yy(j)} stroke={r.color} strokeWidth={2} />
                <circle cx={xa} cy={yy(i)} r={5} fill={r.color} />
                <circle cx={xb} cy={yy(j)} r={5} fill={r.color} />
                <text x={xa - 10} y={yy(i) + 4} textAnchor="end" className="axis-t">
                  {r.label} {r.dev.toFixed(3)}
                </text>
                <text x={xb + 10} y={yy(j) + 4} className="axis-t">
                  {r.bench.toFixed(3)} {r.label}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="source">reports/results/hybrid_dev.json · reports/results/benchmark.json</div>
      </div>
    </div>
  )
}

function Published() {
  const r2 = benchmarkRuns[1]
  const r4 = benchmarkRuns[3]
  const ours = [
    { system: 'This project · classical + hubness (run 2)', MAP: r2.metrics.classical_alignment_hubcorr.MAP! },
    { system: 'This project · hybrid + hubness (run 4)', MAP: r4.metrics['hybrid_hubcorr_K30_alpha0.10'].MAP! },
  ]
  const all = [...PUBLISHED.map((p) => ({ ...p, ours: false })), ...ours.map((o) => ({ ...o, input: 'HPCP', source: '', ours: true }))].sort((a, b) => a.MAP - b.MAP)
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const x = linScale([0, 0.75], [16, width - 16])
  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">In context</h3>
          <p>
            Same benchmark, same protocol, published systems. The strongest result here is well below Qmax on the same HPCP input, and far below learned
            systems trained on much more data. The project's findings are about <em>why</em> its systems behave as they do, not about where they rank.
          </p>
        </div>
      </div>
      <div ref={ref}>
        <svg width={width} height={all.length * 26 + 30} role="img" aria-label="MAP on the Da-TACOS benchmark: published systems and this project">
          {[0, 0.25, 0.5, 0.75].map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={0} y2={all.length * 26} stroke="var(--rule)" />
              <text x={x(v)} y={all.length * 26 + 18} textAnchor="middle" className="axis-t">
                {v.toFixed(2)}
              </text>
            </g>
          ))}
          {all.map((p, i) => (
            <g key={p.system}>
              <line x1={x(0)} x2={x(p.MAP)} y1={i * 26 + 13} y2={i * 26 + 13} stroke={p.ours ? 'var(--path)' : 'var(--faint)'} strokeWidth={p.ours ? 2 : 1} />
              <circle cx={x(p.MAP)} cy={i * 26 + 13} r={4} fill={p.ours ? 'var(--path)' : 'var(--ink)'} />
              <text x={x(p.MAP) + (x(p.MAP) > width * (width < 640 ? 0.3 : 0.6) ? -10 : 10)} y={i * 26 + 17} textAnchor={x(p.MAP) > width * (width < 640 ? 0.3 : 0.6) ? 'end' : 'start'} className="axis-t" fill={p.ours ? 'var(--path)' : 'var(--ink-2)'}>
                {width < 640 ? p.system.replace('This project · ', 'Ours · ').replace(' (Serrà et al. 2009)', '') : `${p.system} · ${p.input}`} · {p.MAP.toFixed(3)}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <Caption label="Figure 15" source="notes/literature/novelty_assessment.md (published values: Yesiler et al. 2019, Table 2; Yesiler, Serrà & Gómez 2020; Du et al. 2021)">
        MAP on the Da-TACOS benchmark subset. Published learned systems were trained on far larger data (Re-MOVE's training set alone is about 17×
        what this encoder saw). Not a leaderboard: an honest placement.
      </Caption>
    </div>
  )
}
