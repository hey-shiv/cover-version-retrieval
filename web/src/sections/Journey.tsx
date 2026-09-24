import { Chapter, Caption } from '../components/Chapter'
import { LOG } from '../data/experiment-history'
import { benchmarkRuns, bestHybrid, bestOverall, RUN_LABEL, STAGE1 } from '../data/benchmark-data'
import { useWidth } from '../lib/hooks'
import { linScale } from '../lib/format'

const KIND_MARK: Record<string, string> = {
  build: '□',
  evaluate: '○',
  negative: '✕',
  diagnose: '◇',
  experiment: '△',
  lesson: '—',
}

export function Journey() {
  return (
    <Chapter id="journey" no="10 · Research log" title={<>We discovered <em>why</em>, not just how much</>} dark>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            The first benchmark evaluation was a negative result. Everything after it was an attempt to explain that result, and each explanation was
            tested on data the benchmark never touched.
          </p>
        </div>
        <aside className="aside">
          Dates and commits from the repository history; decisions D-0xx from notes/decisions.md. Every entry names the files that hold its evidence.
        </aside>
      </div>
      <Trend />
      <ol className="log">
        {LOG.map((e, i) => (
          <li key={i} className={`log-entry k-${e.kind}`}>
            <div className="log-meta">
              <span className="log-mark" aria-hidden="true">
                {KIND_MARK[e.kind]}
              </span>
              <span className="mono">{e.date}</span>
              <span className="mono dim">{e.commit}</span>
            </div>
            <div className="log-body">
              <h3>{e.title}</h3>
              <p>{e.body}</p>
              <div className="log-refs mono">
                {e.decisions?.map((d) => (
                  <span key={d} className="log-d">
                    {d}
                  </span>
                ))}
                {e.evidence.map((s) => (
                  <span key={s} className="source">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </Chapter>
  )
}

function Trend() {
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const series = [
    { label: 'Stage-1 MAP', get: (r: (typeof benchmarkRuns)[number]) => r.metrics[STAGE1].MAP!, fmt: (v: number) => v.toFixed(3) },
    { label: 'Best hybrid MAP', get: bestHybrid, fmt: (v: number) => v.toFixed(3) },
    { label: 'Shortlist recall', get: (r: (typeof benchmarkRuns)[number]) => r.shortlist_recall, fmt: (v: number) => v.toFixed(3) },
    {
      label: 'Best system ÷ best hybrid',
      get: (r: (typeof benchmarkRuns)[number]) => bestOverall(r).map / bestHybrid(r),
      fmt: (v: number) => `${v.toFixed(2)}×`,
    },
  ]
  const cols = width > 860 ? 4 : 2
  const w = (width - (cols - 1) * 20) / cols
  const H = 120
  return (
    <div className="figure">
      <div ref={ref} className="trend" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {series.map((s) => {
          const vals = benchmarkRuns.map(s.get)
          const lo = Math.min(0, ...vals)
          const hi = Math.max(...vals) * 1.15
          const x = linScale([0, 3], [30, w - 30])
          const y = linScale([lo, hi], [H - 22, 20])
          return (
            <div key={s.label}>
              <div className="label">{s.label}</div>
              <svg width={w} height={H} role="img" aria-label={`${s.label} across runs 1 to 4: ${vals.map(s.fmt).join(', ')}`}>
                <path d={vals.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')} fill="none" stroke="var(--panel-ink)" strokeWidth={1.5} />
                {vals.map((v, i) => (
                  <g key={i}>
                    <circle cx={x(i)} cy={y(v)} r={3.5} fill={i === 3 ? 'var(--path)' : 'var(--panel-ink)'} />
                    <text x={x(i)} y={y(v) - 9} textAnchor="middle" className="axis-t light">
                      {s.fmt(v)}
                    </text>
                    <text x={x(i)} y={H - 4} textAnchor="middle" className="axis-t dim">
                      {RUN_LABEL[benchmarkRuns[i].id].replace('Run ', 'R')}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          )
        })}
      </div>
      <Caption label="Figure 16" source={benchmarkRuns.map((r) => r.source)}>
        Across the four benchmark evaluations. "Best system" is exhaustive classical alignment: uncorrected in run 1 (0.084), hubness-corrected from
        run 2 on (0.136, evaluated once in run 2). Ratios are computed from unrounded MAPs, so run 1 reads 4.5× where the README, rounding first,
        says 4.6×.
      </Caption>
    </div>
  )
}
