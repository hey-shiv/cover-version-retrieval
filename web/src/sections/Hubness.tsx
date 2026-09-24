import { useMemo, useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { hub, hubGraph } from '../data/hubness'
import { run, rankGrid } from '../data/benchmark-data'
import { useWidth } from '../lib/hooks'
import { linScale, logScale } from '../lib/format'

export function Hubness() {
  return (
    <Chapter id="hubness" no="08 · Failure" title={<>The tracks <em>that match everything</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            Some recordings barely move harmonically. Every frame sits close to the track's own average chroma, so any path through their matrix is
            cheap, and they turn up near the top of rankings for queries they have nothing to do with.
          </p>
          <p>
            Across the 120 development candidates, <strong>tonal dispersion</strong> (how far a track's frames stray from its mean chroma) predicts how
            often it is a top-10 false positive: Spearman ρ = {hub.summary.classical.spearman_dispersion_vs_top10_fp.toFixed(2)} for classical
            alignment, {hub.summary.hybrid.spearman_dispersion_vs_top10_fp.toFixed(2)} for the hybrid. The worst hub,{' '}
            <span className="pid h">P_400551</span>, has dispersion {hub.summary.classical.top_hubs[0].dispersion.toFixed(3)} against a pool median of{' '}
            {hub.summary.dispersion_median.toFixed(3)}, and is a top-10 false positive for {hub.summary.classical.top_hubs[0].top10_fp_count} of 20
            queries. Chance would give {hub.summary.expected_top10_fp_count_if_uniform.toFixed(1)}.
          </p>
        </div>
        <aside className="aside">
          This is a known phenomenon in music similarity (Aucouturier &amp; Pachet 2008; Flexer et al. 2012), and hubness correction for cover
          identification is prior art (Seo 2022; Li &amp; Chen 2018). What this project adds is a measurement at Da-TACOS scale, with held-out calibration.
        </aside>
      </div>
      <HubGraph />
      <Correction />
    </Chapter>
  )
}

function HubGraph() {
  const [system, setSystem] = useState<'classical' | 'hybrid'>('classical')
  const g = useMemo(() => hubGraph(system), [system])
  const [focus, setFocus] = useState<string | null>(null)
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const narrow = width < 640
  const shown = g.candidates.filter((c) => c.queries.length >= 2).slice(0, narrow ? 12 : 18)
  const rest = g.candidates.length - shown.length
  const rowH = 24
  const H = Math.max(g.queries.length * 16, shown.length * rowH) + 40
  const qx = narrow ? 64 : 120
  const cx = width * (narrow ? 0.55 : 0.5)
  const qy = (i: number) => 24 + i * ((H - 40) / (g.queries.length - 1))
  const cy = (i: number) => 24 + i * rowH
  const qIndex = new Map(g.queries.map((q, i) => [q, i]))
  const maxCount = Math.max(...g.candidates.map((c) => c.queries.length))
  const barX = linScale([0, maxCount], [cx + 110, width - (narrow ? 40 : 150)])
  const top = focus ?? shown[0].pid

  return (
    <div className="figure">
      <div className="controls">
        <Seg
          label="System"
          value={system}
          onChange={(v) => {
            setSystem(v)
            setFocus(null)
          }}
          options={[
            { value: 'classical', label: 'Classical alignment' },
            { value: 'hybrid', label: 'Hybrid (K = 30)' },
          ]}
        />
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Queries linked to the candidates that appear in their top ten as false positives, ${system}.`}>
          <text x={qx} y={10} textAnchor="end" className="axis-t">
            20 QUERIES
          </text>
          <text x={cx + 8} y={10} className="axis-t">
            {narrow ? 'CANDIDATE' : 'FALSE-POSITIVE CANDIDATE'}
          </text>
          <text x={width - 4} y={10} className="axis-t" textAnchor="end">
            TIMES IN A TOP 10
          </text>
          {shown.map((c, ci) =>
            c.queries.map((q) => {
              const on = c.pid === top
              return (
                <path
                  key={c.pid + q}
                  d={`M ${qx + 6} ${qy(qIndex.get(q)!)} C ${(qx + cx) / 2} ${qy(qIndex.get(q)!)}, ${(qx + cx) / 2} ${cy(ci)}, ${cx} ${cy(ci)}`}
                  fill="none"
                  stroke={on ? 'var(--hub)' : 'var(--rule)'}
                  strokeWidth={on ? 1.6 : 0.8}
                  opacity={on ? 1 : 0.7}
                />
              )
            }),
          )}
          {g.queries.map((q, i) => {
            const hit = shown.find((c) => c.pid === top)?.queries.includes(q)
            return (
              <g key={q}>
                <circle cx={qx + 4} cy={qy(i)} r={3.2} fill={hit ? 'var(--query)' : 'var(--paper)'} stroke="var(--query)" />
                {!narrow && (
                  <text x={qx - 6} y={qy(i) + 3.5} textAnchor="end" className="axis-t" fontSize={9.5}>
                    {q}
                  </text>
                )}
              </g>
            )
          })}
          {/* expected count if false positives were spread uniformly */}
          <line x1={barX(hub.summary.expected_top10_fp_count_if_uniform)} x2={barX(hub.summary.expected_top10_fp_count_if_uniform)} y1={16} y2={H - 14} stroke="var(--ink)" strokeDasharray="2 3" />
          <text x={barX(hub.summary.expected_top10_fp_count_if_uniform) + 4} y={H - 2} className="axis-t">
            chance ≈ {hub.summary.expected_top10_fp_count_if_uniform.toFixed(1)}
          </text>
          {shown.map((c, ci) => {
            const on = c.pid === top
            const d = g.dispersion.get(c.pid)
            return (
              <g
                key={c.pid}
                onMouseEnter={() => setFocus(c.pid)}
                onFocus={() => setFocus(c.pid)}
                onClick={() => setFocus(c.pid)}
                tabIndex={0}
                role="button"
                aria-label={`${c.pid}: top-10 false positive for ${c.queries.length} queries${d ? `, dispersion ${d.toFixed(3)}` : ''}`}
                style={{ cursor: 'pointer' }}
              >
                <rect x={cx} y={cy(ci) - rowH / 2} width={width - cx} height={rowH} fill={on ? 'var(--paper-2)' : 'transparent'} />
                <circle cx={cx} cy={cy(ci)} r={3 + c.queries.length * 0.6} fill={on ? 'var(--hub)' : 'var(--faint)'} />
                <text x={cx + 14} y={cy(ci) + 4} className="axis-t" fill={on ? 'var(--ink)' : undefined}>
                  {c.pid}
                </text>
                <rect x={barX(0)} y={cy(ci) - 4} width={barX(c.queries.length) - barX(0)} height={8} fill={on ? 'var(--hub)' : 'var(--paper-3)'} />
                <text x={barX(c.queries.length) + 5} y={cy(ci) + 4} className="axis-t num">
                  {c.queries.length}
                  {d !== undefined && !narrow ? `  · disp ${d.toFixed(3)}` : ''}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <Caption label="Figure 11" source={[...hub.edges_source, hub.source]}>
        Every link is one development query whose top 10 contains that candidate although it is a different work. The {shown.length} most frequent
        candidates are shown; {rest} others appear less often. Dispersion is printed where the analysis file records it (its five worst hubs). Hover a
        candidate to trace its queries.
      </Caption>
    </div>
  )
}

function Correction() {
  const [res, setRes] = useState<96 | 384>(96)
  const corr = hub.corrections.find((c) => c.n_frames === res)!
  const grid = corr.grid.filter((g) => g.method === 'mean')
  const [lam, setLam] = useState(corr.locked.lambda)
  const [ref, width] = useWidth<HTMLDivElement>(600)
  const H = 210
  const x = linScale([0, 1.5], [40, width - 16])
  const y = linScale([0, 0.45], [H - 24, 8])
  const atLam = grid.reduce((best, g) => (Math.abs(g.lambda - lam) < Math.abs(best.lambda - lam) ? g : best), grid[0])

  const r2 = run('run2')
  const classical = r2.first_rank.classical_alignment
  const classicalHub = r2.first_rank.classical_alignment_hubcorr
  const [ref2, w2] = useWidth<HTMLDivElement>(600)
  const x2 = logScale([1, 15000], [40, w2 - 16])
  const y2 = linScale([0, 1], [H - 24, 8])
  const curve = (cdf: number[]) => cdf.map((v, i) => `${i ? 'L' : 'M'}${x2(rankGrid[i]).toFixed(1)},${y2(v).toFixed(1)}`).join(' ')

  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">One scalar, calibrated on ten works</h3>
          <p>
            The correction subtracts each candidate's general score level, measured against a fixed set of 200 training-split probe tracks:
          </p>
          <p className="formula mono">
            corrected[q, c] = score[q, c] − <span className="p">λ</span> · reference[c]
          </p>
          <p>
            λ was grid-searched on the ten calibration works only. The probe reference, computed without any labels, correlates with tonal dispersion at
            ρ = {corr.reference_vs_dispersion_spearman.toFixed(2)}: it rediscovers "static tracks are hubs" on its own.
          </p>
        </div>
      </div>
      <div className="two-up">
        <div>
          <div className="controls">
            <Seg
              label="Resolution"
              value={String(res) as '96' | '384'}
              onChange={(v) => {
                const n = Number(v) as 96 | 384
                setRes(n)
                setLam(hub.corrections.find((c) => c.n_frames === n)!.locked.lambda)
              }}
              options={[
                { value: '96', label: '96 frames' },
                { value: '384', label: '384 frames' },
              ]}
            />
          </div>
          <div ref={ref}>
            <svg width={width} height={H} role="img" aria-label={`Calibration MAP against lambda at ${res} frames`}>
              {[0, 0.1, 0.2, 0.3, 0.4].map((v) => (
                <g key={v}>
                  <line x1={40} x2={width - 16} y1={y(v)} y2={y(v)} stroke="var(--rule)" strokeDasharray={v ? '2 3' : undefined} />
                  <text x={34} y={y(v) + 4} textAnchor="end" className="axis-t">
                    {v.toFixed(1)}
                  </text>
                </g>
              ))}
              {[0, 0.5, 1, 1.5].map((v) => (
                <text key={v} x={x(v)} y={H - 6} textAnchor="middle" className="axis-t">
                  λ {v}
                </text>
              ))}
              <path d={grid.map((g, i) => `${i ? 'L' : 'M'}${x(g.lambda)},${y(g.MAP)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={1.6} />
              {grid.map((g) => (
                <circle key={g.lambda} cx={x(g.lambda)} cy={y(g.MAP)} r={2.5} fill="var(--ink)" />
              ))}
              <line x1={x(corr.locked.lambda)} x2={x(corr.locked.lambda)} y1={8} y2={H - 24} stroke="var(--path)" />
              <text x={x(corr.locked.lambda) + 4} y={18} className="axis-t" fill="var(--path)">
                locked λ = {corr.locked.lambda}
              </text>
              <circle cx={x(atLam.lambda)} cy={y(atLam.MAP)} r={6} fill="none" stroke="var(--hub)" strokeWidth={2} />
            </svg>
          </div>
          <label className="slider">
            <span className="readout">λ = {atLam.lambda.toFixed(1)}</span>
            <input type="range" min={0} max={1.5} step={0.1} value={lam} onChange={(e) => setLam(Number(e.target.value))} aria-label="Lambda" />
            <span className="readout">
              <span className="k">calibration MAP </span>
              {atLam.MAP.toFixed(3)}
            </span>
          </label>
        </div>
        <div>
          <div className="label ecdf-title">Benchmark run 2 · classical alignment, 96 frames</div>
          <div ref={ref2}>
            <svg width={w2} height={H} role="img" aria-label="Share of benchmark queries whose first cover is within rank r, with and without hubness correction">
              {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                <g key={v}>
                  <line x1={40} x2={w2 - 16} y1={y2(v)} y2={y2(v)} stroke="var(--rule)" strokeDasharray={v ? '2 3' : undefined} />
                  <text x={34} y={y2(v) + 4} textAnchor="end" className="axis-t">
                    {v * 100}%
                  </text>
                </g>
              ))}
              {[1, 10, 100, 1000, 10000].map((t) => (
                <text key={t} x={x2(t)} y={H - 6} textAnchor="middle" className="axis-t">
                  {t.toLocaleString('en-US')}
                </text>
              ))}
              <path d={curve(classical.cdf)} fill="none" stroke="var(--faint)" strokeWidth={1.8} />
              <path d={curve(classicalHub.cdf)} fill="none" stroke="var(--hub)" strokeWidth={2.2} />
              <line x1={x2(classical.median)} x2={x2(classical.median)} y1={y2(0.5)} y2={H - 24} stroke="var(--faint)" strokeDasharray="2 2" />
              <line x1={x2(classicalHub.median)} x2={x2(classicalHub.median)} y1={y2(0.5)} y2={H - 24} stroke="var(--hub)" strokeDasharray="2 2" />
              <text x={x2(classicalHub.median) - 4} y={y2(0.5) - 6} textAnchor="end" className="axis-t" fill="var(--hub)">
                median {classicalHub.median}
              </text>
              <text x={x2(classical.median) + 4} y={y2(0.5) + 14} className="axis-t">
                median {classical.median}
              </text>
            </svg>
          </div>
          <div className="readout">
            MAP <span className="num">{r2.metrics.classical_alignment.MAP!.toFixed(3)}</span> →{' '}
            <b className="h num">{r2.metrics.classical_alignment_hubcorr.MAP!.toFixed(3)}</b> · Hit@10{' '}
            {r2.metrics.classical_alignment['Hit@10']!.toFixed(3)} → {r2.metrics.classical_alignment_hubcorr['Hit@10']!.toFixed(3)}{' '}
            <span className="k">(13,000 queries)</span>
          </div>
        </div>
      </div>
      <Caption label="Figure 12" source={[corr.source, r2.per_query_source, r2.source]}>
        Left: calibration MAP over the λ grid (mean reference), the only data used to choose λ. Right: share of the 13,000 benchmark queries whose first
        cover lies within rank <i>r</i>, before (grey) and after (ochre) the correction. The same λ, never tuned on the benchmark, cuts the median
        first-cover rank from {classical.median} to {classicalHub.median}. At 384 frames the calibration gain is smaller, and on the 20-query
        development set the correction even lowered MAP (0.384 → 0.355): higher resolution and correction partly fix the same thing.
      </Caption>
    </div>
  )
}
