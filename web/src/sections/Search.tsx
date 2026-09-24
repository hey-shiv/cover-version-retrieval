import { useEffect, useRef, useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { benchmarkRuns, rankEdges, RUN_SETUP, STAGE1 } from '../data/benchmark-data'
import { devRun } from '../data/retrieval-cases'
import { useInView, useReducedMotion, useWidth } from '../lib/hooks'
import { pct } from '../lib/format'

const K = 30

export function Search() {
  return (
    <Chapter id="search" no="04 · Search" title={<>Thirty chances <em>in fifteen thousand</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            Stage 1 scores the query against every candidate with one dot product each, ranks them all, and keeps the top <strong>K = 30</strong>.
            Only those thirty are ever aligned.
          </p>
          <p>
            That is the whole bargain of the hybrid: exact cosine search over 15,000 vectors takes a fraction of a millisecond per query, and expensive
            alignment is confined to 0.2% of the pairs. It also means one hard rule. <strong>If the true cover is not among the thirty, nothing after
            Stage 1 can recover it.</strong>
          </p>
        </div>
        <aside className="aside">
          On the 20-query development protocol, K = 30 covered a quarter of a 119-item pool. On the benchmark the same K covers 0.2%. That
          difference, more than any model choice, explains why the development numbers misled (D-022).
        </aside>
      </div>
      <CandidateField />
      <RankField />
    </Chapter>
  )
}

function CandidateField() {
  const [pool, setPool] = useState<'dev' | 'bench'>('bench')
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [seenRef, seen] = useInView<HTMLDivElement>()
  const reduced = useReducedMotion()
  const [collapse, setCollapse] = useState(0)
  const n = pool === 'dev' ? 119 : 14999

  useEffect(() => {
    if (!seen) return
    if (reduced) {
      setCollapse(1)
      return
    }
    setCollapse(0)
    let raf = 0
    const t0 = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0 - 400) / 1600)
      setCollapse(Math.max(0, t))
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [seen, pool, reduced])

  const cols = pool === 'dev' ? 17 : Math.max(60, Math.floor(width / 4.4))
  const cell = width / cols
  const rows = Math.ceil(n / cols)
  const height = Math.max(120, rows * cell)

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    c.width = width * dpr
    c.height = height * dpr
    const ctx = c.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    // candidates are drawn in Stage-1 rank order: the first K are the shortlist
    const r = pool === 'dev' ? cell * 0.36 : Math.max(0.9, cell * 0.28)
    for (let i = 0; i < n; i++) {
      const x = (i % cols) * cell + cell / 2
      const y = Math.floor(i / cols) * cell + cell / 2
      const inShort = i < K
      ctx.globalAlpha = inShort ? 1 : 1 - 0.55 * collapse
      ctx.fillStyle = inShort ? '#16140f' : '#a39b89'
      ctx.beginPath()
      ctx.arc(x, y, inShort ? Math.max(r, 1.6) : r, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }, [width, height, n, cols, cell, collapse, pool])

  return (
    <div className="figure" ref={seenRef}>
      <div className="controls">
        <Seg
          label="Candidate pool"
          value={pool}
          onChange={setPool}
          options={[
            { value: 'dev', label: 'Development · 119' },
            { value: 'bench', label: 'Benchmark · 14,999' },
          ]}
        />
        <span className="readout">
          <span className="k">shortlist </span>
          {K} / {n.toLocaleString('en-US')} = {pct(K / n, pool === 'dev' ? 0 : 1)}
        </span>
      </div>
      <div ref={ref} className="field">
        <canvas
          ref={canvas}
          style={{ width: '100%', height }}
          role="img"
          aria-label={`${n} candidates in Stage-1 rank order; the first ${K} form the shortlist.`}
        />
      </div>
      <Caption label="Figure 5">
        Every candidate a query is ranked against, one dot each, in Stage-1 order (read left to right, top to bottom). The {K} dark dots are the
        shortlist handed to the aligner. Everything else keeps its Stage-1 rank.
      </Caption>
    </div>
  )
}

function RankField() {
  const runs = benchmarkRuns.filter((r) => r.id !== 'run3') // run 3 reuses run 2's Stage 1 (same encoder)
  const [id, setId] = useState<'run1' | 'run2' | 'run4'>('run4')
  const r = benchmarkRuns.find((x) => x.id === id)!
  const fr = r.first_rank[STAGE1]
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const H = 240
  const bins = fr.log_bins
  // density per log-decade, so bins of different widths compare honestly
  const density = (bs: number[]) => bs.map((v, i) => v / (Math.log10(rankEdges[i + 1]) - Math.log10(rankEdges[i])))
  const maxD = Math.max(...runs.flatMap((x) => density(x.first_rank[STAGE1].log_bins)))
  const dens = density(bins)
  const xOf = (rank: number) => 20 + (Math.log10(rank) / Math.log10(15000)) * (width - 40)
  const dev = devRun('base')

  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">The problem with Stage 1</h3>
          <p>
            For every one of the 13,000 benchmark queries, where does the <em>first</em> true cover land in the Stage-1 ranking? Everything right of
            the line is invisible to the reranker. On the development set this was five queries out of twenty. On the benchmark it was{' '}
            {pct(1 - benchmarkRuns[0].first_rank[STAGE1].in_shortlist_30, 0)} of queries in the first run, and still{' '}
            {pct(1 - benchmarkRuns[3].first_rank[STAGE1].in_shortlist_30, 0)} after every improvement.
          </p>
        </div>
      </div>
      <div className="controls">
        <Seg
          label="Benchmark run"
          value={id}
          onChange={setId}
          options={runs.map((x) => ({ value: x.id as 'run1' | 'run2' | 'run4', label: `${x.id.replace('run', 'Run ')} · ${RUN_SETUP[x.id].encoder}` }))}
        />
      </div>
      <div className="rank-stats">
        <div>
          <div className="big num c">{pct(fr.in_shortlist_30, 1)}</div>
          <div className="label">queries with ≥ 1 cover in the top 30</div>
        </div>
        <div>
          <div className="big num">{r.shortlist_recall.toFixed(3)}</div>
          <div className="label">shortlist recall (all 12 covers)</div>
        </div>
        <div>
          <div className="big num">{fr.median}</div>
          <div className="label">median rank of first cover</div>
        </div>
      </div>
      <div ref={ref} className="rankfield">
        <svg width={width} height={H} role="img" aria-label={`Histogram of first-cover Stage-1 rank for ${id}, log scale; ${pct(fr.in_shortlist_30)} fall inside the top 30.`}>
          {dens.map((v, i) => {
            const h = (v / maxD) * (H - 50)
            const inside = rankEdges[i + 1] <= K + 1
            return (
              <rect
                key={i}
                x={xOf(rankEdges[i]) + 0.5}
                y={H - 26 - h}
                width={Math.max(1, xOf(rankEdges[i + 1]) - xOf(rankEdges[i]) - 1)}
                height={h}
                fill={inside ? 'var(--cover)' : 'var(--faint)'}
                style={{ transition: 'height .5s, y .5s' }}
              />
            )
          })}
          <line x1={xOf(K + 1)} x2={xOf(K + 1)} y1={8} y2={H - 20} stroke="var(--ink)" strokeWidth={1.5} />
          <text x={xOf(K + 1) + 6} y={20} className="axis-t strong">
            K = 30{width < 640 ? '' : ' · shortlist boundary'}
          </text>
          <text x={xOf(K + 1) + 6} y={34} className="axis-t">
            {width < 640 ? 'Stage 2 never sees these' : 'beyond here Stage 2 never sees the cover'}
          </text>
          {[1, 10, 100, 1000, 10000].map((t) => (
            <text key={t} x={xOf(t)} y={H - 8} textAnchor="middle" className="axis-t">
              {t.toLocaleString('en-US')}
            </text>
          ))}
        </svg>
      </div>
      <Caption label="Figure 6" source={[r.per_query_source, r.source]}>
        Rank of the first correct cover under Stage 1 alone, 13,000 queries; bar height is queries per log-decade of rank, so bins of different widths compare fairly. Shortlist recall is the share of <em>all</em> relevant
        items (12 per query) inside the top 30, as reported in the run's result file. Run 3 is omitted: it reuses run 2's encoder. For contrast, the
        development protocol's shortlist recall was {dev.shortlist_recall.toFixed(2)} with the weakest encoder.
      </Caption>
    </div>
  )
}
