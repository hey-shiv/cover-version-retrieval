import { useEffect, useMemo, useRef, useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { alignments } from '../data/figures'
import { caseStudy, CASE_PARTNER, CASE_QUERY } from '../data/retrieval-cases'
import { subsequenceDTW } from '../lib/dtw'
import { useReducedMotion, useWidth } from '../lib/hooks'
import { lut } from '../components/ramp'
import { linScale } from '../lib/format'

export function Rescue() {
  return (
    <Chapter id="rescue" no="06 · Alignment" title={<>When similarity isn't enough, <em>alignment begins</em></>} dark>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            Query <span className="pid">{CASE_QUERY}</span> has exactly one other recording in the pool: <span className="pid">{CASE_PARTNER}</span>. The
            embedding puts it fifteenth.
          </p>
          <p>
            Fifteenth is inside the shortlist, so Stage 2 gets a chance. For each of the thirty candidates it picks the best key rotation, builds a
            96 × 96 matrix of cosine distances between every query frame and every candidate frame, and finds the cheapest monotone path through it.
            A cover that follows the same harmonic sequence leaves a dark diagonal groove. A static track leaves stripes, not a groove.
          </p>
        </div>
        <aside className="aside">
          Base configuration: 1,500-work encoder, 96 frames, K = 30, α = 0.05 (locked on calibration works). The error-analysis report calls this "the
          clearest case of reranking fixing Stage 1". It was selected by rule (top-ranked successes), not by eye.
        </aside>
      </div>
      <RankFlip />
      <AlignmentLab />
    </Chapter>
  )
}

/* ------------------------------------------------------------------ rank 15 → 1 */

const STEPS = [
  { id: 0, name: 'Stage 1', text: 'Global cosine similarity ranks the true partner 15th of 119 (cos 0.917). Six of the ten candidates drawn here score higher.' },
  { id: 1, name: 'Shortlist', text: 'Rank 15 is inside K = 30, so the partner survives. The five development queries whose partners ranked below 30 end here.' },
  { id: 2, name: 'Align', text: 'Each shortlisted pair is key-rotated and aligned. The partner gets the highest alignment score in the shortlist: 0.979.' },
  { id: 3, name: 'Rerank', text: 'The blended score (α = 0.05, nearly all weight on alignment) reorders the shortlist. The partner moves to rank 1.' },
]

function RankFlip() {
  const cs = useMemo(() => caseStudy(), [])
  const [step, setStep] = useState(0)
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const narrow = width < 620
  const H = 380
  const pad = { t: 34, b: 28 }
  const colA = narrow ? 70 : 120
  const colB = width * 0.52
  const colC = width - (narrow ? 70 : 120)
  const yCos = linScale([0.87, 0.98], [H - pad.b, pad.t])
  const yAlign = linScale([0.954, 0.981], [H - pad.b, pad.t])
  const yRank = linScale([10.5, 0.5], [H - pad.b, pad.t])
  const items = cs.top

  return (
    <div className="figure">
      <div className="stepper" role="tablist" aria-label="Case study steps">
        {STEPS.map((s) => (
          <button key={s.id} role="tab" type="button" aria-selected={s.id === step} className="step" onClick={() => setStep(s.id)}>
            <span className="num">0{s.id + 1}</span> {s.name}
          </button>
        ))}
        <button type="button" className="btn step-next" onClick={() => setStep((step + 1) % 4)}>
          {step === 3 ? 'Replay' : 'Next →'}
        </button>
      </div>
      <p className="step-text" aria-live="polite">
        {STEPS[step].text}
      </p>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Slope chart: the hybrid's final top ten candidates for query P_797406, from Stage-1 cosine to alignment score to final rank.">
          <text x={colA} y={14} textAnchor="middle" className="axis-t light">
            STAGE-1 COSINE
          </text>
          <text x={colB} y={14} textAnchor="middle" className="axis-t light" opacity={step >= 2 ? 1 : 0.25}>
            ALIGNMENT SCORE
          </text>
          <text x={colC} y={14} textAnchor="middle" className="axis-t light" opacity={step >= 3 ? 1 : 0.25}>
            FINAL RANK
          </text>
          {[colA, colB, colC].map((x, i) => (
            <line key={x} x1={x} x2={x} y1={pad.t - 8} y2={H - pad.b + 8} stroke="#37332a" opacity={i === 0 || step >= i + 1 ? 1 : 0.3} />
          ))}
          {[0.88, 0.9, 0.92, 0.94, 0.96, 0.98].map((v) => (
            <text key={v} x={colA - 10} y={yCos(v) + 4} textAnchor="end" className="axis-t dim">
              {v.toFixed(2)}
            </text>
          ))}
          {step >= 2 &&
            [0.955, 0.965, 0.975].map((v) => (
              <text key={v} x={colB + 10} y={yAlign(v) + 4} className="axis-t dim">
                {v.toFixed(3)}
              </text>
            ))}
          {items.map((it) => {
            const isP = it.candidate === CASE_PARTNER
            const color = isP ? '#56b58f' : '#8f8877'
            const a = yCos(it.cos)
            const b = yAlign(it.align)
            const c = yRank(it.rank)
            return (
              <g key={it.candidate} style={{ transition: 'opacity .4s' }}>
                {step >= 2 && <line x1={colA} y1={a} x2={colB} y2={b} stroke={isP ? 'var(--path)' : '#4a453a'} strokeWidth={isP ? 2.2 : 1} />}
                {step >= 3 && <line x1={colB} y1={b} x2={colC} y2={c} stroke={isP ? 'var(--path)' : '#4a453a'} strokeWidth={isP ? 2.2 : 1} />}
                <circle cx={colA} cy={a} r={isP ? 6 : 3.5} fill={isP ? color : 'none'} stroke={color} strokeWidth={1.4} />
                {step >= 2 && <circle cx={colB} cy={b} r={isP ? 6 : 3.5} fill={isP ? color : 'none'} stroke={color} strokeWidth={1.4} />}
                {step >= 3 && (
                  <>
                    <circle cx={colC} cy={c} r={isP ? 6 : 3.5} fill={isP ? color : 'none'} stroke={color} strokeWidth={1.4} />
                    <text x={colC + 12} y={c + 4} className="axis-t" fill={isP ? '#56b58f' : '#8f8877'}>
                      {narrow ? `#${it.rank}` : `#${it.rank} ${it.candidate}`}
                    </text>
                  </>
                )}
              </g>
            )
          })}
          {/* partner annotation at Stage 1 */}
          <g>
            <text x={colA + 14} y={yCos(cs.case.partner_cos) + 4} className="axis-t" fill="#56b58f">
              {CASE_PARTNER} · Stage-1 rank {cs.case.rank_global}
            </text>
          </g>
          {step === 1 && (
            <g>
              <rect x={colA - 4} y={pad.t - 8} width={colB - colA - 30} height={H - pad.b - pad.t + 16} fill="none" stroke="var(--panel-ink)" strokeDasharray="3 4" />
              <text x={colA + 4} y={H - 6} className="axis-t light">
                all ten are inside the K = 30 shortlist
              </text>
            </g>
          )}
        </svg>
      </div>
      <Caption
        label="Figure 8"
        source={['reports/results/hybrid_dev_rankings_top10.csv (cosine)', 'reports/results/classical_dev_rankings.csv (alignment score)', 'reports/results/error_cases.csv']}
      >
        The ten candidates the hybrid finally ranks highest for <span className="pid">{CASE_QUERY}</span>, with their Stage-1 cosine and 96-frame
        alignment score. Other shortlist members are not drawn. The partner's full Stage-1 rank (15) comes from the error-case file.
      </Caption>
    </div>
  )
}

/* ------------------------------------------------------------------ the matrix */

const PAIRS = [
  { key: `${CASE_QUERY}|${CASE_PARTNER}`, label: `True cover · ${CASE_PARTNER}` },
  { key: `${CASE_QUERY}|P_31055`, label: 'Top non-cover · P_31055' },
  { key: 'P_242247|P_476324', label: 'Hero pair · P_476324' },
]

function AlignmentLab() {
  const [key, setKey] = useState(PAIRS[0].key)
  const a = alignments[key]
  const dtw = useMemo(() => subsequenceDTW(a.cost), [a])
  const reduced = useReducedMotion()
  const [prog, setProg] = useState(1) // 0..1 fill, 1..2 backtrack
  const [view, setView] = useState<'C' | 'D'>('C')
  const [showPub, setShowPub] = useState(false)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const raf = useRef(0)

  const run = () => {
    cancelAnimationFrame(raf.current)
    if (reduced) {
      setProg(2)
      return
    }
    setView('D')
    const t0 = performance.now()
    const tick = (now: number) => {
      const t = Math.min(2, (now - t0) / 1700)
      setProg(t)
      if (t < 2) raf.current = requestAnimationFrame(tick)
      else setView('C')
    }
    raf.current = requestAnimationFrame(tick)
  }
  useEffect(() => () => cancelAnimationFrame(raf.current), [])
  useEffect(() => setProg(2), [key])

  const pubByRow = useMemo(() => new Map(a.published_path.map(([i, j]) => [i, j])), [a])
  const ourByRow = useMemo(() => {
    const m = new Map<number, number[]>()
    for (const [i, j] of dtw.path) m.set(i, [...(m.get(i) ?? []), j])
    return m
  }, [dtw])
  const deviation = useMemo(() => {
    const d: number[] = []
    for (const [i, j] of pubByRow) {
      const ours = ourByRow.get(i)
      if (ours) d.push(Math.abs(ours.reduce((s, x) => s + x, 0) / ours.length - j))
    }
    return d.reduce((s, x) => s + x, 0) / d.length
  }, [pubByRow, ourByRow])

  const hv = hover ? a.cost[hover[0]][hover[1]] : null
  const onPath = hover ? dtw.path.some(([i, j]) => i === hover[0] && j === hover[1]) : false

  return (
    <div className="figure lab">
      <div className="split">
        <div className="body prose">
          <h3 className="sub">Why it moved</h3>
          <p>
            Below is the actual matrix for this pair, recovered from the committed figure. Run the alignment and the page computes subsequence DTW in
            your browser, with the repository's steps and weights: the accumulated cost fills row by row, then the cheapest path is traced back from
            the best end column. Compare the true cover with <span className="pid">P_31055</span>, a tonally static hub that is a top-10 false
            positive for 10 of 20 development queries under classical alignment.
          </p>
        </div>
      </div>
      <div className="controls">
        <Seg label="Pair" value={key} onChange={setKey} options={PAIRS.map((p) => ({ value: p.key, label: p.label }))} />
        <button type="button" className="btn" onClick={run}>
          ▶ Run DTW
        </button>
        <Seg
          label="Matrix"
          value={view}
          onChange={setView}
          options={[
            { value: 'C', label: 'Local cost C' },
            { value: 'D', label: 'Accumulated D' },
          ]}
        />
        <label className="check">
          <input type="checkbox" checked={showPub} onChange={(e) => setShowPub(e.target.checked)} /> published path
        </label>
      </div>
      <div className="lab-grid">
        <MatrixCanvas
          cost={a.cost}
          dtw={dtw}
          view={view}
          prog={prog}
          pub={showPub ? a.published_path : null}
          onHover={setHover}
          label={key}
        />
        <div className="lab-side">
          <div className="readout">
            <div className="label">Pair</div>
            <div>
              <span className="q">{key.split('|')[0]}</span> × <span className={a.kind === 'cover' ? 'c' : 'h'}>{key.split('|')[1]}</span>
            </div>
            <div>
              <span className="k">relation </span>
              {a.kind}
            </div>
            <div>
              <span className="k">key shift </span>+{a.shift}
            </div>
            <div>
              <span className="k">path cost (published) </span>
              <b>{a.published_cost.toFixed(3)}</b>
            </div>
            <div>
              <span className="k">recomputed path vs published </span>
              {deviation.toFixed(1)} frames mean |Δ|
            </div>
            <div className="lab-hover">
              {hover ? (
                <>
                  <span className="k">query </span>
                  {hover[0]} <span className="k">· candidate </span>
                  {hover[1]}
                  <br />
                  <span className="k">relative cost </span>
                  {hv!.toFixed(2)} {onPath && <span className="p">on path</span>}
                </>
              ) : (
                <span className="k">hover the matrix to inspect a cell</span>
              )}
            </div>
          </div>
          <StepGlyph />
        </div>
      </div>
      <Caption label="Figure 9" source={`${a.source} (colour-map inversion; 0 = figure's 1st percentile, 1 = 99th)`}>
        Query frames run up, candidate frames across; bright = similar (low cosine distance). Each matrix has its own colour scale, as in the repository's figures, so compare the shape of the groove, not overall brightness. Costs are the figure's relative colour scale, so the path the page finds can
        differ by a few frames from the one the repository drew from exact values; tick "published path" to compare. Path costs quoted are the
        repository's. Lower is better.
      </Caption>
    </div>
  )
}

function MatrixCanvas({
  cost,
  dtw,
  view,
  prog,
  pub,
  onHover,
  label,
}: {
  cost: number[][]
  dtw: ReturnType<typeof subsequenceDTW>
  view: 'C' | 'D'
  prog: number
  pub: [number, number][] | null
  onHover: (h: [number, number] | null) => void
  label: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>(520)
  const size = Math.max(160, Math.min(width - 26, 560))
  const canvas = useRef<HTMLCanvasElement>(null)
  const N = cost.length
  const M = cost[0].length

  // row-normalised accumulated cost, for display
  const accNorm = useMemo(() => {
    const out: number[][] = []
    for (let i = 0; i < N; i++) {
      let lo = Infinity
      let hi = -Infinity
      for (let j = 0; j < M; j++) {
        const v = dtw.acc[i * M + j]
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
      }
      out.push(Array.from({ length: M }, (_, j) => {
        const v = dtw.acc[i * M + j]
        return Number.isFinite(v) ? (v - lo) / (hi - lo || 1) : 1
      }))
    }
    return out
  }, [dtw, N, M])

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    c.width = size * dpr
    c.height = size * dpr
    const ctx = c.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const cw = size / M
    const ch = size / N
    const table = lut('ember')
    const rowsFilled = Math.floor(Math.min(1, prog) * N)
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        let v: number
        if (view === 'D') v = i < rowsFilled || prog >= 1 ? 1 - accNorm[i][j] : 0
        else v = 1 - cost[i][j]
        const k = Math.max(0, Math.min(255, Math.round(v * 255))) * 4
        ctx.fillStyle = `rgb(${table[k]},${table[k + 1]},${table[k + 2]})`
        ctx.fillRect(j * cw, size - (i + 1) * ch, cw + 0.5, ch + 0.5)
      }
    }
    if (view === 'D' && prog < 1) {
      ctx.fillStyle = 'rgba(223,59,30,.9)'
      ctx.fillRect(0, size - rowsFilled * ch - 1.5, size, 2)
    }
    const drawPath = (pts: [number, number][], color: string, w: number, dash: number[] = []) => {
      ctx.strokeStyle = color
      ctx.lineWidth = w
      ctx.setLineDash(dash)
      ctx.beginPath()
      pts.forEach(([i, j], k) => {
        const x = (j + 0.5) * cw
        const y = size - (i + 0.5) * ch
        if (k) ctx.lineTo(x, y)
        else ctx.moveTo(x, y)
      })
      ctx.stroke()
      ctx.setLineDash([])
    }
    if (pub) drawPath(pub, '#f7efdd', 1.5, [4, 3])
    if (prog > 1) {
      // backtracking: reveal from the end cell towards the start
      const n = Math.ceil((prog - 1) * dtw.path.length)
      const pts = dtw.path.slice(dtw.path.length - n)
      drawPath(pts, '#df3b1e', 2.4)
      if (pts.length) {
        const [i, j] = pts[0]
        ctx.fillStyle = '#df3b1e'
        ctx.beginPath()
        ctx.arc((j + 0.5) * cw, size - (i + 0.5) * ch, 4, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }, [size, cost, accNorm, view, prog, pub, dtw, N, M])

  const onMove = (e: React.PointerEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const j = Math.floor(((e.clientX - r.left) / r.width) * M)
    const i = N - 1 - Math.floor(((e.clientY - r.top) / r.height) * N)
    if (i >= 0 && i < N && j >= 0 && j < M) onHover([i, j])
  }

  return (
    <div ref={ref} className="matrix">
      <div className="matrix-axes" style={{ width: size }}>
        <canvas
          ref={canvas}
          style={{ width: size, height: size, touchAction: 'none' }}
          onPointerMove={onMove}
          onPointerDown={onMove}
          onPointerLeave={() => onHover(null)}
          role="img"
          aria-label={`Cross-similarity matrix and DTW path for ${label.replace('|', ' versus ')}`}
        />
        <div className="axis-lbl x">candidate frame →</div>
        <div className="axis-lbl y">query frame →</div>
      </div>
    </div>
  )
}

function StepGlyph() {
  const s = 26
  return (
    <div className="step-glyph">
      <div className="label">Allowed steps · weights</div>
      <svg width={s * 4} height={s * 3.2} viewBox={`0 0 ${s * 4} ${s * 3.2}`} aria-label="Steps (1,1) weight 1, (2,1) weight 2, (1,2) weight 1">
        {[0, 1, 2, 3].map((x) => [0, 1, 2].map((y) => <circle key={`${x}${y}`} cx={x * s + 10} cy={(2 - y) * s + 12} r={2} fill="#6d6758" />))}
        {[
          [1, 1, '1'],
          [2, 1, '2'],
          [1, 2, '1'],
        ].map(([di, dj, w]) => (
          <g key={`${di}${dj}`}>
            <line x1={10} y1={2 * s + 12} x2={Number(dj) * s + 10} y2={(2 - Number(di)) * s + 12} stroke="#df3b1e" strokeWidth={1.6} />
            <text x={Number(dj) * s + 16} y={(2 - Number(di)) * s + 10} className="axis-t light">
              {w}
            </text>
          </g>
        ))}
      </svg>
      <p className="glyph-note">
        Each step advances the query by one or two frames and the candidate by one or two, so tempo can differ up to 2:1 locally. The query may
        start and end anywhere in the candidate (subsequence DTW).
      </p>
    </div>
  )
}
