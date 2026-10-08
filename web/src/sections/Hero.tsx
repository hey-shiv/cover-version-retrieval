import { useEffect, useMemo, useRef, useState } from 'react'
import { alignments, hpcp, rotationPairs } from '../data/figures'
import { subsequenceDTW } from '../lib/dtw'
import { PITCH_CLASSES } from '../lib/chroma'
import { useReducedMotion, useStickyProgress, useWidth } from '../lib/hooks'
import { ease, phase } from '../lib/format'
import { lut } from '../components/ramp'

const QUERY = 'P_242247'
const CAND = 'P_476324'
const SHIFT = 7 // best shift in calibration_rotation_scores.png

export function Hero() {
  const reduced = useReducedMotion()
  const [boxRef, progress] = useStickyProgress<HTMLDivElement>()
  const [manual, setManual] = useState(1)
  const p = reduced ? manual : progress

  const q = hpcp[QUERY].values
  const c = hpcp[CAND].values
  const pair = rotationPairs.find((r) => r.query === QUERY && r.candidate === CAND)!
  // The path the site computes itself: subsequence DTW over the decoded cost matrix.
  const path = useMemo(() => subsequenceDTW(alignments[`${QUERY}|${CAND}`].cost).path, [])

  const rot = ease(phase(p, 0.18, 0.46)) * SHIFT
  const lines = ease(phase(p, 0.5, 0.82))
  const stage = p < 0.16 ? 0 : p < 0.48 ? 1 : p < 0.86 ? 2 : 3
  const shownShift = Math.round(rot)

  return (
    <div ref={boxRef} className="hero-track" style={{ height: reduced ? 'auto' : '320vh' }}>
      <header className="hero panel" style={{ position: reduced ? 'relative' : 'sticky' }}>
        <div className="hero-top wrap">
          <div className="label hero-kicker">Cover Version Retrieval · research log · Da-TACOS</div>
          <h1 className="hero-title">Structure-Aware Hybrid Retrieval for Cover Song Identification</h1>
        </div>

        <div className="hero-stage wrap">
          <p className="statement hero-statement" aria-label="Same work. Different recording.">
            <span>Same work.</span> <em>Different recording.</em>
          </p>

          <HeroCanvas q={q} c={c} rot={rot} lines={lines} path={path} />

          <ol className="hero-steps" aria-label="What the figure shows">
            <li data-on={stage >= 0}>
              <span className="num">01</span> Two recordings of one work, as the system sees them: 12 pitch classes × 96 time steps.
            </li>
            <li data-on={stage >= 1}>
              <span className="num">02</span> Different key. Rotate the candidate{' '}
              <span className="num">+{shownShift}</span> semitones · profile cosine{' '}
              <span className="num">{pair.profile_cosine[shownShift].toFixed(2)}</span>
            </li>
            <li data-on={stage >= 2}>
              <span className="num">03</span> Different timing. Dynamic time warping pairs each query frame with a candidate frame.
            </li>
            <li data-on={stage >= 3}>
              <span className="num">04</span> Same work. Path cost <span className="num">{alignments[`${QUERY}|${CAND}`].published_cost.toFixed(3)}</span>;
              against a non-cover, <span className="num">{alignments[`${QUERY}|P_130947`].published_cost.toFixed(3)}</span>.
            </li>
          </ol>
        </div>

        <div className="hero-foot wrap">
          <span className="source">
            {QUERY} × {CAND}, a cover pair from the calibration works · HPCP decoded from reports/figures/calibration_cover_pairs_hpcp.png ·
            path recomputed in your browser
          </span>
          {reduced ? (
            <button className="btn" type="button" onClick={() => setManual(manual === 1 ? 0 : 1)}>
              {manual === 1 ? 'Show original key' : 'Show aligned'}
            </button>
          ) : (
            <span className="label hero-scroll" style={{ opacity: 1 - phase(p, 0, 0.1) }}>
              Scroll ↓
            </span>
          )}
        </div>
      </header>
    </div>
  )
}

function HeroCanvas({
  q,
  c,
  rot,
  lines,
  path,
}: {
  q: number[][]
  c: number[][]
  rot: number
  lines: number
  path: [number, number][]
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>(900)
  const ref = useRef<HTMLCanvasElement>(null)
  const labelW = width < 560 ? 26 : 34
  const [vh, setVh] = useState(() => (typeof window === 'undefined' ? 900 : window.innerHeight))
  useEffect(() => {
    const on = () => setVh(window.innerHeight)
    window.addEventListener('resize', on)
    return () => window.removeEventListener('resize', on)
  }, [])
  // fit the whole hero in one viewport: strips and gap scale with height as well as width
  const stripH = Math.max(52, Math.min(130, width * 0.12, vh * 0.11))
  const gap = Math.max(44, Math.min(120, width * 0.12, vh * 0.1))
  const height = stripH * 2 + gap + 36

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    const table = lut('ember')
    const x0 = labelW
    const w = width - labelW
    const cw = w / 96
    const rh = stripH / 12
    const topY = 18
    const botY = topY + stripH + gap
    const color = (v: number) => {
      const i = Math.max(0, Math.min(255, Math.round(v * 255))) * 4
      return `rgb(${table[i]},${table[i + 1]},${table[i + 2]})`
    }

    // query strip (pitch class 0 at the bottom)
    for (let pc = 0; pc < 12; pc++)
      for (let t = 0; t < 96; t++) {
        ctx.fillStyle = color(q[pc][t])
        ctx.fillRect(x0 + t * cw, topY + (11 - pc) * rh, cw + 0.6, rh + 0.6)
      }

    // candidate strip, cyclically rotated by `rot` semitones (fractional while animating)
    ctx.save()
    ctx.beginPath()
    ctx.rect(x0, botY, w, stripH)
    ctx.clip()
    for (let pc = 0; pc < 12; pc++) {
      for (const wrap of [0, -12]) {
        const pos = pc + rot + wrap // rotated bin position
        if (pos < -1 || pos > 12) continue
        for (let t = 0; t < 96; t++) {
          ctx.fillStyle = color(c[pc][t])
          ctx.fillRect(x0 + t * cw, botY + (11 - pos) * rh, cw + 0.6, rh + 0.6)
        }
      }
    }
    ctx.restore()

    // pitch-class labels
    ctx.font = `${width < 560 ? 9 : 10}px 'IBM Plex Mono', monospace`
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let pc = 0; pc < 12; pc++) {
      if (width < 560 && pc % 2) continue
      ctx.fillStyle = '#8f8877'
      ctx.fillText(PITCH_CLASSES[pc], x0 - 6, topY + (11.5 - pc) * rh)
      ctx.fillText(PITCH_CLASSES[pc], x0 - 6, botY + (11.5 - pc) * rh)
    }

    // correspondence: one line per aligned pair, drawn progressively along the path
    if (lines > 0) {
      const n = Math.floor(path.length * lines)
      ctx.lineWidth = 1
      for (let k = 0; k < n; k++) {
        if (k % 2) continue
        const [i, j] = path[k]
        const xa = x0 + (i + 0.5) * cw
        const xb = x0 + (j + 0.5) * cw
        const fade = Math.min(1, (n - k) / 12)
        ctx.strokeStyle = `rgba(223,59,30,${0.35 + 0.55 * fade})`
        ctx.beginPath()
        ctx.moveTo(xa, topY + stripH + 3)
        ctx.bezierCurveTo(xa, topY + stripH + gap * 0.5, xb, botY - gap * 0.5, xb, botY - 3)
        ctx.stroke()
      }
    }

    // strip captions
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.font = `11px 'IBM Plex Mono', monospace`
    ctx.fillStyle = '#7f9be0'
    ctx.fillText(`QUERY ${QUERY}`, x0, topY - 6)
    ctx.fillStyle = '#56b58f'
    ctx.fillText(`CANDIDATE ${CAND}${rot > 0.05 ? `  · rotated +${rot.toFixed(rot % 1 ? 1 : 0)}` : ''}`, x0, botY + stripH + 16)
  }, [q, c, rot, lines, path, width, height, labelW, stripH, gap])

  return (
    <div ref={wrapRef} className="hero-canvas">
      <canvas
        ref={ref}
        style={{ width: '100%', height }}
        role="img"
        aria-label={`Chroma of query ${QUERY} above candidate ${CAND}. The candidate rotates by ${SHIFT} semitones until its strong pitch classes line up with the query's, then red lines connect frames matched by dynamic time warping.`}
      />
    </div>
  )
}
