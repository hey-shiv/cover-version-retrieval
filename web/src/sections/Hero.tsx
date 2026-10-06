import { useEffect, useMemo, useRef, useState } from 'react'
import { alignments, hpcp, rotationPairs } from '../data/figures'
import { subsequenceDTW } from '../lib/dtw'
import { PITCH_CLASSES } from '../lib/chroma'
import { useReducedMotion, useStickyProgress, useWidth } from '../lib/hooks'
import { ease, phase } from '../lib/format'
import { lut } from '../components/ramp'
import { Fluid } from '../lib/fluid'
import { PC_CSS, PC_RGB, wheelRGB } from '../lib/pitch'
import { NIGHT, alpha } from '../lib/palette'
import { PAPER_URL, REPO_URL } from '../components/Nav'
import { Seg } from '../components/Chapter'

const QUERY = 'P_242247'
const CAND = 'P_476324'
const SHIFT = 7 // best shift in calibration_rotation_scores.png

/* ------------------------------------------------------------------ hero: chroma poured into a fluid */

const FPS = 3.2 // chroma frames per second: one 96-frame strip every 30 s
const RECORDINGS = {
  q: { pid: QUERY, name: 'Original' },
  c: { pid: CAND, name: 'Cover' },
} as const
type Rec = keyof typeof RECORDINGS

/** Where the wheel of nozzles sits, in UV (y up) with radius as a fraction of height. */
function wheel(w: number, h: number) {
  if (w >= 900) return { cx: 0.71, cy: 0.5, r: Math.min(0.3, (w * 0.2) / h) }
  return { cx: 0.5, cy: 0.73, r: Math.min(0.13, (w * 0.34) / h) }
}

export function Hero() {
  const reduced = useReducedMotion()
  const root = useRef<HTMLElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const labels = useRef<(HTMLSpanElement | null)[]>([])
  const bars = useRef<(HTMLSpanElement | null)[]>([])
  const frameOut = useRef<HTMLSpanElement>(null)
  const [rec, setRec] = useState<Rec>('q')
  const [playing, setPlaying] = useState(!reduced)
  const [supported, setSupported] = useState(true)
  const [geom, setGeom] = useState({ w: 1200, h: 800 })
  const state = useRef({ rec, playing, visible: true })
  state.current.rec = rec
  state.current.playing = playing

  useEffect(() => setPlaying(!reduced), [reduced])

  useEffect(() => {
    const el = root.current
    const cv = canvas.current
    if (!el || !cv) return
    const dpr = Math.min(1.5, window.devicePixelRatio || 1)
    const size = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      if (!w || !h) return false
      cv.width = Math.round(w * dpr)
      cv.height = Math.round(h * dpr)
      setGeom({ w, h })
      return true
    }
    size()
    const narrow = el.clientWidth < 700
    const fluid = Fluid.create(cv, {
      simRes: narrow ? 96 : 128,
      dyeRes: narrow ? 512 : 840,
      velocityDissipation: 0.55,
      densityDissipation: 0.85,
      pressureIterations: 18,
      curl: 22,
      splatRadius: narrow ? 0.34 : 0.26,
      background: [11 / 255, 29 / 255, 54 / 255],
    })
    if (!fluid) {
      setSupported(false)
      return
    }

    const points = new Float32Array(24)
    const forces = new Float32Array(36)
    const colors = new Float32Array(36)
    let t = 0 // position in chroma frames
    let lastFrame = -1

    const layout = () => {
      const { cx, cy, r } = wheel(el.clientWidth, el.clientHeight)
      const aspect = el.clientWidth / el.clientHeight
      for (let p = 0; p < 12; p++) {
        const th = (p / 12) * Math.PI * 2
        points[p * 2] = cx + (r * Math.sin(th)) / aspect
        points[p * 2 + 1] = cy + r * Math.cos(th)
      }
    }
    layout()

    /** Energy per pitch class at time t, sharpened so the strong classes lead. */
    const energy = (vals: number[][], p: number) => {
      const a = Math.floor(t) % 96
      const b = (a + 1) % 96
      const f = t - Math.floor(t)
      const v = vals[p][a] * (1 - f) + vals[p][b] * f
      return Math.max(0, (v - 0.22) / 0.78) ** 1.6
    }

    const pour = (dt: number) => {
      const vals = hpcp[RECORDINGS[state.current.rec].pid].values
      for (let p = 0; p < 12; p++) {
        const e = energy(vals, p)
        const th = (p / 12) * Math.PI * 2
        const s = Math.sin(th)
        const c = Math.cos(th)
        // clockwise along the wheel, a little inward: the classes stir into each other
        const k = e * 2400 * dt
        forces[p * 3] = (c * 0.9 - s * 0.5) * k
        forces[p * 3 + 1] = (-s * 0.9 - c * 0.5) * k
        const d = e * 3.6 * dt
        colors[p * 3] = PC_RGB[p][0] * d
        colors[p * 3 + 1] = PC_RGB[p][1] * d
        colors[p * 3 + 2] = PC_RGB[p][2] * d
        const lab = labels.current[p]
        if (lab) lab.style.opacity = String(0.28 + 0.72 * Math.min(1, e * 1.4))
        const bar = bars.current[p]
        if (bar) bar.style.transform = `scaleY(${0.06 + 0.94 * Math.min(1, e * 1.2)})`
      }
      fluid.nozzles(points, forces, colors)
      const fr = Math.floor(t) % 96
      if (fr !== lastFrame && frameOut.current) {
        lastFrame = fr
        frameOut.current.textContent = String(fr + 1).padStart(2, '0')
      }
    }

    // reduced motion: one still picture of the first 12 s, then nothing moves unless asked
    if (!state.current.playing) {
      for (let i = 0; i < 200; i++) {
        t += FPS / 60
        pour(1 / 60)
        fluid.step(1 / 60)
      }
      fluid.render()
    }

    // dev only: step the simulation by hand when the page is hidden (headless review)
    const devStep = (e: Event) => {
      for (let i = 0; i < (e as CustomEvent<number>).detail; i++) {
        t = (t + FPS / 60) % 96
        pour(1 / 60)
        fluid.step(1 / 60)
      }
      fluid.render()
    }
    if (import.meta.env.DEV) window.addEventListener('hero-step', devStep)

    // pointer stirs the fluid, coloured by the pitch class it is nearest to
    let last: { x: number; y: number } | null = null
    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect()
      const x = (e.clientX - r.left) / r.width
      const y = 1 - (e.clientY - r.top) / r.height
      if (last && state.current.playing) {
        const { cx, cy } = wheel(r.width, r.height)
        const ang = Math.atan2((x - cx) * (r.width / r.height), y - cy)
        const col = wheelRGB(ang).map((v) => v * 0.32) as [number, number, number]
        fluid.splat(x, y, (x - last.x) * 5200, (y - last.y) * 5200, col, 0.8)
      }
      last = { x, y }
    }
    const onLeave = () => (last = null)
    el.addEventListener('pointermove', onMove)
    el.addEventListener('pointerleave', onLeave)

    const io = new IntersectionObserver(([en]) => (state.current.visible = en.isIntersecting))
    io.observe(el)

    let raf = 0
    let prev = performance.now()
    const loop = (now: number) => {
      const dt = Math.min(1 / 30, (now - prev) / 1000)
      prev = now
      if (state.current.playing && state.current.visible && !document.hidden) {
        t = (t + dt * FPS) % 96
        pour(dt)
        fluid.step(dt)
        fluid.render()
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)

    let rt = 0
    const onResize = () => {
      clearTimeout(rt)
      rt = window.setTimeout(() => {
        // a zero-sized (hidden) hero keeps its old buffers
        if (!size()) return
        fluid.resize()
        layout()
        fluid.render()
      }, 150)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(rt)
      io.disconnect()
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerleave', onLeave)
      window.removeEventListener('resize', onResize)
      window.removeEventListener('hero-step', devStep)
      fluid.dispose()
    }
  }, [])

  const { cx, cy, r } = wheel(geom.w, geom.h)
  const lr = r * geom.h * 1.2 // label radius in px

  return (
    <header ref={root} className="hero panel" id="top-hero">
      <canvas ref={canvas} className="hero-fluid" aria-hidden="true" />
      {!supported && <div className="hero-fallback" aria-hidden="true" style={{ left: `${cx * 100}%`, top: `${(1 - cy) * 100}%` }} />}
      <div className="hero-wheel" aria-hidden="true" style={{ left: `${cx * 100}%`, top: `${(1 - cy) * 100}%` }}>
        {PITCH_CLASSES.map((n, p) => {
          const th = (p / 12) * Math.PI * 2
          return (
            <span
              key={n}
              ref={(e) => {
                labels.current[p] = e
              }}
              style={{ transform: `translate(${Math.sin(th) * lr}px, ${-Math.cos(th) * lr}px) translate(-50%, -50%)`, color: PC_CSS[p] }}
            >
              {n}
            </span>
          )
        })}
      </div>

      <div className="hero-copy wrap">
        <p className="hero-kicker">A research report on cover song identification</p>
        <h1 className="hero-title">
          <span className="hero-l1">Same work,</span>
          <span className="hero-l2">different recording.</span>
        </h1>
        <p className="hero-lede">
          Can a machine tell that two recordings are the same song when the key, the tempo and the arrangement have all changed? This page follows one
          retrieval system through 13,000 queries, and shows where it fails.
        </p>
        <div className="hero-actions">
          <a className="btn solid" href="#overture">
            Start reading
          </a>
          <a className="btn" href={PAPER_URL}>
            Read the paper
          </a>
          <a className="btn" href={REPO_URL}>
            Code and results
          </a>
        </div>
      </div>

      <div className="hero-console wrap">
        <div className="console-chroma" aria-hidden="true">
          {PITCH_CLASSES.map((n, p) => (
            <span key={n} className="console-bar">
              <span
                ref={(e) => {
                  bars.current[p] = e
                }}
                style={{ background: PC_CSS[p] }}
              />
            </span>
          ))}
        </div>
        <p className="console-text">
          The fluid is fed by real chroma: twelve nozzles, one per pitch class, pouring {RECORDINGS[rec].pid} frame{' '}
          <span ref={frameOut} className="num">
            01
          </span>{' '}
          of 96.{' '}
          {rec === 'c' ? 'Same work in another key, so every colour turns.' : 'Switch to its cover to see the same work in another key.'}
        </p>
        <div className="console-controls">
          <Seg
            label="Recording"
            value={rec}
            onChange={setRec}
            options={[
              { value: 'q', label: 'Original' },
              { value: 'c', label: 'Cover' },
            ]}
          />
          <button type="button" className="btn console-pause" aria-pressed={!playing} onClick={() => setPlaying(!playing)}>
            {playing ? 'Pause' : 'Play'}
          </button>
        </div>
      </div>
    </header>
  )
}

/* ------------------------------------------------------------------ overture: the idea in one scroll */

export function Overture() {
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
    <div ref={boxRef} id="overture" className="ov-track panel" style={{ height: reduced ? 'auto' : '300vh' }}>
      <section className="ov" style={{ position: reduced ? 'relative' : 'sticky' }} aria-labelledby="ov-title">
        <div className="ov-head wrap">
          <h2 id="ov-title" className="ov-title">
            The whole idea, in one pair
          </h2>
          <p className="ov-sub">Two recordings of one work, as the system sees them: twelve pitch classes across 96 moments.</p>
        </div>

        <div className="ov-stage wrap">
          <OvertureCanvas q={q} c={c} rot={rot} lines={lines} path={path} />
          <ol className="ov-steps" aria-label="What the figure shows">
            <li data-on={stage >= 0}>
              <span className="num">1</span> Same composition, recorded twice. The query on top, a known cover below.
            </li>
            <li data-on={stage >= 1}>
              <span className="num">2</span> Different key. Rotate the cover <span className="num">+{shownShift}</span> semitones; profile cosine
              rises to <span className="num">{pair.profile_cosine[shownShift].toFixed(2)}</span>.
            </li>
            <li data-on={stage >= 2}>
              <span className="num">3</span> Different timing. Dynamic time warping pairs each query moment with a cover moment.
            </li>
            <li data-on={stage >= 3}>
              <span className="num">4</span> Same work. Path cost <span className="num">{alignments[`${QUERY}|${CAND}`].published_cost.toFixed(3)}</span>;
              against a non-cover, <span className="num">{alignments[`${QUERY}|P_130947`].published_cost.toFixed(3)}</span>.
            </li>
          </ol>
        </div>

        <div className="ov-foot wrap">
          <span className="source">
            {QUERY} and {CAND}, a cover pair from the calibration works. HPCP decoded from reports/figures/calibration_cover_pairs_hpcp.png; path
            recomputed in your browser.
          </span>
          {reduced ? (
            <button className="btn" type="button" onClick={() => setManual(manual === 1 ? 0 : 1)}>
              {manual === 1 ? 'Show original key' : 'Show aligned'}
            </button>
          ) : (
            <span className="ov-scroll" style={{ opacity: 1 - phase(p, 0, 0.1) }}>
              Keep scrolling to align them
            </span>
          )}
        </div>
      </section>
    </div>
  )
}

function OvertureCanvas({
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
  // fit the whole figure in one viewport: strips and gap scale with height as well as width
  const stripH = Math.max(52, Math.min(130, width * 0.12, vh * 0.12))
  const gap = Math.max(44, Math.min(130, width * 0.12, vh * 0.11))
  const height = stripH * 2 + gap + 40

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
    const topY = 20
    const botY = topY + stripH + gap
    const color = (v: number) => {
      const i = Math.max(0, Math.min(255, Math.round(v * 255))) * 4
      return `rgb(${table[i]},${table[i + 1]},${table[i + 2]})`
    }
    const rounded = (x: number, y: number, ww: number, hh: number) => {
      ctx.beginPath()
      ctx.roundRect(x, y, ww, hh, 6)
    }

    // query strip (pitch class 0 at the bottom)
    ctx.save()
    rounded(x0, topY, w, stripH)
    ctx.clip()
    for (let pc = 0; pc < 12; pc++)
      for (let t = 0; t < 96; t++) {
        ctx.fillStyle = color(q[pc][t])
        ctx.fillRect(x0 + t * cw, topY + (11 - pc) * rh, cw + 0.6, rh + 0.6)
      }
    ctx.restore()

    // candidate strip, cyclically rotated by `rot` semitones (fractional while animating)
    ctx.save()
    rounded(x0, botY, w, stripH)
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

    // pitch-class labels, in their wheel colours
    ctx.font = `500 ${width < 560 ? 9 : 10.5}px 'Archivo Variable', sans-serif`
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let pc = 0; pc < 12; pc++) {
      if (width < 560 && pc % 2) continue
      ctx.fillStyle = PC_CSS[pc]
      ctx.fillText(PITCH_CLASSES[pc], x0 - 7, topY + (11.5 - pc) * rh)
      ctx.fillText(PITCH_CLASSES[pc], x0 - 7, botY + (11.5 - pc) * rh)
    }

    // correspondence: one line per aligned pair, drawn progressively along the path
    if (lines > 0) {
      const n = Math.floor(path.length * lines)
      ctx.lineWidth = 1.1
      for (let k = 0; k < n; k++) {
        if (k % 2) continue
        const [i, j] = path[k]
        const xa = x0 + (i + 0.5) * cw
        const xb = x0 + (j + 0.5) * cw
        const fade = Math.min(1, (n - k) / 12)
        ctx.strokeStyle = alpha(NIGHT.path, 0.3 + 0.6 * fade)
        ctx.beginPath()
        ctx.moveTo(xa, topY + stripH + 4)
        ctx.bezierCurveTo(xa, topY + stripH + gap * 0.5, xb, botY - gap * 0.5, xb, botY - 4)
        ctx.stroke()
      }
    }

    // strip captions
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.font = `600 12px 'Archivo Variable', sans-serif`
    ctx.fillStyle = NIGHT.query
    ctx.fillText(`Query ${QUERY}`, x0, topY - 7)
    ctx.fillStyle = NIGHT.cover
    ctx.fillText(`Cover ${CAND}${rot > 0.05 ? `, rotated +${rot.toFixed(rot % 1 ? 1 : 0)}` : ''}`, x0, botY + stripH + 17)
  }, [q, c, rot, lines, path, width, height, labelW, stripH, gap])

  return (
    <div ref={wrapRef} className="ov-canvas">
      <canvas
        ref={ref}
        style={{ width: '100%', height }}
        role="img"
        aria-label={`Chroma of query ${QUERY} above its cover ${CAND}. The cover rotates by ${SHIFT} semitones until its strong pitch classes line up with the query's, then lines connect moments matched by dynamic time warping.`}
      />
    </div>
  )
}
