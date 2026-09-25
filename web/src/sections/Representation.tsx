import { useEffect, useRef, useState } from 'react'
import { Chapter, Caption } from '../components/Chapter'
import { Heatmap } from '../components/Heatmap'
import { profileFigure } from '../data/figures'
import { PITCH_CLASSES } from '../lib/chroma'
import { rampCss } from '../components/ramp'

const RAW = profileFigure.raw_frames // 15,164 raw HPCP frames for P_130947 (figure title)
const N = 96

export function Representation() {
  const pre = profileFigure.preprocessed
  const raw = profileFigure.raw_pixels
  const [pos, setPos] = useState(52)
  const held = useRef(false)
  const t = Math.min(N - 1, Math.floor(pos))
  const setT = (v: number | ((x: number) => number)) =>
    setPos((p) => (typeof v === 'function' ? v(Math.floor(p)) : v))

  // Slow, endless drift along the strip; pauses only while the pointer is on it.
  useEffect(() => {
    const SPEED = 1.2 // frames per second
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      if (!held.current) setPos((p) => (p + dt * SPEED) % N)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])
  const per = RAW / N
  const rawFrom = Math.round(t * per)
  const rawTo = Math.round((t + 1) * per) - 1
  const column = pre.map((row) => row[t])

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    setT(Math.max(0, Math.min(N - 1, Math.floor(((e.clientX - r.left) / r.width) * N))))
  }
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowRight') setT((x) => Math.min(N - 1, x + 1))
    if (e.key === 'ArrowLeft') setT((x) => Math.max(0, x - 1))
  }

  return (
    <Chapter id="representation" no="02 · What the system sees" title={<>Twelve numbers <em>per moment</em></>} dark>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            Da-TACOS ships no audio. What it ships is HPCP: for every short frame, how much energy falls on each of the 12 pitch classes, octave folded
            away.
          </p>
          <p>
            That throws out most of what distinguishes one recording from another (timbre, loudness, production) and keeps harmony. A recording of
            about four minutes becomes roughly 18,000 frames of 12 numbers. The classical stage then averages that down to a fixed <strong>96</strong>{' '}
            frames, normalising each frame to unit length before and after.
          </p>
        </div>
        <aside className="aside">
          <span className="mono">(T, 12) → (12, T) → frame L2 → area-resample → frame L2</span>
          <br />
          Silent frames stay zero and cost 1 against everything. The encoder sees the same recipe at 512 frames, pooled to 256.
        </aside>
      </div>

      <div className="figure hpcp-instrument">
        <div className="hpcp-raw">
          <div className="label">
            Raw · {RAW.toLocaleString('en-US')} frames · P_130947
          </div>
          <div className="strip raw" style={{ aspectRatio: `${raw[0].length} / 30` }}>
            <Heatmap data={raw} ramp="ember" ariaLabel="Raw HPCP of P_130947, 15,164 frames" />
            <div className="bracket" style={{ left: `${(t / N) * 100}%`, width: `${100 / N}%` }} />
          </div>
          <div className="funnel" aria-hidden="true">
            <svg viewBox="0 0 100 10" preserveAspectRatio="none">
              <path d={`M ${(t / N) * 100} 0 L ${((t + 1) / N) * 100} 0 L ${((t + 1) / N) * 100} 10 L ${(t / N) * 100} 10 Z`} fill="rgba(223,59,30,.35)" />
            </svg>
          </div>
          <div className="label">Preprocessed · 96 frames · each averages ≈ {Math.round(per)} raw frames</div>
          <div
            className="strip scrub"
            style={{ aspectRatio: '96 / 22' }}
            onPointerMove={onMove}
            onPointerDown={onMove}
            onPointerEnter={() => (held.current = true)}
            onPointerLeave={() => (held.current = false)}
            onKeyDown={onKey}
            tabIndex={0}
            role="slider"
            aria-label="Time frame"
            aria-valuemin={0}
            aria-valuemax={N - 1}
            aria-valuenow={t}
          >
            <Heatmap data={pre} ramp="ember" ariaLabel="Preprocessed HPCP, 12 by 96" />
            <div className="cursor" style={{ left: `${(pos / N) * 100}%` }} />
          </div>
          <div className="scrub-hint label">Drifts on its own · hover or drag to take over · ← → keys</div>
        </div>

        <div className="hpcp-readout">
          <PitchClock values={column} />
          <div className="readout">
            <div>
              <span className="k">frame </span>
              {String(t).padStart(2, '0')} / 95
            </div>
            <div>
              <span className="k">raw span </span>
              {rawFrom.toLocaleString('en-US')}–{rawTo.toLocaleString('en-US')}
            </div>
            <div>
              <span className="k">strongest </span>
              {PITCH_CLASSES[column.indexOf(Math.max(...column))]}
            </div>
          </div>
        </div>
      </div>
      <Caption
        label="Figure 2"
        source="reports/figures/profile_raw_vs_resampled.png (colour-map inversion, per-panel scale; values are relative)"
      >
        Real HPCP of <span className="pid">P_130947</span>, a calibration track, recovered from the committed figure. The dial shows one 96-frame
        column: spoke length is the relative energy of each pitch class. The raw strip is shown at screen resolution, so it is itself an average.
      </Caption>
    </Chapter>
  )
}

function PitchClock({ values }: { values: number[] }) {
  const R = 86
  const r0 = 22
  return (
    <svg viewBox="-136 -136 272 272" className="pitch-clock" role="img" aria-label="Pitch-class energy for the selected frame">
      <circle r={r0 + R} fill="none" stroke="#37332a" />
      <circle r={r0} fill="none" stroke="#37332a" />
      {values.map((v, p) => {
        const a = (p / 12) * Math.PI * 2 - Math.PI / 2
        const x1 = Math.cos(a) * r0
        const y1 = Math.sin(a) * r0
        const len = r0 + v * R
        return (
          <g key={p}>
            <line x1={x1} y1={y1} x2={Math.cos(a) * (r0 + R)} y2={Math.sin(a) * (r0 + R)} stroke="#2a2720" />
            <line x1={x1} y1={y1} x2={Math.cos(a) * len} y2={Math.sin(a) * len} stroke={rampCss('ember', 0.35 + v * 0.65)} strokeWidth={9} strokeLinecap="butt" />
            <text
              x={Math.cos(a) * (r0 + R + 13)}
              y={Math.sin(a) * (r0 + R + 13)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={10}
              fontFamily="var(--mono)"
              fill="#8f8877"
            >
              {PITCH_CLASSES[p]}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
