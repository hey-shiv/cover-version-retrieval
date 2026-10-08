import { useMemo, useRef, useState } from 'react'
import { Chapter, Caption, Seg } from '../components/Chapter'
import { hpcp, rotationPairs, rotationSource } from '../data/figures'
import { results } from '../data/results'
import { PITCH_CLASSES, profile, rotate } from '../lib/chroma'

export function Rotation() {
  const [pi, setPi] = useState(1)
  const pair = rotationPairs[pi]
  const [k, setK] = useState(0)
  const qp = useMemo(() => profile(hpcp[pair.query].values), [pair])
  const cp = useMemo(() => profile(hpcp[pair.candidate].values), [pair])
  const rotated = rotate(cp, k)
  const cos = pair.profile_cosine
  const shifts = results.classical.shift_histogram
  const totalPairs = shifts.reduce((a, b) => a + b, 0)

  return (
    <Chapter id="rotation" no="05 · Key" title={<>Twelve ways <em>to be in tune</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">A cover in another key is the same pattern, cyclically shifted along the pitch-class axis.</p>
          <p>
            So before aligning, the system tries all twelve rotations of the candidate and keeps the one whose average pitch-class profile has the
            highest cosine with the query's. Key annotations in the metadata are never used; the choice comes from the features alone.
          </p>
          <p>
            Turn the dial. Then switch to a <strong>non-cover</strong> pair and notice the catch: rotation gives <em>every</em> candidate twelve chances to
            match. Against query <span className="pid">P_510723</span>, the non-cover <span className="pid">P_355091</span> reaches 0.98, higher than
            that query's true cover, which peaks at 0.97.
          </p>
        </div>
        <aside className="aside">
          Across all {totalPairs.toLocaleString('en-US')} development query–candidate pairs the chosen shift is spread over all twelve values. Half of
          the twenty true partners needed no shift at all. On that sample, key handling neither helped nor hurt (ΔAP +0.055 without it, CI spans zero).
        </aside>
      </div>

      <div className="figure rotation-fig">
        <div className="controls">
          <Seg
            label="Pair"
            value={String(pi)}
            onChange={(v) => {
              setPi(Number(v))
              setK(0)
            }}
            options={rotationPairs.map((p, i) => ({ value: String(i), label: `${p.kind === 'cover' ? 'Cover' : 'Non-cover'} · ${p.query.slice(2)}/${p.candidate.slice(2)}` }))}
          />
        </div>
        <div className="rotation-grid">
          <Dial qp={qp} cp={rotated} k={k} onK={setK} />
          <div>
            <div className="label">Profile cosine for each candidate rotation (published)</div>
            <div className="bars" role="group" aria-label="Profile cosine by rotation">
              {cos.map((v, s) => (
                <button
                  key={s}
                  type="button"
                  className="bar"
                  aria-pressed={s === k}
                  aria-label={`Rotate ${s} semitones: cosine ${v.toFixed(2)}`}
                  onClick={() => setK(s)}
                >
                  <span
                    className="bar-fill"
                    style={{
                      height: `${((v - 0.5) / 0.5) * 100}%`,
                      background: s === pair.best_shift ? 'var(--path)' : s === k ? 'var(--ink)' : 'var(--faint)',
                    }}
                  />
                  <span className="bar-x num">{s}</span>
                </button>
              ))}
            </div>
            <div className="readout rot-readout">
              <div>
                <span className="k">rotation </span>+{k} semitones
              </div>
              <div>
                <span className="k">profile cosine </span>
                {cos[k].toFixed(3)} {k === pair.best_shift && <span className="p">← chosen</span>}
              </div>
              <div>
                <span className="k">pair </span>
                <span className="q">{pair.query}</span> · <span className={pair.kind === 'cover' ? 'c' : 'h'}>{pair.candidate}</span> ({pair.kind})
              </div>
            </div>
          </div>
        </div>
      </div>
      <Caption label="Figure 7" source={[rotationSource + ' (bar heights, y axis 0.5–1 shown)', 'reports/figures/calibration_*_pairs_hpcp.png (profiles)']}>
        The dial draws the mean pitch-class profile of the <span className="q">query</span> and of the rotated{' '}
        <span className={pair.kind === 'cover' ? 'c' : 'h'}>candidate</span>, computed in your browser from HPCP decoded from the committed figures.
        The bars are the profile cosines read from the published figure. Recomputing them from the decoded features picks the same best shift for
        all four pairs.
      </Caption>
    </Chapter>
  )
}

function Dial({ qp, cp, k, onK }: { qp: number[]; cp: number[]; k: number; onK: (k: number) => void }) {
  const R = 118
  const svg = useRef<SVGSVGElement>(null)
  const max = Math.max(...qp, ...cp)
  const pt = (v: number, p: number) => {
    const a = (p / 12) * Math.PI * 2 - Math.PI / 2
    const r = 18 + (v / max) * (R - 18)
    return `${(Math.cos(a) * r).toFixed(1)},${(Math.sin(a) * r).toFixed(1)}`
  }
  const drag = (e: React.PointerEvent) => {
    if (e.type === 'pointermove' && e.buttons !== 1) return
    const r = svg.current!.getBoundingClientRect()
    const x = e.clientX - (r.left + r.width / 2)
    const y = e.clientY - (r.top + r.height / 2)
    const ang = (Math.atan2(y, x) + Math.PI / 2 + Math.PI * 2) % (Math.PI * 2)
    onK(Math.round((ang / (Math.PI * 2)) * 12) % 12)
  }
  return (
    <div className="dial-wrap">
      <svg
        ref={svg}
        viewBox="-150 -160 300 310"
        className="dial"
        onPointerDown={drag}
        onPointerMove={drag}
        role="slider"
        tabIndex={0}
        aria-label="Candidate rotation in semitones"
        aria-valuemin={0}
        aria-valuemax={11}
        aria-valuenow={k}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onK((k + 1) % 12)
          if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onK((k + 11) % 12)
        }}
      >
        {[0.33, 0.66, 1].map((f) => (
          <circle key={f} r={18 + f * (R - 18)} fill="none" stroke="var(--rule)" />
        ))}
        {PITCH_CLASSES.map((n, p) => {
          const a = (p / 12) * Math.PI * 2 - Math.PI / 2
          return (
            <g key={n}>
              <line x1={Math.cos(a) * 18} y1={Math.sin(a) * 18} x2={Math.cos(a) * R} y2={Math.sin(a) * R} stroke="var(--rule)" />
              <text x={Math.cos(a) * (R + 16)} y={Math.sin(a) * (R + 16)} textAnchor="middle" dominantBaseline="middle" className="axis-t">
                {n}
              </text>
            </g>
          )
        })}
        <polygon points={qp.map(pt).join(' ')} fill="rgba(42,74,148,.12)" stroke="var(--query)" strokeWidth={1.8} />
        <polygon points={cp.map(pt).join(' ')} fill="rgba(29,115,86,.10)" stroke="var(--cover)" strokeWidth={1.8} strokeDasharray="4 3" style={{ transition: 'all .35s ease' }} />
        <g transform={`rotate(${(k / 12) * 360})`} style={{ transition: 'transform .35s ease' }}>
          <line x1={0} y1={-18} x2={0} y2={-R} stroke="var(--path)" strokeWidth={1.5} strokeDasharray="3 3" />
          <path d={`M 0 ${-R - 26} l -5 -8 h 10 z`} fill="var(--path)" />
        </g>
        <text textAnchor="middle" dominantBaseline="middle" className="dial-k num">
          +{k}
        </text>
      </svg>
      <div className="label dial-hint">Drag the dial · ← → keys · or pick a bar</div>
    </div>
  )
}
