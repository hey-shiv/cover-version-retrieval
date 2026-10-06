import { useEffect, useState } from 'react'
import { PC_CSS } from '../lib/pitch'

/** The pipeline itself is the table of contents: each stage links to its chapter. */
const STAGES = [
  { id: 'representation', label: 'HPCP' },
  { id: 'encoder', label: 'TCN' },
  { id: 'search', label: 'Top 30' },
  { id: 'rotation', label: '12 keys' },
  { id: 'rescue', label: 'DTW' },
  { id: 'explorer', label: 'Rerank' },
  { id: 'results', label: 'Benchmark' },
  { id: 'hubness', label: 'Failure' },
]

export const REPO_URL = 'https://github.com/hey-shiv/cover-version-retrieval'
export const PAPER_URL = `${REPO_URL}/blob/main/paper/main.pdf`

/** Twelve dots in the pitch-class colours: the site's mark. */
export function WheelMark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-10 -10 20 20" aria-hidden="true">
      {PC_CSS.map((c, p) => {
        const a = (p / 12) * Math.PI * 2
        return <circle key={p} cx={Math.sin(a) * 7} cy={-Math.cos(a) * 7} r={1.7} fill={c} />
      })}
    </svg>
  )
}

export function Nav() {
  const [progress, setProgress] = useState(0)
  const [onDark, setOnDark] = useState(true)
  const [active, setActive] = useState<string | null>(null)
  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const h = document.documentElement.scrollHeight - window.innerHeight
      setProgress(h > 0 ? window.scrollY / h : 0)
      // read the background under the bar: night hero / panels vs paper
      const under = document.elementsFromPoint(window.innerWidth / 2, 28).find((e) => !e.closest('.nav'))
      setOnDark(!!under?.closest('.panel'))
      // the stage whose chapter spans the upper third of the viewport
      const probe = window.innerHeight * 0.33
      let current: string | null = null
      for (const st of STAGES) {
        const el = document.getElementById(st.id)
        if (!el) continue
        const r = el.getBoundingClientRect()
        if (r.top <= probe && r.bottom > probe) current = st.id
      }
      setActive(current)
    }
    const on = () => {
      if (!raf) raf = requestAnimationFrame(update)
    }
    update()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => {
      window.removeEventListener('scroll', on)
      window.removeEventListener('resize', on)
    }
  }, [])
  return (
    <nav className={`nav${onDark ? ' on-dark' : ''}`} aria-label="Primary">
      <a href="#top" className="nav-brand">
        <WheelMark />
        <span>Cover Version Retrieval</span>
      </a>
      <ol className="rail" aria-label="Pipeline stages">
        {STAGES.map((st) => (
          <li key={st.id} data-on={active === st.id}>
            <a href={`#${st.id}`} aria-current={active === st.id ? 'step' : undefined}>
              {st.label}
            </a>
          </li>
        ))}
      </ol>
      <ul className="nav-links">
        <li className="nav-log">
          <a href="#journey">Research log</a>
        </li>
        <li>
          <a href={PAPER_URL}>Paper</a>
        </li>
        <li>
          <a href={REPO_URL}>GitHub</a>
        </li>
      </ul>
      <div className="nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </nav>
  )
}
