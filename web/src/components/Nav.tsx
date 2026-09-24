import { useEffect, useState } from 'react'

const LINKS = [
  { href: '#problem', label: 'Research' },
  { href: '#encoder', label: 'System' },
  { href: '#journey', label: 'Log' },
  { href: '#results', label: 'Results' },
]
/** The pipeline itself is the table of contents: each stage links to its chapter. */
const STAGES = [
  { id: 'representation', label: 'HPCP' },
  { id: 'encoder', label: 'TCN' },
  { id: 'search', label: 'top-30' },
  { id: 'rotation', label: '×12 keys' },
  { id: 'rescue', label: 'DTW' },
  { id: 'explorer', label: 'rerank' },
  { id: 'results', label: 'benchmark' },
  { id: 'hubness', label: 'failure' },
]

export const REPO_URL = 'https://github.com/hey-shiv/cover-version-retrieval'

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
      // read the background under the bar: dark hero / panels vs paper
      const under = document.elementsFromPoint(window.innerWidth / 2, 20).find((e) => !e.closest('.nav'))
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
        Cover Version Retrieval
      </a>
      <ol className="rail" aria-label="Pipeline stages">
        {STAGES.map((st, i) => (
          <li key={st.id} data-on={active === st.id}>
            {i > 0 && <span aria-hidden="true">→</span>}
            <a href={`#${st.id}`} aria-current={active === st.id ? 'step' : undefined}>
              {st.label}
            </a>
          </li>
        ))}
      </ol>
      <ul>
        {LINKS.map((l) => (
          <li key={l.href} className={l.href === '#journey' ? 'keep' : undefined}>
            <a href={l.href}>{l.label}</a>
          </li>
        ))}
        <li className="keep">
          <a href={REPO_URL}>GitHub ↗</a>
        </li>
      </ul>
      <div className="nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </nav>
  )
}
