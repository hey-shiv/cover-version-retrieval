import { useEffect, useState } from 'react'

const LINKS = [
  { href: '#problem', label: 'Research' },
  { href: '#encoder', label: 'System' },
  { href: '#journey', label: 'Experiments' },
  { href: '#results', label: 'Results' },
]
export const REPO_URL = 'https://github.com/hey-shiv/cover-version-retrieval'

export function Nav() {
  const [progress, setProgress] = useState(0)
  const [onDark, setOnDark] = useState(true)
  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const h = document.documentElement.scrollHeight - window.innerHeight
      setProgress(h > 0 ? window.scrollY / h : 0)
      // read the background under the bar: dark hero / panels vs paper
      const under = document.elementsFromPoint(window.innerWidth / 2, 20).find((e) => !e.closest('.nav'))
      setOnDark(!!under?.closest('.panel'))
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
      <ul>
        {LINKS.map((l) => (
          <li key={l.href}>
            <a href={l.href}>{l.label}</a>
          </li>
        ))}
        <li>
          <a href={REPO_URL}>GitHub ↗</a>
        </li>
      </ul>
      <div className="nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </nav>
  )
}
