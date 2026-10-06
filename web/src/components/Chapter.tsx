import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'

export function Chapter({
  id,
  no,
  title,
  dark,
  children,
}: {
  id: string
  /** "03 · Stage 1": the chapter number, then a short name for the stage */
  no: string
  title: ReactNode
  dark?: boolean
  children: ReactNode
}) {
  const [num, kicker] = no.split(' · ')
  return (
    <section id={id} className={`chapter${dark ? ' panel' : ''}`} aria-labelledby={`${id}-title`}>
      <div className="wrap">
        <header className="chapter-head">
          <div className="chapter-no" aria-hidden="true">
            {num}
          </div>
          <div>
            {kicker && <p className="chapter-kicker">{kicker}</p>}
            <h2 id={`${id}-title`}>{title}</h2>
          </div>
        </header>
        {children}
      </div>
    </section>
  )
}

export function Caption({ label, children, source }: { label: string; children: ReactNode; source?: string | string[] }) {
  const sources = source === undefined ? [] : Array.isArray(source) ? source : [source]
  return (
    <div className="caption">
      <b>{label}</b>
      <div>
        {children}
        {sources.map((s) => (
          <div className="source" key={s}>
            {s}
          </div>
        ))}
      </div>
    </div>
  )
}

/** Segmented control. A thumb slides to the pressed option, so the change reads as one movement. */
export function Seg<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: ReactNode }[]
  value: T
  onChange: (v: T) => void
  label: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const [thumb, setThumb] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const measure = () => {
      const b = el.querySelector<HTMLButtonElement>('button[aria-pressed="true"]')
      if (b) setThumb({ x: b.offsetLeft, y: b.offsetTop, w: b.offsetWidth, h: b.offsetHeight })
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [value, options.length])
  return (
    <div ref={box} className="seg" role="group" aria-label={label} data-ready={thumb ? '' : undefined}>
      {thumb && (
        <span
          className="seg-thumb"
          aria-hidden="true"
          style={{ width: thumb.w, height: thumb.h, transform: `translate(${thumb.x}px, ${thumb.y}px)`, left: 0, top: 0 }}
        />
      )}
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
