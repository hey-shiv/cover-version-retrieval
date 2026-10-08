import type { ReactNode } from 'react'

export function Chapter({
  id,
  no,
  title,
  dark,
  children,
}: {
  id: string
  no: string
  title: ReactNode
  dark?: boolean
  children: ReactNode
}) {
  return (
    <section id={id} className={`chapter${dark ? ' panel' : ''}`} aria-labelledby={`${id}-title`}>
      <div className="wrap">
        <header className="chapter-head">
          <div className="chapter-no">{no}</div>
          <h2 id={`${id}-title`}>{title}</h2>
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
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
