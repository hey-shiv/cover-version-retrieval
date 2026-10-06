/**
 * The CSS tokens again, for canvas code that cannot read custom properties.
 * Keep in step with styles/global.css (:root and .panel).
 */
export const DAY = {
  paper: '#e9edee',
  ink: '#0e2340',
  mute: '#56677b',
  faint: '#93a1b1',
  rule: '#c3ccd4',
  query: '#5243d6',
  cover: '#0a7f67',
  path: '#d6245f',
  hub: '#b77300',
}

export const NIGHT = {
  paper: '#081629',
  ink: '#e6eef6',
  mute: '#8ea3bd',
  faint: '#4d6587',
  rule: '#253c5d',
  query: '#a49bff',
  cover: '#45d0a8',
  path: '#ff5c8f',
  hub: '#f2b43c',
}

/** '#rrggbb' + alpha → 'rgba(…)' */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}
