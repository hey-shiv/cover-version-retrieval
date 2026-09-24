export const fmt = (x: number, d = 3) => x.toFixed(d)
export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`
export const int = (x: number) => Math.round(x).toLocaleString('en-US')
export const ratio = (a: number, b: number, d = 1) => `${(a / b).toFixed(d)}×`

export const clamp = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x))
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t
/** smoothstep between edges */
export const ease = (t: number) => {
  const x = clamp(t)
  return x * x * (3 - 2 * x)
}
/** progress of t within [a, b], clamped */
export const phase = (t: number, a: number, b: number) => clamp((t - a) / (b - a))

export function logScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain.map(Math.log10)
  return (v: number) => range[0] + ((Math.log10(v) - d0) / (d1 - d0)) * (range[1] - range[0])
}
export function linScale(domain: [number, number], range: [number, number]) {
  return (v: number) => range[0] + ((v - domain[0]) / (domain[1] - domain[0])) * (range[1] - range[0])
}
