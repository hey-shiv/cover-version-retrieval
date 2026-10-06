/**
 * Sequential ramps, both printed like a cyanotype.
 * "ink" reads on paper (0 = paper, 1 = Prussian ink); "ember" reads on night panels
 * (0 = night, 1 = paper-white). The name "ember" is kept so call sites stay stable.
 */
type RGB = [number, number, number]
const hex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB

const STOPS: Record<'ink' | 'ember', RGB[]> = {
  ink: ['#eef2f3', '#c4d3df', '#7f9bb8', '#36557e', '#0e2340'].map(hex),
  ember: ['#081629', '#11305a', '#2a5f98', '#7fb2d9', '#eef7fb'].map(hex),
}

export type RampName = keyof typeof STOPS

/** 256-entry lookup table, packed as RGBA bytes for ImageData. */
const LUTS = new Map<RampName, Uint8ClampedArray>()
export function lut(name: RampName): Uint8ClampedArray {
  let t = LUTS.get(name)
  if (t) return t
  t = new Uint8ClampedArray(256 * 4)
  const stops = STOPS[name]
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1)
    const k = Math.min(stops.length - 2, Math.floor(x))
    const f = x - k
    for (let c = 0; c < 3; c++) t[i * 4 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f
    t[i * 4 + 3] = 255
  }
  LUTS.set(name, t)
  return t
}

export function rampCss(name: RampName, v: number): string {
  const t = lut(name)
  const i = Math.max(0, Math.min(255, Math.round(v * 255))) * 4
  return `rgb(${t[i]},${t[i + 1]},${t[i + 2]})`
}
