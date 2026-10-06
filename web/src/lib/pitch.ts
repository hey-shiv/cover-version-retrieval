/**
 * Twelve pitch-class hues. Hue follows the chromatic circle (C at 0°, +30° per
 * semitone), so transposing a recording by k semitones rotates every colour by
 * k × 30°. Used only where colour means "which pitch class".
 */
type RGB = [number, number, number]

/** OKLCH → linear sRGB → sRGB, clipped. Good enough for a fixed 12-entry table. */
function oklch(l: number, c: number, hDeg: number): RGB {
  const h = (hDeg * Math.PI) / 180
  const a = c * Math.cos(h)
  const b = c * Math.sin(h)
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b
  const s_ = l - 0.0894841775 * a - 1.291485548 * b
  const L = l_ ** 3
  const M = m_ ** 3
  const S = s_ ** 3
  const lin = [
    4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S,
    -1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S,
    -0.0041960863 * L - 0.7034186147 * M + 1.707614701 * S,
  ]
  return lin.map((v) => {
    const x = Math.max(0, Math.min(1, v))
    return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055
  }) as RGB
}

// start the wheel at a cool blue for C so the default key sits inside the site's palette
const HUE0 = 250

/** 0..1 RGB per pitch class, for the fluid. */
export const PC_RGB: RGB[] = Array.from({ length: 12 }, (_, p) => oklch(0.74, 0.15, HUE0 + p * 30))

/** CSS colour per pitch class. */
export const PC_CSS: string[] = PC_RGB.map(([r, g, b]) => `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`)

/** RGB at an arbitrary angle on the wheel (radians, 0 = C at the top, clockwise). */
export function wheelRGB(angle: number): RGB {
  const turns = (((angle / (Math.PI * 2)) % 1) + 1) % 1
  return oklch(0.74, 0.15, HUE0 + turns * 360)
}
