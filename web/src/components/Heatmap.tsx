import { useEffect, useRef } from 'react'
import { lut, type RampName } from './ramp'

interface Props {
  /** rows x cols; row 0 is drawn at the bottom when `originLower` */
  data: number[][]
  ramp: RampName
  originLower?: boolean
  /** map a raw value to [0, 1] before colouring */
  transform?: (v: number) => number
  className?: string
  style?: React.CSSProperties
  ariaLabel: string
}

/**
 * Pixel-exact matrix renderer: one canvas pixel per cell, scaled up with
 * `image-rendering: pixelated` so every cell stays a crisp rectangle.
 */
export function Heatmap({ data, ramp, originLower = true, transform, className, style, ariaLabel }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const rows = data.length
    const cols = data[0].length
    canvas.width = cols
    canvas.height = rows
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    const table = lut(ramp)
    for (let r = 0; r < rows; r++) {
      const y = originLower ? rows - 1 - r : r
      for (let c = 0; c < cols; c++) {
        let v = data[r][c]
        if (transform) v = transform(v)
        const i = Math.max(0, Math.min(255, Math.round(v * 255))) * 4
        const o = (y * cols + c) * 4
        img.data[o] = table[i]
        img.data[o + 1] = table[i + 1]
        img.data[o + 2] = table[i + 2]
        img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [data, ramp, originLower, transform])
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      className={className}
      style={{ width: '100%', height: '100%', imageRendering: 'pixelated', ...style }}
    />
  )
}
