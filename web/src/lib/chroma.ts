/**
 * Pitch-class utilities, mirroring src/cover_retrieval/alignment/transposition.py.
 * Matrices are (12, T): values[p][t].
 */

export const PITCH_CLASSES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'] as const

/** rotate(X, k)[p] = X[(p - k) mod 12] — moves every bin up by k semitones (np.roll). */
export function rotate<T>(rows: T[], k: number): T[] {
  const s = ((k % 12) + 12) % 12
  return rows.map((_, p) => rows[(p - s + 12) % 12])
}

/** Time-averaged, frame-L2-normalised, L2-normalised profile (chroma_profile). */
export function profile(values: number[][]): number[] {
  const T = values[0].length
  const out = new Array(12).fill(0)
  for (let t = 0; t < T; t++) {
    let norm = 0
    for (let p = 0; p < 12; p++) norm += values[p][t] ** 2
    norm = Math.sqrt(norm) || 1
    for (let p = 0; p < 12; p++) out[p] += values[p][t] / norm / T
  }
  const n = Math.hypot(...out) || 1
  return out.map((v) => v / n)
}

export function dot(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0)
}

/** Cosine between query profile and candidate profile rotated by 0..11. */
export function rotationScores(query: number[][], candidate: number[][]): number[] {
  const q = profile(query)
  const c = profile(candidate)
  return Array.from({ length: 12 }, (_, k) => dot(q, rotate(c, k)))
}

/** Cosine-distance cross-similarity matrix between two (12, T) sequences. */
export function costMatrix(query: number[][], candidate: number[][]): number[][] {
  const col = (x: number[][], t: number) => {
    const v = x.map((row) => row[t])
    const n = Math.hypot(...v) || 1
    return v.map((e) => e / n)
  }
  const Q = Array.from({ length: query[0].length }, (_, t) => col(query, t))
  const C = Array.from({ length: candidate[0].length }, (_, t) => col(candidate, t))
  return Q.map((q) => C.map((c) => 1 - dot(q, c)))
}
