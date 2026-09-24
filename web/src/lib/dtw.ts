/**
 * Subsequence DTW, ported line for line from the reference implementation in
 * src/cover_retrieval/alignment/dtw.py (`dtw(..., subsequence=True)`).
 *
 *   D[0, j] = C[0, j]                                  (query may start anywhere)
 *   D[i, j] = min_s  D[i - di_s, j - dj_s] + w_s * C[i, j]
 *   end     = argmin_j D[N - 1, j]
 *   normalised cost = D[end] / accumulated step weight along the path
 *
 * Project defaults (configs/base.yaml, D-006): steps (1,1), (2,1), (1,2) with weights
 * 1, 2, 1. Ties keep the earlier step, and the earliest end column wins, as in Python.
 */

export type Step = readonly [number, number]

export const SLOPE_STEPS: readonly Step[] = [
  [1, 1],
  [2, 1],
  [1, 2],
]
export const SLOPE_WEIGHTS: readonly number[] = [1, 2, 1]

export interface DTWResult {
  /** accumulated cost, row-major N x M (Infinity where unreachable) */
  acc: Float64Array
  n: number
  m: number
  /** [queryFrame, candidateFrame] from start to end */
  path: [number, number][]
  normalizedCost: number
}

export function subsequenceDTW(
  cost: number[][],
  steps: readonly Step[] = SLOPE_STEPS,
  weights: readonly number[] = SLOPE_WEIGHTS,
): DTWResult {
  const n = cost.length
  const m = cost[0].length
  const acc = new Float64Array(n * m).fill(Infinity)
  const accW = new Float64Array(n * m)
  const back = new Int8Array(n * m).fill(-1)
  for (let j = 0; j < m; j++) {
    acc[j] = cost[0][j]
    accW[j] = 1
  }
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let best = Infinity
      let bestStep = -1
      for (let s = 0; s < steps.length; s++) {
        const pi = i - steps[s][0]
        const pj = j - steps[s][1]
        if (pi < 0 || pj < 0) continue
        const prev = acc[pi * m + pj]
        if (!Number.isFinite(prev)) continue
        const cand = prev + weights[s] * cost[i][j]
        if (cand < best) {
          best = cand
          bestStep = s
        }
      }
      if (bestStep >= 0) {
        const k = i * m + j
        acc[k] = best
        back[k] = bestStep
        accW[k] = accW[(i - steps[bestStep][0]) * m + (j - steps[bestStep][1])] + weights[bestStep]
      }
    }
  }
  let endJ = 0
  for (let j = 1; j < m; j++) if (acc[(n - 1) * m + j] < acc[(n - 1) * m + endJ]) endJ = j
  const path: [number, number][] = []
  let i = n - 1
  let j = endJ
  path.push([i, j])
  while (i > 0) {
    const s = back[i * m + j]
    if (s < 0) break
    i -= steps[s][0]
    j -= steps[s][1]
    path.push([i, j])
  }
  path.reverse()
  const total = acc[(n - 1) * m + endJ]
  return { acc, n, m, path, normalizedCost: total / accW[(n - 1) * m + endJ] }
}
