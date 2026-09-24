import { describe, expect, it } from 'vitest'
import { subsequenceDTW } from '../lib/dtw'
import { alignments } from '../data/figures'

describe('subsequenceDTW (port of alignment/dtw.py)', () => {
  it('finds a zero-cost diagonal embedded in a larger candidate', () => {
    // query of 4 frames matches candidate columns 2..5 exactly
    const cost = Array.from({ length: 4 }, (_, i) => Array.from({ length: 8 }, (_, j) => (j === i + 2 ? 0 : 1)))
    const r = subsequenceDTW(cost)
    expect(r.path).toEqual([
      [0, 2],
      [1, 3],
      [2, 4],
      [3, 5],
    ])
    expect(r.normalizedCost).toBe(0)
  })

  it('has weighted path length N for the slope-constrained steps', () => {
    // with steps (1,1),(2,1),(1,2) and weights 1,2,1 every path's weight equals N (D-006)
    const n = 7
    const cost = Array.from({ length: n }, (_, i) => Array.from({ length: 12 }, (_, j) => Math.abs(Math.sin(i * 3.1 + j * 1.7))))
    const r = subsequenceDTW(cost)
    let total = 0
    let weight = 1
    total += cost[r.path[0][0]][r.path[0][1]]
    for (let k = 1; k < r.path.length; k++) {
      const [i, j] = r.path[k]
      const di = i - r.path[k - 1][0]
      const dj = j - r.path[k - 1][1]
      const w = di === 2 && dj === 1 ? 2 : 1
      weight += w
      total += w * cost[i][j]
      expect([
        [1, 1],
        [2, 1],
        [1, 2],
      ]).toContainEqual([di, dj])
    }
    expect(weight).toBe(n)
    expect(r.normalizedCost).toBeCloseTo(total / weight, 10)
  })

  it('reproduces the published paths from decoded matrices within a few frames', () => {
    for (const [key, a] of Object.entries(alignments)) {
      const ours = subsequenceDTW(a.cost).path
      const byRow = new Map<number, number[]>()
      for (const [i, j] of ours) byRow.set(i, [...(byRow.get(i) ?? []), j])
      // (2,1) steps skip query rows, so compare only rows the path visits
      const d = a.published_path
        .filter(([i]) => byRow.has(i))
        .map(([i, j]) => {
          const js = byRow.get(i)!
          return Math.abs(js.reduce((s, x) => s + x, 0) / js.length - j)
        })
      const mean = d.reduce((s, x) => s + x, 0) / d.length
      expect(mean, key).toBeLessThan(5)
    }
  })
})
