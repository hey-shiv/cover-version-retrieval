import { useMemo, useState } from 'react'
import { Chapter, Caption } from '../components/Chapter'
import { Heatmap } from '../components/Heatmap'
import { hpcp } from '../data/figures'
import { rotate } from '../lib/chroma'

type Change = 'key' | 'tempo' | 'timbre' | 'arrangement' | 'structure'

const BASE = 'P_242247'

/** Deterministic illustrative transforms applied to one real (decoded) chroma strip. */
function transform(x: number[][], change: Change | null): number[][] {
  const T = x[0].length
  switch (change) {
    case 'key':
      return rotate(x, 3)
    case 'tempo': {
      const n = Math.round(T * 0.72)
      return x.map((row) =>
        Array.from({ length: n }, (_, t) => {
          const s = (t * (T - 1)) / (n - 1)
          const a = Math.floor(s)
          const f = s - a
          return row[a] * (1 - f) + row[Math.min(T - 1, a + 1)] * f
        }),
      )
    }
    case 'arrangement': {
      // re-voicing: smear energy into the neighbouring fifth and blur in time
      return x.map((row, p) =>
        row.map((v, t) => {
          const prev = row[Math.max(0, t - 1)]
          const next = row[Math.min(T - 1, t + 1)]
          const fifth = x[(p + 7) % 12][t]
          const wobble = 0.5 + 0.5 * Math.sin(t * 1.7 + p * 2.3)
          return Math.min(1, 0.45 * v + 0.2 * (prev + next) + 0.25 * fifth * wobble)
        }),
      )
    }
    case 'structure':
      // drop frames 24-40, repeat 60-76 at the end: a cut verse and an extra chorus
      return x.map((row) => [...row.slice(0, 24), ...row.slice(40), ...row.slice(60, 76)])
    default:
      return x
  }
}

const CHANGES: { id: Change; name: string; what: string; cope: string; verdict: 'handled' | 'partly' | 'not modelled' }[] = [
  {
    id: 'key',
    name: 'Key',
    what: 'Transposed three semitones up: every pitch-class row moves cyclically.',
    cope: 'Try all 12 cyclic rotations of the candidate; keep the one whose mean chroma profile best matches the query (D-004).',
    verdict: 'handled',
  },
  {
    id: 'tempo',
    name: 'Tempo',
    what: 'Played 28% faster: the same events arrive in fewer frames.',
    cope: 'Every track is resampled to a fixed number of frames, then subsequence DTW absorbs local rate changes with steps up to 2:1 (D-005, D-006).',
    verdict: 'handled',
  },
  {
    id: 'timbre',
    name: 'Instruments · voice',
    what: 'A piano instead of a guitar, a different singer. The strip does not change.',
    cope: 'HPCP folds all energy onto 12 pitch classes, so timbre is largely gone before the model sees anything (D-002). Melody and voicing changes still leak through.',
    verdict: 'partly',
  },
  {
    id: 'arrangement',
    name: 'Arrangement',
    what: 'Re-voiced chords, added harmony: energy spreads to neighbouring pitch classes.',
    cope: 'Nothing targets it. It raises the frame-to-frame cost of a true cover, and static, blurry material starts to look similar to everything (hubness).',
    verdict: 'not modelled',
  },
  {
    id: 'structure',
    name: 'Structure',
    what: 'A verse cut, a chorus repeated at the end.',
    cope: 'Subsequence DTW aligns the whole query in order, so dropped or reordered sections are forced through unrelated material. Local alignment (Qmax) is untested future work.',
    verdict: 'not modelled',
  },
]

export function Problem() {
  const [change, setChange] = useState<Change>('key')
  const base = hpcp[BASE].values
  const changed = useMemo(() => transform(base, change), [base, change])
  const active = CHANGES.find((c) => c.id === change)!

  return (
    <Chapter id="problem" no="01 · The question" title={<>What survives <em>the change?</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            A cover is another recording of the same composition. Almost everything you would measure about the audio can change.
          </p>
          <p>
            The task is <strong>retrieval</strong>: given one recording, rank a catalogue so that other versions of the same work come first. Relevance
            is defined only by the SecondHandSongs <em>work</em> ID (<span className="pid">W_…</span>); every recording has a performance ID (
            <span className="pid">P_…</span>). No titles, no artists, no key or tempo annotations are ever used.
          </p>
        </div>
        <aside className="aside">
          Cover song identification is hard because the invariances are musical, not acoustic: two versions can share almost no spectral content and
          still be, unmistakably, the same song.
        </aside>
      </div>

      <div className="figure change-lab">
        <div className="change-list" role="tablist" aria-label="Transformations">
          {CHANGES.map((c) => (
            <button
              key={c.id}
              role="tab"
              type="button"
              aria-selected={c.id === change}
              className="change-item"
              onClick={() => setChange(c.id)}
            >
              <span className="change-name">{c.name}</span>
              <span className={`change-verdict v-${c.verdict.replace(' ', '-')}`}>{c.verdict}</span>
            </button>
          ))}
        </div>
        <div className="change-view" role="tabpanel" aria-live="polite">
          <div className="change-strips">
            <div className="label">Original · {BASE}</div>
            <div className="strip" style={{ aspectRatio: '96 / 12', width: '100%' }}>
              <Heatmap data={base} ramp="ink" ariaLabel={`Chroma of ${BASE}`} />
            </div>
            <div className="label" style={{ marginTop: 18 }}>
              After: {active.name.toLowerCase()} · {changed[0].length} frames
            </div>
            <div className="strip" style={{ aspectRatio: `${changed[0].length} / 12`, width: `${(changed[0].length / 96) * 100}%` }}>
              <Heatmap data={changed} ramp="ink" ariaLabel={`Chroma after ${active.name} change`} />
            </div>
          </div>
          <dl className="change-text">
            <dt className="label">What changes</dt>
            <dd>{active.what}</dd>
            <dt className="label">What the system does</dt>
            <dd>{active.cope}</dd>
          </dl>
        </div>
      </div>
      <Caption label="Figure 1" source="HPCP of P_242247 decoded from reports/figures/calibration_cover_pairs_hpcp.png">
        One real chroma strip (12 pitch classes, C at the bottom, × 96 frames) and a <em>deterministic, illustrative</em> version of each kind of
        change. The transforms are drawn by this page to show the idea. They are not audio and not data from any real cover.
      </Caption>
    </Chapter>
  )
}
