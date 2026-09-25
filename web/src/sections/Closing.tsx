import { useState } from 'react'
import { Chapter, Caption } from '../components/Chapter'
import { DATASET, FUTURE, REPRO, ROLES, SEED, TEST_COUNT } from '../data/project'
import { benchmarkRuns } from '../data/benchmark-data'

export function Reproducibility() {
  const [open, setOpen] = useState<string>('SEED')
  const active = REPRO.find((r) => r.key === open)!
  return (
    <Chapter id="reproducibility" no="12 · Method" title={<>Every number <em>can be rebuilt</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">
            A result you can't regenerate is an anecdote. The pipeline is deterministic from committed configs and ID-only manifests, and tested by{' '}
            <strong>{TEST_COUNT} tests</strong>.
          </p>
          <p>
            From a fresh clone with the same data, the manifests rebuild byte-identically, the classical rankings match exactly, and retraining the encoder
            reproduces the selected checkpoint bit for bit.
          </p>
        </div>
        <aside className="aside">
          <span className="mono">pytest</span> · {TEST_COUNT} passed, including a synthetic end-to-end pipeline that exercises every script. Its numbers go
          to <span className="mono">runs/smoke/</span>, never to <span className="mono">reports/</span>.
        </aside>
      </div>
      <div className="figure chain">
        <ol className="chain-list" role="tablist" aria-label="Reproducibility controls">
          {REPRO.map((r, i) => (
            <li key={r.key}>
              <button type="button" role="tab" aria-selected={r.key === open} onClick={() => setOpen(r.key)} className="chain-item">
                <span className="chain-key">{r.key}</span>
                <span className="chain-val">{r.value}</span>
              </button>
              {i < REPRO.length - 1 && (
                <span className="chain-arrow" aria-hidden="true">
                  →
                </span>
              )}
            </li>
          ))}
        </ol>
        <div className="chain-detail" role="tabpanel" aria-live="polite">
          <p>{active.body}</p>
          <div className="source">{active.where}</div>
        </div>
      </div>
      <div className="figure scroll-x">
        <table className="data roles">
          <thead>
            <tr>
              <th>Role (Cover Analysis, WID-disjoint)</th>
              <th className="n">Works</th>
              <th className="n">Recordings</th>
              <th>Used for</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['calibration', 'the only data that chooses α, λ and resolution'],
                ['query', '20 development queries'],
                ['distractor', 'development negatives'],
                ['validation', 'encoder checkpoint selection'],
                ['train', 'encoder training'],
              ] as const
            ).map(([role, use]) => (
              <tr key={role}>
                <td className="mono">{role}</td>
                <td className="n">{ROLES[role].wids.toLocaleString('en-US')}</td>
                <td className="n">{ROLES[role].tracks.toLocaleString('en-US')}</td>
                <td>{use}</td>
              </tr>
            ))}
            <tr>
              <td className="mono">benchmark</td>
              <td className="n">3,000</td>
              <td className="n">15,000</td>
              <td>final test only · never trained, tuned or calibrated on</td>
            </tr>
          </tbody>
        </table>
      </div>
      <Caption label="Table 1" source={['data/manifests/manifest_info.json', 'DATASET_CARD.md']}>
        Split by work, along a permutation seeded with {SEED}. The two Da-TACOS subsets share no work and no recording, so training cannot leak into
        the benchmark.
      </Caption>
    </Chapter>
  )
}

export function DataLicense() {
  return (
    <Chapter id="data" no="13 · Data" title={<>No audio, <em>by design</em></>}>
      <div className="split">
        <div className="body prose">
          <p>
            This project uses <strong>{DATASET.name}</strong> (Yesiler et al., ISMIR 2019): metadata for the work and performance identifiers, and HPCP
            features pre-extracted by its authors. {DATASET.name} contains no audio. The features, about {DATASET.archivesGB} GB of archives, are
            licensed <strong>CC BY-NC-SA 4.0</strong> and are never committed to the repository, never used to train a published model, and not
            distributed by this site.
          </p>
          <p>
            So nothing on this page plays sound, and it cannot take your audio: the site is a static reading of precomputed results. Wherever a figure
            shows feature-like data, it was decoded from a plot already committed to the repository (colour-scale positions, not feature values) and says
            so in its caption. Trained checkpoints are derived from NonCommercial ShareAlike data and are not released.
          </p>
        </div>
        <aside className="aside">
          {DATASET.citation}
          <br />
          <br />
          Licence: {DATASET.license}.<br />
          <a href={DATASET.zenodo}>Zenodo record</a> · <a href={DATASET.homepage}>project page</a>
        </aside>
      </div>
    </Chapter>
  )
}

export function Limits() {
  const r4 = benchmarkRuns[3]
  const items = [
    ['Precomputed only', 'Every visual here comes from committed result files. There is no model running, no search box, and no audio upload.'],
    ['Below published systems', 'The best locked system (0.136 MAP) and even the best research run (0.222: fused Stage 1, K = 500) are well below Qmax on the same HPCP input (0.333), and far below learned systems like MOVE or ByteCover.'],
    ['Stage 1 is still the constraint', `Shortlist recall at K = 30 is ${r4.shortlist_recall.toFixed(3)} in the final run. The hybrid cannot retrieve what the shortlist omits. Larger K helps, but beyond K ≈ 100 the reranker, not the shortlist, loses most queries.`],
    ['The strongest likely system was never run', 'Classical alignment at 384 frames with hubness correction over all 1.95 × 10⁸ pairs: about 42 h of CPU.'],
    ['Whole-query alignment', 'Subsequence DTW aligns the entire query in order. Covers that drop, add or reorder sections are penalised; local alignment is not implemented.'],
    ['A small development protocol', '20 queries against 120 candidates. It reversed two conclusions at benchmark scale and hid the value of key rotation. Its numbers are plumbing checks and hypotheses.'],
    ['Dataset scope', 'Feature-only, from 2019, Western-pop-centric. Nothing here establishes robustness to short, live, noisy or partial queries, or to production-scale catalogues.'],
  ]
  return (
    <Chapter id="limits" no="14 · Limits" title={<>What remains <em>unsolved</em></>}>
      <dl className="limits">
        {items.map(([k, v]) => (
          <div key={k} className="limit">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <Future />
    </Chapter>
  )
}

function Future() {
  const lanes = ['Representation', 'Stage 1', 'Stage 2'] as const
  return (
    <div className="figure future">
      <h3 className="sub">Next, in the order the evidence asks for</h3>
      <div className="future-map">
        {lanes.map((lane) => (
          <div key={lane} className="lane">
            <div className="label lane-name">{lane}</div>
            {FUTURE.map((f, i) =>
              f.target === lane ? (
                <div key={f.id} className="future-item">
                  <span className="num future-no">{i + 1}</span>
                  <div>
                    <div className="future-title">{f.title}</div>
                    <p>{f.body}</p>
                  </div>
                </div>
              ) : null,
            )}
          </div>
        ))}
      </div>
      <Caption label="Map" source="README.md → What's next">
        Only directions the repository itself documents, numbered in its priority order and placed on the part of the pipeline they would change. None
        of them has been run yet.
      </Caption>
    </div>
  )
}
