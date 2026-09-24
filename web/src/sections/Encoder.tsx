import { useState } from 'react'
import { Chapter, Caption } from '../components/Chapter'
import { ENCODER, PARAMS, TRAINING, dilations, receptiveField, trainingRuns } from '../data/architecture'
import { useWidth } from '../lib/hooks'
import { linScale } from '../lib/format'

const STAGES = [
  { id: 'in', name: 'Input', dims: '12 × 256', note: 'Area-pooled from a 512-frame cache and frame-normalised. Random crops (≥ 60% of the track) and random pitch rotations during training.' },
  { id: 'proj', name: '1×1 conv', dims: '12 → 128', note: 'Lifts each frame from 12 pitch classes to 128 channels.' },
  ...dilations.map((d, b) => ({
    id: `b${b + 1}`,
    name: `Block ${b + 1}`,
    dims: `dilation ${d}`,
    note: `Two dilated convolutions (kernel 3, dilation ${d}) with batch norm, GELU, dropout 0.1 and a residual skip. After this block one output frame sees ${receptiveField(b + 1)} input frames.`,
  })),
  { id: 'pool', name: 'Pool', dims: 'mean ‖ max → 256', note: 'Collapse time: the average and the maximum of every channel over the whole track, concatenated. This is where the order of events is thrown away.' },
  { id: 'head', name: 'Projection', dims: '256 → 128 → 128', note: 'Linear, GELU, linear.' },
  { id: 'out', name: 'L2', dims: '128-d unit vector', note: 'One point on a 128-dimensional sphere per recording. Retrieval is exact cosine similarity between these points.' },
]

export function Encoder() {
  const [sel, setSel] = useState('b4')
  const block = sel.startsWith('b') ? Number(sel.slice(1)) : sel === 'in' || sel === 'proj' ? 0 : 6
  const stage = STAGES.find((s) => s.id === sel)!

  return (
    <Chapter id="encoder" no="03 · Stage 1" title={<>One vector <em>per recording</em></>}>
      <div className="split">
        <div className="body prose">
          <p className="lede">Stage 1 compresses an entire recording into a single 128-dimensional vector, so that a catalogue can be searched in milliseconds.</p>
          <p>
            The encoder is a small temporal convolutional network: <strong>{PARAMS.total.toLocaleString('en-US')} parameters</strong>, six residual
            blocks whose dilation doubles each time. Stacking dilations lets the last block see almost the whole 256-frame input, and a covering pair
            should land close together on the sphere. It is trained with supervised contrastive loss (τ = {TRAINING.temperature}) on pairs of
            recordings of the same work, deterministically on CPU.
          </p>
        </div>
        <aside className="aside">
          Parameters: input {PARAMS.input.toLocaleString('en-US')} · each block {PARAMS.block.toLocaleString('en-US')} · head{' '}
          {PARAMS.head.toLocaleString('en-US')}. Computed from the layer shapes and checked against <span className="mono">n_parameters</span> in the
          training summary.
        </aside>
      </div>

      <div className="figure encoder-fig">
        <ol className="stack" aria-label="Encoder layers">
          {STAGES.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                className="stack-item"
                aria-pressed={s.id === sel}
                onClick={() => setSel(s.id)}
                onMouseEnter={() => setSel(s.id)}
                onFocus={() => setSel(s.id)}
              >
                <span className="stack-name">{s.name}</span>
                <span className="stack-dims mono">{s.dims}</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="encoder-right">
          <ReceptiveField block={block} />
          <p className="encoder-note" aria-live="polite">
            <span className="label">{stage.name}</span>
            <br />
            {stage.note}
          </p>
        </div>
      </div>
      <Caption label="Figure 3" source="src/cover_retrieval/models/tcn_encoder.py · configs/base.yaml (encoder)">
        Receptive field of a single output frame after each block, over the real 256-frame input: 1 + 2(k − 1)(2<sup>b</sup> − 1) frames, reaching{' '}
        {receptiveField(6)} after block 6. Select a layer to see what it does.
      </Caption>

      <TrainingCurves />
    </Chapter>
  )
}

function ReceptiveField({ block }: { block: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(700)
  const H = 250
  const padL = 64
  const x = linScale([0, ENCODER.inputFrames], [padL, width - 40])
  const rowY = (b: number) => H - 26 - b * 34
  const centre = 128
  return (
    <div ref={ref} className="rf">
      <svg width={width} height={H} role="img" aria-label={`Receptive field after block ${block}: ${receptiveField(block)} of 256 input frames`}>
        {/* input frames */}
        {Array.from({ length: ENCODER.inputFrames }, (_, t) => {
          const rf = receptiveField(block)
          const inside = Math.abs(t - centre) <= (rf - 1) / 2
          return <rect key={t} x={x(t)} y={rowY(0) - 3} width={Math.max(0.6, x(1) - x(0) - 0.4)} height={14} fill={inside ? 'var(--path)' : 'var(--paper-3)'} />
        })}
        <text x={0} y={rowY(0) + 8} className="rf-t">
          input
        </text>
        {Array.from({ length: 6 }, (_, i) => {
          const b = i + 1
          const rf = receptiveField(b)
          const on = b <= block
          const y = rowY(b)
          const half = (rf - 1) / 2
          return (
            <g key={b} opacity={on ? 1 : 0.28}>
              <text x={0} y={y + 4} className="rf-t">
                block {b}
              </text>
              <line x1={x(0)} x2={x(ENCODER.inputFrames)} y1={y} y2={y} stroke="var(--rule)" />
              {on && b === block && (
                <path
                  d={`M ${x(centre - half)} ${rowY(0) - 4} L ${x(centre + half + 1)} ${rowY(0) - 4} L ${x(centre + 1)} ${y} L ${x(centre)} ${y} Z`}
                  fill="var(--path)"
                  opacity={0.12}
                />
              )}
              <rect x={x(centre - half)} y={y - 3} width={x(rf) - x(0)} height={6} fill={on ? 'var(--ink)' : 'var(--faint)'} />
              <text x={Math.min(x(centre + half + 1) + 6, width - 70)} y={y + 4} className="rf-n">
                {rf}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function TrainingCurves() {
  const [ref, width] = useWidth<HTMLDivElement>(700)
  const runs = [
    { key: 'sweep_base', label: '1,500 works', color: 'var(--faint)' },
    { key: 'encoder_full_train', label: '4,780 works · 60 ep', color: 'var(--mute)' },
    { key: 'encoder_full_long', label: '4,780 works · 150 ep', color: 'var(--ink)' },
  ] as const
  const H = 220
  const x = linScale([1, 150], [44, width - (width < 640 ? 130 : 190)])
  const y = linScale([0, 0.45], [H - 26, 10])
  return (
    <div className="figure">
      <div className="split">
        <div className="body prose">
          <p>
            Data mattered more than anything else. With 1,500 training works (a bandwidth limit, not a choice) the best validation MAP was 0.181. With
            all 4,780 it rose to 0.319, and training for 150 epochs instead of 60 took it to{' '}
            <strong>{trainingRuns.encoder_full_long.summary.best_val_map.toFixed(3)}</strong>. It was still drifting upward when training stopped.
          </p>
        </div>
      </div>
      <div ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Validation MAP by epoch for three encoder training runs">
          {[0, 0.1, 0.2, 0.3, 0.4].map((v) => (
            <g key={v}>
              <line x1={44} x2={width - 150} y1={y(v)} y2={y(v)} stroke="var(--rule)" strokeDasharray={v ? '2 3' : undefined} />
              <text x={38} y={y(v) + 4} textAnchor="end" className="axis-t">
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          {[1, 30, 60, 90, 120, 150].map((e) => (
            <text key={e} x={x(e)} y={H - 6} textAnchor="middle" className="axis-t">
              {e === 150 ? '150 ep' : e}
            </text>
          ))}
          {runs.map((r) => {
            const pts = trainingRuns[r.key].epochs
            const d = pts.map(([e, , m], i) => `${i ? 'L' : 'M'}${x(e).toFixed(1)},${y(m).toFixed(1)}`).join(' ')
            const last = pts[pts.length - 1]
            const best = trainingRuns[r.key].summary
            return (
              <g key={r.key}>
                <path d={d} fill="none" stroke={r.color} strokeWidth={r.key === 'encoder_full_long' ? 1.8 : 1.2} />
                <circle cx={x(best.best_epoch)} cy={y(best.best_val_map)} r={3} fill={r.color} />
                <text x={x(last[0]) + 6} y={y(last[2]) + 4} className="axis-t" fill={r.color}>
                  {width < 640 ? r.label.replace(' works', '') : r.label}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <Caption
        label="Figure 4"
        source={['reports/results/training_history_sweep_base.csv', 'reports/results/training_history_encoder_full_train.csv', 'reports/results/training_history_encoder_full_long.csv']}
      >
        Validation MAP per epoch on 150 held-out works (300 recordings, each queried against the other 299). Dots mark the checkpoint each run kept.
        Validation works select the checkpoint and nothing else.
      </Caption>
    </div>
  )
}
