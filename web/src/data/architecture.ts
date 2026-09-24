/**
 * Stage-1 encoder: src/cover_retrieval/models/tcn_encoder.py with configs/base.yaml
 * (encoder block) — unchanged in configs/full_train.yaml. Parameter counts below are
 * computed from the layer shapes and checked against `n_parameters` = 645,504 and
 * `receptive_field_frames` = 253 in reports/results/training_summary_encoder_full_long.json
 * (see architecture.test.ts).
 */
import { results } from './results'

export const ENCODER = {
  inputFrames: 256, // encoder.input_frames (area-pooled from a 512-frame cache)
  cacheFrames: 512,
  channels: 128,
  nBlocks: 6,
  kernel: 3,
  embeddingDim: 128,
  dropout: 0.1,
} as const

export const TRAINING = {
  loss: 'SupCon',
  temperature: 0.1,
  batchWorks: 32, // 32 works x 2 recordings
  lr: 0.001,
  weightDecay: 0.0001,
  cropMinFraction: 0.6,
  pitchShiftAugment: true,
  seed: 20260817,
  device: 'cpu (deterministic)',
} as const

export const dilations = Array.from({ length: ENCODER.nBlocks }, (_, b) => 2 ** b)

/** receptive field after block b (1-based): 1 + 2 (k - 1) (2^b - 1) */
export const receptiveField = (b: number) => 1 + 2 * (ENCODER.kernel - 1) * (2 ** b - 1)

const conv = (cin: number, cout: number, k: number) => cin * cout * k + cout
const bn = (c: number) => 2 * c
const linear = (i: number, o: number) => i * o + o

export const PARAMS = (() => {
  const C = ENCODER.channels
  const input = conv(12, C, 1)
  const block = 2 * conv(C, C, ENCODER.kernel) + 2 * bn(C)
  const head = linear(2 * C, C) + linear(C, ENCODER.embeddingDim)
  return { input, block, blocks: block * ENCODER.nBlocks, head, total: input + block * ENCODER.nBlocks + head }
})()

export const trainingRuns = results.training.runs
export const encoderSweep = results.training.sweep
