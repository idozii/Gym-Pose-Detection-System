/** M41: development-only paired 200ms vs 400ms pose observation audit.
 *
 * Exactly one chronological dense (~200ms) MediaPipe trace per video is
 * evaluated at full density and every-other-index temporal subsampling.
 * This isolates availability of extra pose observations *conditional on
 * the same dense MediaPipe tracking context*. It does NOT prove the effect
 * of running a second independent 400ms pose estimator.
 *
 * M34 keeps its own unmodified >=400ms sampling gate. The low-to-high
 * detector receives every observation in each compared trace. Never
 * interpret M34's 200ms condition as lowering its internal sample gate.
 */
import { compareTrainTrace } from './m39_compare_train_trace.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PROTOCOL = 'm41-train-cadence-sensitivity-v1';

export function pairedCadenceComparison(denseTrace) {
  if (!denseTrace || !Array.isArray(denseTrace.frames) ||
      denseTrace.frames.length < 16) {
    throw new Error('M41 needs at least 16 chronological dense pose samples');
  }
  const sparseFrames = denseTrace.frames.filter((_, i) => i % 2 === 0);
  const dense = compareTrainTrace(denseTrace);
  const sparse = compareTrainTrace({...denseTrace, frames: sparseFrames});
  if (dense.annotation_rep_count !== sparse.annotation_rep_count) {
    throw new Error('M41 same-video annotated counts disagree');
  }
  return {
    protocol: PROTOCOL,
    source_split: 'train',
    annotated_rep_count: dense.annotation_rep_count,
    dense_200ms: dense,
    subsample_400ms: sparse,
    design: {
      nominal_dense_ms: 200, downsample_factor: 2,
      shared_python_pose_tracking_context: true,
      m34_internal_sample_gate_unchanged_ms: 400,
      exact_independent_400ms_pose_estimator_parity: false
    },
    limitation: 'TRAIN-only descriptive cadence sensitivity. No raw poses, source IDs or event times saved. 400ms subsample is not a separately tracked MediaPipe 400ms stream.'
  };
}

function main() {
  const [, , tracePath, outputPath] = process.argv;
  if (!tracePath || !outputPath) {
    throw new Error('Usage: node web/scripts/m41_compare_train_cadence.js <dense_trace.json> <output.json>');
  }
  const source = JSON.parse(readFileSync(resolve(tracePath), 'utf8'));
  const report = pairedCadenceComparison(source);
  writeFileSync(resolve(outputPath), JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
  console.log('M41 train-only paired cadence comparison completed');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
