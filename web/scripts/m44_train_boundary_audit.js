/** M44: TRAIN-only, post-hoc angle and annotation-boundary audit.
 *
 * Uses frozen M43 event and M41 baseline methods, with no changes to the
 * detection thresholds or source event matching. Diagnostic bins are
 * hypothesis-generating evidence, NOT trained features, recovered events,
 * causal claims, squat-quality judgments or independent evaluation.
 */
import { measureCycleAngle, REP_POLICIES } from '../src/rep_counter.js';
import { detectLowToHighEvents } from './m39_compare_train_trace.js';
import {
  analyzeTrainTrace, detailedMatch, gtEvents
} from './m43_analyse_train_errors.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PROTOCOL = 'm44-train-boundary-angle-observability-v1';
// Predeclared visual diagnostic context: original M37 minimum event tolerance.
export const CONTEXT_MS = 750;
export const ANGLE_BIN_KEYS = Object.freeze([
  'min_gt_angle_gt120_to135',
  'min_gt_angle_gt135_to150',
  'min_gt_angle_gt150_to165',
  'min_gt_angle_above165'
]);
export const CONTEXT_KEYS = Object.freeze([
  'low_before_only', 'low_after_only', 'low_both_sides',
  'low_neither_side'
]);
export const FP_LOCATION_KEYS = Object.freeze([
  'unmatched_inside_any_gt_interval',
  'unmatched_outside_all_gt_intervals'
]);

function empty(keys) {
  return Object.fromEntries(keys.map(k => [k, 0]));
}

function minAngleBucket(value) {
  if (!Number.isFinite(value) || value <= REP_POLICIES.squat.lowDeg) {
    throw new Error('M44 no-low minimum angle must exceed original low threshold');
  }
  if (value <= 135) return ANGLE_BIN_KEYS[0];
  if (value <= 150) return ANGLE_BIN_KEYS[1];
  if (value <= 165) return ANGLE_BIN_KEYS[2];
  return ANGLE_BIN_KEYS[3];
}

function diagnosticForCondition(source, frozen, m43Diagnostics, gt) {
  const measured = source.frames.map(frame => {
    const angle = measureCycleAngle('squat', frame.landmarks);
    return {timeMs: frame.time_ms, degrees: angle ? angle.degrees : null};
  });
  const predictions = detectLowToHighEvents(source.frames).event_times_ms;
  const matched = detailedMatch(predictions, gt);
  const mins = empty(ANGLE_BIN_KEYS);
  const context = empty(CONTEXT_KEYS);
  const falseLocations = empty(FP_LOCATION_KEYS);
  let noLow = 0, beforeUsable = 0, afterUsable = 0;
  const low = REP_POLICIES.squat.lowDeg;
  for (let i = 0; i < gt.length; i++) {
    if (matched.matchedGroundTruth.has(i)) continue;
    const g = gt[i];
    const within = measured.filter(x =>
      x.timeMs >= g.startMs && x.timeMs < g.endMs &&
      x.degrees !== null);
    if (within.length === 0 || within.some(x => x.degrees <= low)) continue;
    noLow++;
    const minimum = Math.min(...within.map(x => x.degrees));
    mins[minAngleBucket(minimum)]++;
    const before = measured.filter(x =>
      x.timeMs >= g.startMs - CONTEXT_MS &&
      x.timeMs < g.startMs && x.degrees !== null);
    const after = measured.filter(x =>
      x.timeMs >= g.endMs && x.timeMs < g.endMs + CONTEXT_MS &&
      x.degrees !== null);
    if (before.length) beforeUsable++;
    if (after.length) afterUsable++;
    const left = before.some(x => x.degrees <= low);
    const right = after.some(x => x.degrees <= low);
    if (left && right) context.low_both_sides++;
    else if (left) context.low_before_only++;
    else if (right) context.low_after_only++;
    else context.low_neither_side++;
  }
  for (let p = 0; p < predictions.length; p++) {
    if (matched.matchedPredictions.has(p)) continue;
    const inside = gt.some(g =>
      predictions[p] >= g.startMs && predictions[p] < g.endMs);
    falseLocations[inside ?
      'unmatched_inside_any_gt_interval' :
      'unmatched_outside_all_gt_intervals']++;
  }

  const ref = m43Diagnostics.unmatched_gt_window_observations
    .usable_but_no_low_in_gt_window;
  if (noLow !== ref ||
      Object.values(mins).reduce((a,c)=>a+c,0) !== noLow ||
      Object.values(context).reduce((a,c)=>a+c,0) !== noLow ||
      Object.values(falseLocations).reduce((a,c)=>a+c,0) !== frozen.low_to_high.event_fp) {
    throw new Error('M44 descriptive angle/FP categories disagree with frozen M43');
  }
  return {
    unmatched_gt_no_low_inside: noLow,
    min_angle_bins_for_no_low_gt: mins,
    low_in_750ms_context_for_no_low_gt: context,
    no_low_gt_with_usable_angle_before: beforeUsable,
    no_low_gt_with_usable_angle_after: afterUsable,
    unmatched_prediction_location: falseLocations
  };
}

export function analyzeTrainBoundaryTrace(source) {
  const m43 = analyzeTrainTrace(source);
  const gt = gtEvents(source);
  const sparseSource = {
    ...source, frames: source.frames.filter((_, i) => i % 2 === 0)
  };
  const observations = {
    dense_200ms: diagnosticForCondition(source, m43.m41.dense_200ms,
      m43.diagnostics.dense_200ms, gt),
    subsample_400ms: diagnosticForCondition(sparseSource,
      m43.m41.subsample_400ms, m43.diagnostics.subsample_400ms, gt)
  };
  return {
    protocol: PROTOCOL, source_split: 'train',
    annotated_rep_count: gt.length,
    m43, // Strict M43/M41 report of the identical replay, only until aggregation.
    context_ms: CONTEXT_MS,
    conditions: observations,
    limits: 'TRAIN-only descriptive boundaries. Context can include neighboring repetitions. Original low threshold, event matching, and dense MediaPipe context all unchanged.'
  };
}

function main() {
  const [, , sourcePath, outputPath] = process.argv;
  if (!sourcePath || !outputPath) {
    throw new Error('Usage: node web/scripts/m44_train_boundary_audit.js <dense_trace.json> <output.json>');
  }
  const data = JSON.parse(readFileSync(resolve(sourcePath), 'utf8'));
  const result = analyzeTrainBoundaryTrace(data);
  writeFileSync(resolve(outputPath), JSON.stringify(result, null, 2) + '\n',
    {flag: 'wx'});
  console.log('M44 TRAIN-only descriptive boundary-angle audit completed');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
