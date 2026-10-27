/** M43: observational error audit of frozen M39/M41 TRAIN-only methods.
 *
 * Only cohort aggregates are retained. Detection timestamps, publisher
 * intervals, source poses and video IDs stay in ephemeral local processing.
 * No model parameters, M34, sampling cadence or event tolerance are tuned.
 */
import { measureCycleAngle, REP_POLICIES } from '../src/rep_counter.js';
import {
  matchEvents, TOLERANCE_MS, DURATION_FRACTION, MAX_TOLERANCE_MS
} from './m37_eval_repcount_trace.js';
import { detectLowToHighEvents } from './m39_compare_train_trace.js';
import { pairedCadenceComparison } from './m41_compare_train_cadence.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PROTOCOL = 'm43-train-temporal-error-audit-v1';

export const FN_KEYS = Object.freeze([
  'no_usable_angle_in_gt_window',
  'usable_but_no_low_in_gt_window',
  'low_but_no_later_high_in_gt_window',
  'low_then_high_in_gt_window_still_unmatched'
]);
export const FP_KEYS = Object.freeze([
  'unmatched_within_some_gt_end_tolerance',
  'unmatched_outside_all_gt_end_tolerances'
]);

export function gtEvents(source) {
  const bounds = source.cycle_bounds_frames;
  const gt = [];
  for (let i = 0; i < bounds.length; i += 2) {
    const startMs = bounds[i] / source.fps * 1000;
    const endMs = bounds[i + 1] / source.fps * 1000;
    gt.push({
      startMs,
      endMs,
      toleranceMs: Math.min(MAX_TOLERANCE_MS,
        Math.max(TOLERANCE_MS, (endMs - startMs) * DURATION_FRACTION))
    });
  }
  return gt;
}

/** Exactly the same greedy, nearest GT-end, one-to-one matching policy
 * and tie order as M37. Expose indices only within this module.
 */
export function detailedMatch(predictedMs, gt) {
  const edges = [];
  for (let p = 0; p < predictedMs.length; p++) {
    for (let g = 0; g < gt.length; g++) {
      const distance = Math.abs(predictedMs[p] - gt[g].endMs);
      if (distance <= gt[g].toleranceMs) edges.push({p, g, distance});
    }
  }
  edges.sort((a, b) => a.distance - b.distance || a.g - b.g || a.p - b.p);
  const matchedPredictions = new Set();
  const matchedGroundTruth = new Set();
  for (const edge of edges) {
    if (matchedPredictions.has(edge.p) || matchedGroundTruth.has(edge.g)) continue;
    matchedPredictions.add(edge.p);
    matchedGroundTruth.add(edge.g);
  }
  const original = matchEvents(predictedMs, gt);
  if (original.tp !== matchedGroundTruth.size ||
      original.fp !== predictedMs.length - matchedPredictions.size ||
      original.fn !== gt.length - matchedGroundTruth.size) {
    throw new Error('M43 event matching disagrees with frozen M37');
  }
  return {matchedPredictions, matchedGroundTruth};
}

function emptyBuckets(keys) {
  return Object.fromEntries(keys.map(key => [key, 0]));
}

function analyzeCondition(source, baseline, gt) {
  const candidate = detectLowToHighEvents(source.frames);
  const predicted = candidate.event_times_ms;
  const matched = detailedMatch(predicted, gt);
  const metrics = baseline.low_to_high;
  if (matched.matchedGroundTruth.size !== metrics.event_tp ||
      predicted.length - matched.matchedPredictions.size !== metrics.event_fp ||
      gt.length - matched.matchedGroundTruth.size !== metrics.event_fn ||
      predicted.length !== metrics.predicted_count) {
    throw new Error('M43 detector events do not reproduce M41 TP/FP/FN');
  }
  const angleSamples = source.frames.map(sample => {
    const geometry = measureCycleAngle('squat', sample.landmarks);
    return {
      time: sample.time_ms,
      degrees: geometry ? geometry.degrees : null,
      side: geometry ? geometry.side : null
    };
  });
  const phenotypes = emptyBuckets(FN_KEYS);
  const falsePositive = emptyBuckets(FP_KEYS);
  let gtUsableLt4 = 0, gtUsableLt6 = 0, gtNoUsable = 0;
  let gtWithLowThenHigh = 0, gtWithLow = 0;
  for (let i = 0; i < gt.length; i++) {
    const window = gt[i];
    // Original publisher windows are HALF-OPEN [start, end). A high
    // return after publisher end is not included in these observations.
    const within = angleSamples.filter(a =>
      a.time >= window.startMs && a.time < window.endMs &&
      a.degrees !== null);
    const usable = within.length;
    if (usable < 4) gtUsableLt4++;
    if (usable < 6) gtUsableLt6++;
    if (usable === 0) gtNoUsable++;
    const lowIndex = within.findIndex(a =>
      a.degrees <= REP_POLICIES.squat.lowDeg);
    const low = lowIndex >= 0;
    if (low) gtWithLow++;
    // Descriptive ordered low then high in publisher window. A
    // side switch is deliberately not treated as sufficient for a rep.
    const ordered = low && within.some((a, j) =>
      j > lowIndex && a.degrees >= REP_POLICIES.squat.highDeg);
    if (ordered) gtWithLowThenHigh++;
    if (matched.matchedGroundTruth.has(i)) continue;
    if (!usable) phenotypes.no_usable_angle_in_gt_window++;
    else if (!low) phenotypes.usable_but_no_low_in_gt_window++;
    else if (!ordered) phenotypes.low_but_no_later_high_in_gt_window++;
    else phenotypes.low_then_high_in_gt_window_still_unmatched++;
  }
  for (let i = 0; i < predicted.length; i++) {
    if (matched.matchedPredictions.has(i)) continue;
    const nearSome = gt.some(g =>
      Math.abs(predicted[i] - g.endMs) <= g.toleranceMs);
    falsePositive[nearSome ?
      'unmatched_within_some_gt_end_tolerance' :
      'unmatched_outside_all_gt_end_tolerances']++;
  }
  if (Object.values(phenotypes).reduce((a, b) => a + b, 0) !== metrics.event_fn ||
      Object.values(falsePositive).reduce((a, b) => a + b, 0) !== metrics.event_fp) {
    throw new Error('M43 FN and FP diagnostic partitions are not exhaustive');
  }
  return {
    gt_events: gt.length,
    tp: metrics.event_tp, fp: metrics.event_fp, fn: metrics.event_fn,
    unmatched_gt_window_observations: phenotypes,
    unmatched_predictions_by_tolerance: falsePositive,
    all_gt_windows_usable_lt4: gtUsableLt4,
    all_gt_windows_usable_lt6: gtUsableLt6,
    all_gt_windows_no_usable: gtNoUsable,
    all_gt_windows_with_any_low: gtWithLow,
    all_gt_windows_with_low_then_high: gtWithLowThenHigh,
    side_switch_resets_while_pending: candidate.side_resets_pending,
    tracking_gap_resets_while_pending: candidate.tracking_gap_resets_pending,
    _matchedGt: matched.matchedGroundTruth
  };
}

/** Per-video paired diagnostic, no source ID/landmark/frame timestamps emitted. */
export function analyzeTrainTrace(source) {
  const m41 = pairedCadenceComparison(source); // frozen evaluation and validation
  if (m41.source_split !== 'train') throw new Error('M43 requires TRAIN');
  const gt = gtEvents(source);
  const subsampled = {...source, frames: source.frames.filter((_, i) => i % 2 === 0)};
  const d = analyzeCondition(source, m41.dense_200ms, gt);
  const s = analyzeCondition(subsampled, m41.subsample_400ms, gt);
  const pairedGt = {
    both_matched: 0,
    dense_only_matched: 0,
    subsample_only_matched: 0,
    neither_matched: 0
  };
  for (let i = 0; i < gt.length; i++) {
    const denseHit = d._matchedGt.has(i);
    const sparseHit = s._matchedGt.has(i);
    if (denseHit && sparseHit) pairedGt.both_matched++;
    else if (denseHit) pairedGt.dense_only_matched++;
    else if (sparseHit) pairedGt.subsample_only_matched++;
    else pairedGt.neither_matched++;
  }
  delete d._matchedGt;
  delete s._matchedGt;
  return {
    protocol: PROTOCOL,
    source_split: 'train',
    annotated_rep_count: gt.length,
    m41,
    diagnostics: {
      dense_200ms: d, subsample_400ms: s,
      paired_gt_outcomes: pairedGt
    },
    interpretation: 'Descriptive half-open publisher GT windows, not causal FP/FN explanations. The paired conditions share a dense MediaPipe tracking history.'
  };
}

function main() {
  const [, , sourcePath, outputPath] = process.argv;
  if (!sourcePath || !outputPath) {
    throw new Error('Usage: node web/scripts/m43_analyse_train_errors.js <dense_trace.json> <output.json>');
  }
  const source = JSON.parse(readFileSync(resolve(sourcePath), 'utf8'));
  const result = analyzeTrainTrace(source);
  writeFileSync(resolve(outputPath), JSON.stringify(result, null, 2) + '\n',
    {flag: 'wx'});
  console.log('M43 TRAIN-only temporal-error audit completed');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
