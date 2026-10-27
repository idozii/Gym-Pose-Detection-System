/** M39 development-only temporal return baseline for RepCount-A Squat.
 *
 * A deliberately simple, fixed-policy comparator: first observe a low knee
 * angle, then a high angle on the SAME visible side. It has no model training,
 * no user/video identifiers, no threshold search and no claim of good form.
 *
 * This module is not imported by the deployed website. The M34
 * RepCycleCounter and M37 one-to-one event matcher remain unchanged.
 */
import { measureCycleAngle, REP_POLICIES } from '../src/rep_counter.js';
import { evaluateTrace, matchEvents, TOLERANCE_MS, DURATION_FRACTION,
  MAX_TOLERANCE_MS } from './m37_eval_repcount_trace.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const M39_PROTOCOL = 'm39-train-only-temporal-return-v1';
export const FROZEN_RULE = Object.freeze({
  lowDeg: REP_POLICIES.squat.lowDeg,
  highDeg: REP_POLICIES.squat.highDeg,
  minimumExcursionDeg: REP_POLICIES.squat.minimumExcursionDeg,
  minReturnMs: 400,
  maxReturnMs: 12000,
  maxTrackingGapMs: 1250
});

/** Return timestamps only in ephemeral memory (never in aggregate reports).
 * Assumes M37-validated, strictly monotone pose traces.
 */
export function detectLowToHighEvents(frames) {
  if (!Array.isArray(frames)) throw new TypeError('M39 requires pose frames');
  const events = [];
  let side = null, lastGood = null, lowAt = null, minimumDeg = Infinity;
  let sideResets = 0, gapResets = 0;
  for (const sample of frames) {
    if (!sample || !Number.isFinite(sample.time_ms) ||
        !Object.hasOwn(sample, 'landmarks')) {
      throw new TypeError('Invalid M39 pose sample');
    }
    const time = sample.time_ms;
    if (lastGood !== null && time - lastGood > FROZEN_RULE.maxTrackingGapMs) {
      if (lowAt !== null) gapResets++;
      lowAt = null;
      minimumDeg = Infinity;
      side = null;
    }
    const angle = measureCycleAngle('squat', sample.landmarks);
    if (!angle) continue;
    if (side !== null && angle.side !== side) {
      if (lowAt !== null) sideResets++;
      lowAt = null;
      minimumDeg = Infinity;
    }
    side = angle.side;
    lastGood = time;

    if (lowAt !== null && time - lowAt > FROZEN_RULE.maxReturnMs) {
      lowAt = null;
      minimumDeg = Infinity;
    }
    if (angle.degrees <= FROZEN_RULE.lowDeg) {
      if (lowAt === null) lowAt = time;
      minimumDeg = Math.min(minimumDeg, angle.degrees);
    } else if (lowAt !== null &&
               angle.degrees >= FROZEN_RULE.highDeg &&
               time - lowAt >= FROZEN_RULE.minReturnMs &&
               angle.degrees - minimumDeg >= FROZEN_RULE.minimumExcursionDeg) {
      events.push(time);
      lowAt = null;
      minimumDeg = Infinity;
    }
  }
  return { event_times_ms: events, side_resets_pending: sideResets,
    tracking_gap_resets_pending: gapResets };
}

function eventBounds(source) {
  const gt = [];
  for (let i = 0; i < source.cycle_bounds_frames.length; i += 2) {
    const first = source.cycle_bounds_frames[i];
    const last = source.cycle_bounds_frames[i + 1];
    const duration = (last - first) / source.fps * 1000;
    gt.push({
      endMs: last / source.fps * 1000,
      toleranceMs: Math.min(MAX_TOLERANCE_MS,
        Math.max(TOLERANCE_MS, duration * DURATION_FRACTION))
    });
  }
  return gt;
}

function resultForEvents(eventTimes, annotated, gt) {
  const matched = matchEvents(eventTimes, gt);
  const precision = eventTimes.length ?
    matched.tp / eventTimes.length : null;
  const recall = annotated ? matched.tp / annotated : null;
  const f1 = precision === null || recall === null ? null :
    precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  return {
    annotated_count: annotated,
    predicted_count: eventTimes.length,
    absolute_count_error: Math.abs(annotated - eventTimes.length),
    signed_count_error: eventTimes.length - annotated,
    event_tp: matched.tp, event_fp: matched.fp, event_fn: matched.fn,
    event_precision: precision, event_recall: recall, event_f1: f1
  };
}

/** Same timestamped poses, publisher labels and one-to-one matching
 * for M34, always-zero, and a simple low-to-high comparator.
 */
export function compareTrainTrace(source) {
  const original = evaluateTrace(source); // also validates publisher metadata
  const annotated = original.annotated_rep_count;
  const gt = eventBounds(source);
  const candidate = detectLowToHighEvents(source.frames);
  const temporal = resultForEvents(candidate.event_times_ms, annotated, gt);
  const m34 = {
    annotated_count: annotated,
    predicted_count: original.website_observed_cycles,
    absolute_count_error: original.absolute_count_error,
    signed_count_error: original.count_error,
    event_tp: original.event_tp, event_fp: original.event_fp,
    event_fn: original.event_fn, event_precision: original.event_precision,
    event_recall: original.event_recall, event_f1: original.event_f1
  };
  const zero = resultForEvents([], annotated, gt);
  return {
    protocol: M39_PROTOCOL,
    source_split: 'train',
    annotation_rep_count: annotated,
    input_frame_samples: source.frames.length,
    pose_sample_coverage: original.pose_sample_coverage,
    m34,
    low_to_high: temporal,
    always_zero: zero,
    temporal_diagnostics: {
      pending_side_resets: candidate.side_resets_pending,
      pending_tracking_gap_resets: candidate.tracking_gap_resets_pending
    },
    assumptions: 'Development-only; Python Solutions poses; unverified publisher sync and browser parity; NOT physical form quality'
  };
}

function main() {
  const [, , sourcePath, outputPath] = process.argv;
  if (!sourcePath || !outputPath) {
    throw new Error('Usage: node web/scripts/m39_compare_train_trace.js <trace.json> <output.json>');
  }
  const trace = JSON.parse(readFileSync(resolve(sourcePath), 'utf8'));
  const result = compareTrainTrace(trace);
  writeFileSync(resolve(outputPath), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log('M39 train-only paired baselines evaluated');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
