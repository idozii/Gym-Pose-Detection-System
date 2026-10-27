/** M37 replay actual chronological MediaPipe pose traces through *unmodified* M34.
 *
 * Ground truth comes from independently published RepCount cycle bounds,
 * NOT from our detector output. No threshold selection/tuning here.
 * Node replay uses the original browser RepCycleCounter implementation.
 */
import { RepCycleCounter, REP_POLICIES } from '../src/rep_counter.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const EVAL_PROTOCOL = 'm37-repcount-squat-event-v1';
export const TOLERANCE_MS = 750;
export const DURATION_FRACTION = 0.35;
export const MAX_TOLERANCE_MS = 2500;

function gtEvents(metadata) {
  if (!metadata || metadata.action !== 'squat' ||
      !Number.isFinite(metadata.fps) || metadata.fps <= 0 ||
      !Number.isInteger(metadata.total_frames) || metadata.total_frames <= 5 ||
      !Array.isArray(metadata.cycle_bounds_frames) ||
      metadata.cycle_bounds_frames.length % 2 !== 0 ||
      metadata.cycle_bounds_frames.length > 400) {
    throw new Error('M37 requires unmodified paired Squat frame bounds and video FPS');
  }
  const frames = metadata.cycle_bounds_frames;
  const events = [];
  let lastEnd = -1;
  for (let i = 0; i < frames.length; i += 2) {
    const start = frames[i], end = frames[i + 1];
    if (!Number.isInteger(start) || !Number.isInteger(end) ||
        start < 0 || start >= end || end > metadata.total_frames ||
        start < lastEnd) {
      throw new Error('Invalid or overlapping RepCount source cycle frame intervals');
    }
    lastEnd = end;
    const durationMs = (end - start) / metadata.fps * 1000;
    events.push({
      endMs: end / metadata.fps * 1000,
      toleranceMs: Math.min(MAX_TOLERANCE_MS,
        Math.max(TOLERANCE_MS, durationMs * DURATION_FRACTION))
    });
  }
  if (metadata.annotated_count !== events.length) {
    throw new Error('RepCount published count and paired event bounds disagree');
  }
  return events;
}

/** One-to-one nearest-time assignment, deterministic under ties.
 * Given few short clips, sort edges by time separation. Does not count
 * one website event against multiple human annotated events.
 */
export function matchEvents(predictedMs, gt) {
  const edges = [];
  for (let p = 0; p < predictedMs.length; p++) {
    for (let g = 0; g < gt.length; g++) {
      const distance = Math.abs(predictedMs[p] - gt[g].endMs);
      if (distance <= gt[g].toleranceMs) edges.push({ p, g, distance });
    }
  }
  edges.sort((a, b) => a.distance - b.distance || a.g - b.g || a.p - b.p);
  const usedP = new Set(), usedG = new Set();
  const matchedDifferences = [];
  for (const edge of edges) {
    if (usedP.has(edge.p) || usedG.has(edge.g)) continue;
    usedP.add(edge.p); usedG.add(edge.g);
    matchedDifferences.push(edge.distance);
  }
  return {
    tp: usedG.size,
    fp: predictedMs.length - usedP.size,
    fn: gt.length - usedG.size,
    mean_absolute_matched_end_error_ms: matchedDifferences.length
      ? matchedDifferences.reduce((a, b) => a + b, 0) / matchedDifferences.length
      : null
  };
}

export function evaluateTrace(source) {
  const gt = gtEvents(source);
  if (!Array.isArray(source.frames) || source.frames.length < 8 ||
      source.frames.length > 3000) {
    throw new Error('Too few/many actual timestamped pose frames');
  }
  const counter = new RepCycleCounter();
  const predictedMs = [];
  // Post-hoc observability is DESCRIPTIVE, not a tuning experiment.
  // Use the exact existing M34 counter and its published thresholds;
  // expose only aggregate counters (no poses, names or timestamps).
  const observability = {
    protocol: 'm37-squat-counter-observability-v1',
    input_frame_samples: source.frames.length,
    counter_accepted_ticks: 0,
    sampling_gate_skips: 0,
    knee_angle_usable_ticks: 0,
    knee_angle_unavailable_ticks: 0,
    knee_angle_high_ticks: 0,
    knee_angle_low_ticks: 0,
    knee_angle_middle_ticks: 0,
    confirmed_initial_high_transitions: 0,
    confirmed_low_transitions: 0,
    completed_return_transitions: 0,
    tracked_side_changes: 0,
    tracking_gap_resets: 0
  };
  let previousTime = -Infinity, validFrames = 0;
  let previousCount = 0;
  for (const sample of source.frames) {
    if (!sample || !Number.isFinite(sample.time_ms) ||
        sample.time_ms <= previousTime || sample.time_ms < 0 ||
        sample.time_ms > ((source.total_frames + 1) / source.fps * 1000) ||
        !('landmarks' in sample)) {
      throw new Error('Invalid or unsorted source frame timestamp');
    }
    previousTime = sample.time_ms;
    if (sample.landmarks !== null) {
      if (!Array.isArray(sample.landmarks) || sample.landmarks.length !== 33) {
        throw new Error('Unexpected MediaPipe source pose landmark count');
      }
      validFrames++;
    }
    const previousTick = counter.lastTick;
    const previousSide = counter.side;
    const previousGood = counter.lastGood;
    const previousPhase = counter.phase;
    const out = counter.observe('squat', sample.landmarks, sample.time_ms);
    if (previousGood !== null &&
        sample.time_ms - previousGood > counter.trackingGapMs) {
      observability.tracking_gap_resets++;
    }
    if (counter.lastTick === previousTick) {
      observability.sampling_gate_skips++;
    } else {
      observability.counter_accepted_ticks++;
      if (out.tracking === 'visible' && Number.isFinite(out.angle)) {
        observability.knee_angle_usable_ticks++;
        if (out.angle >= REP_POLICIES.squat.highDeg) {
          observability.knee_angle_high_ticks++;
        } else if (out.angle <= REP_POLICIES.squat.lowDeg) {
          observability.knee_angle_low_ticks++;
        } else {
          observability.knee_angle_middle_ticks++;
        }
        if (previousSide !== null && out.side !== previousSide) {
          observability.tracked_side_changes++;
        }
      } else {
        observability.knee_angle_unavailable_ticks++;
      }
      if (previousPhase === 'await_high' && out.phase === 'await_low') {
        observability.confirmed_initial_high_transitions++;
      } else if (previousPhase === 'await_low' && out.phase === 'await_return') {
        observability.confirmed_low_transitions++;
      } else if (previousPhase === 'await_return' && out.phase === 'await_low') {
        observability.completed_return_transitions++;
      }
    }
    if (out.cycles > previousCount) {
      if (out.cycles !== previousCount + 1) throw new Error('Counter increased by multiple events');
      predictedMs.push(sample.time_ms);
      previousCount = out.cycles;
    }
  }
  const match = matchEvents(predictedMs, gt);
  // Per-clip event recall is mathematically undefined when this
  // publisher-annotated source has zero positive repetitions. Report null
  // rather than NaN or an invented 0%/100%; event FP still counts.
  const precision = predictedMs.length ? match.tp / predictedMs.length : null;
  const recall = gt.length ? match.tp / gt.length : null;
  const f1 = precision !== null && recall !== null
    ? (precision + recall ? 2 * precision * recall / (precision + recall) : 0)
    : null;
  return {
    protocol: EVAL_PROTOCOL,
    dataset: 'RepCount-A source video with original cycle annotations; archive provenance separately verifiable',
    action: 'squat',
    annotated_rep_count: gt.length,
    website_observed_cycles: predictedMs.length,
    absolute_count_error: Math.abs(gt.length - predictedMs.length),
    count_error: predictedMs.length - gt.length,
    frame_samples: source.frames.length,
    pose_visible_samples: validFrames,
    pose_sample_coverage: validFrames / source.frames.length,
    m37_observability: observability,
    event_tp: match.tp,
    event_fp: match.fp,
    event_fn: match.fn,
    event_precision: precision,
    event_recall: recall,
    event_f1: f1,
    zero_annotated_rep_negative_control: gt.length === 0,
    negative_control_spurious_cycles: gt.length === 0 ? predictedMs.length : null,
    mean_absolute_matched_end_error_ms: match.mean_absolute_matched_end_error_ms,
    tolerance_policy: 'max(750ms, 35% labeled cycle duration), capped 2500ms',
    source_frame_bounds_within_decoded_clip: true,
    source_frame_annotation_sync_independently_verified: false,
    browser_python_estimator_parity_validated: false,
    m34_counter_parameters_modified: false,
    disclaimer: 'Measured on selected labeled third-party Squat videos. Counts depend on camera pose visibility, source clipping, fixed event tolerance and Python pose estimator. Not a technique safety or correctness classifier.'
  };
}

function main() {
  const [, , tracePath, outputPath] = process.argv;
  if (!tracePath || !outputPath) throw new Error('Usage: node web/scripts/m37_eval_repcount_trace.js <trace.json> <output.json>');
  const trace = JSON.parse(readFileSync(resolve(tracePath), 'utf8'));
  const result = evaluateTrace(trace);
  writeFileSync(resolve(outputPath), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log('M37 Squat trace evaluated:', result.annotated_rep_count,
    'human annotated reps,', result.website_observed_cycles, 'detected events');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
