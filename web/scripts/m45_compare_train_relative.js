/** M45: TRAIN-only hypothesis test - relative excursion vs frozen M39.
 *
 * M44 showed many GT intervals whose lowest observed knee angle
 * remained >120deg. This opt-in RESEARCH comparator uses a peak-to-trough
 * excursion and then returns to a high pose; it does not use absolute
 * low <=120deg. M34, M39, the M37 matcher, pose extraction, fixed TRAIN
 * subset, and deployed browser source remain untouched.
 *
 * Experiment designed AFTER looking at M43/M44 on these same TRAIN clips.
 * Its scores are exploratory development evidence, never held-out TEST.
 */
import { measureCycleAngle, REP_POLICIES } from '../src/rep_counter.js';
import { FROZEN_RULE, detectLowToHighEvents } from './m39_compare_train_trace.js';
import { analyzeTrainBoundaryTrace } from './m44_train_boundary_audit.js';
import { gtEvents, detailedMatch } from './m43_analyse_train_errors.js';
import { matchEvents } from './m37_eval_repcount_trace.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PROTOCOL = 'm45-train-relative-excursion-comparator-v1';
export const RULE = Object.freeze({
  highDeg: REP_POLICIES.squat.highDeg,
  minDropDeg: REP_POLICIES.squat.minimumExcursionDeg,
  minRiseDeg: REP_POLICIES.squat.minimumExcursionDeg,
  minReturnMs: FROZEN_RULE.minReturnMs,
  maxReturnMs: FROZEN_RULE.maxReturnMs,
  maxTrackingGapMs: FROZEN_RULE.maxTrackingGapMs
});

/** No GT or event timestamps are read while making detector predictions.
 * One same-side >=155deg peak, >=35deg decline, then >=35deg recovery
 * ending >=155deg after 400ms and within 12s forms one candidate event.
 */
export function detectRelativeExcursionEvents(frames) {
  if (!Array.isArray(frames)) throw new TypeError('M45 requires pose frames');
  let previousTime = -Infinity, lastGood = null, lastSide = null;
  let peak = null, lowAt = null, minDeg = Infinity;
  let pendingSideResets = 0, pendingGapResets = 0, timedOut = 0;
  const eventTimes = [];
  const reset = () => {peak = null; lowAt = null; minDeg = Infinity;};
  for (const sample of frames) {
    if (!sample || !Number.isFinite(sample.time_ms) ||
        sample.time_ms <= previousTime ||
        !Object.hasOwn(sample, 'landmarks')) {
      throw new TypeError('M45 requires chronological timestamped pose frames');
    }
    const t = sample.time_ms;
    previousTime = t;
    if (lastGood !== null && t - lastGood > RULE.maxTrackingGapMs) {
      if (lowAt !== null) pendingGapResets++;
      reset();
      lastSide = null;
      lastGood = null;
    }
    const angle = measureCycleAngle('squat', sample.landmarks);
    if (!angle) continue;
    if (lastSide !== null && angle.side !== lastSide) {
      if (lowAt !== null) pendingSideResets++;
      reset();
    }
    lastGood = t;
    lastSide = angle.side;
    const v = angle.degrees;
    if (lowAt !== null && t - lowAt > RULE.maxReturnMs) {
      timedOut++;
      reset();
    }
    if (lowAt === null) {
      if (v >= RULE.highDeg) peak = Math.max(peak ?? -Infinity, v);
      if (peak !== null && peak - v >= RULE.minDropDeg) {
        lowAt = t;
        minDeg = v;
      }
      continue;
    }
    minDeg = Math.min(minDeg, v);
    if (t - lowAt >= RULE.minReturnMs &&
        v >= RULE.highDeg &&
        v - minDeg >= RULE.minRiseDeg) {
      eventTimes.push(t);
      reset();
      peak = v;  // a completed return is the next cycle's high anchor
    }
  }
  return {
    event_times_ms: eventTimes,
    pending_side_resets: pendingSideResets,
    pending_gap_resets: pendingGapResets,
    pending_timeouts: timedOut
  };
}

function score(events, gt) {
  const matches = matchEvents(events, gt);
  const n = gt.length, pred = events.length;
  const precision = pred > 0 ? matches.tp / pred : null;
  const recall = n > 0 ? matches.tp / n : null;
  const f1 = precision === null || recall === null ? null :
    precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  return {
    annotated_count: n,
    predicted_count: pred,
    absolute_count_error: Math.abs(pred - n),
    signed_count_error: pred - n,
    event_tp: matches.tp,
    event_fp: matches.fp,
    event_fn: matches.fn,
    event_precision: precision,
    event_recall: recall,
    event_f1: f1
  };
}

export function compareTrainRelativeExcursion(source) {
  // M44 fully validates original source and executes the same M41/M43
  // baselines, preserved unmodified for the official comparison.
  const m44 = analyzeTrainBoundaryTrace(source);
  const gt = gtEvents(source);
  const views = {
    dense_200ms: source.frames,
    subsample_400ms: source.frames.filter((_, i) => i % 2 === 0)
  };
  const conditions = {};
  for (const [key, frames] of Object.entries(views)) {
    const variant = detectRelativeExcursionEvents(frames);
    const baseline = detectLowToHighEvents(frames);
    const newScore = score(variant.event_times_ms, gt);
    const frozen = m44.m43.m41[key].low_to_high;
    const b = detailedMatch(baseline.event_times_ms, gt);
    const v = detailedMatch(variant.event_times_ms, gt);
    const paired = {
      both_matched: 0,
      new_only_matched: 0,
      original_only_matched: 0,
      neither_matched: 0
    };
    for (let i = 0; i < gt.length; i++) {
      const baseHit = b.matchedGroundTruth.has(i);
      const newHit = v.matchedGroundTruth.has(i);
      if (baseHit && newHit) paired.both_matched++;
      else if (newHit) paired.new_only_matched++;
      else if (baseHit) paired.original_only_matched++;
      else paired.neither_matched++;
    }
    if (frozen.event_tp !== b.matchedGroundTruth.size ||
        frozen.event_fp !== baseline.event_times_ms.length - b.matchedPredictions.size ||
        newScore.event_tp !== v.matchedGroundTruth.size ||
        newScore.event_fp !== variant.event_times_ms.length - v.matchedPredictions.size ||
        paired.both_matched + paired.original_only_matched !== frozen.event_tp ||
        paired.both_matched + paired.new_only_matched !== newScore.event_tp) {
      throw new Error('M45 mismatched paired original/new M37 event scores');
    }
    let fpInside = 0, fpOutside = 0;
    for (let i = 0; i < variant.event_times_ms.length; i++) {
      if (v.matchedPredictions.has(i)) continue;
      const timestamp = variant.event_times_ms[i];
      if (gt.some(event => timestamp >= event.startMs && timestamp < event.endMs)) {
        fpInside++;
      } else fpOutside++;
    }
    if (fpInside + fpOutside !== newScore.event_fp) {
      throw new Error('M45 unmatched event interval locations disagree with FP');
    }
    conditions[key] = {
      relative_excursion: newScore,
      paired_gt_vs_original_low_to_high: paired,
      candidate_fp_location: {
        unmatched_inside_any_gt_interval: fpInside,
        unmatched_outside_all_gt_intervals: fpOutside
      },
      pending_resets: {
        side: variant.pending_side_resets,
        gap: variant.pending_gap_resets,
        timeout: variant.pending_timeouts
      }
    };
  }
  return {
    protocol: PROTOCOL, source_split: 'train',
    annotated_rep_count: gt.length,
    rule: RULE,
    original_m44: m44,
    conditions,
    caveat: 'Same previously analyzed TRAIN cohort; relative motion is a post-hoc hypothesis, not a held-out improvement or physical squat-form metric.'
  };
}

function main() {
  const [, , input, output] = process.argv;
  if (!input || !output) {
    throw new Error('Usage: node web/scripts/m45_compare_train_relative.js <trace.json> <result.json>');
  }
  const trace = JSON.parse(readFileSync(resolve(input), 'utf8'));
  const row = compareTrainRelativeExcursion(trace);
  writeFileSync(resolve(output), JSON.stringify(row, null, 2) + '\n', {flag:'wx'});
  console.log('M45 TRAIN-only experimental relative-excursion comparison completed');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
