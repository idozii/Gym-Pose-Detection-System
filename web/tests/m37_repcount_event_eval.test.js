import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import { evaluateTrace, matchEvents, EVAL_PROTOCOL } from '../scripts/m37_eval_repcount_trace.js';

function trace(sequence, bounds, { fps = 10, totalFrames = 45 } = {}) {
  return {
    action: 'squat', fps, total_frames: totalFrames,
    annotated_count: bounds.length / 2,
    cycle_bounds_frames: bounds,
    frames: sequence.map((degrees, i) => ({
      time_ms: i * 400,
      landmarks: degrees === null ? null : poseFromAngle('squat', degrees)
    }))
  };
}

test('M37 replays EXACT original M34 browser state machine on labeled cycles', () => {
  const input = trace(
    [170,170,100,100,170,170,100,100,170,170],
    [0,20,20,36]
  );
  const result = evaluateTrace(input);
  assert.equal(result.protocol, EVAL_PROTOCOL);
  assert.equal(result.annotated_rep_count, 2);
  assert.equal(result.website_observed_cycles, 2);
  assert.equal(result.event_tp, 2);
  assert.equal(result.event_fp, 0);
  assert.equal(result.event_fn, 0);
  assert.equal(result.event_precision, 1);
  assert.equal(result.event_recall, 1);
  assert.equal(result.event_f1, 1);
  assert.equal(result.absolute_count_error, 0);
  assert.equal(result.m34_counter_parameters_modified, false);
  assert.equal(result.source_frame_annotation_sync_independently_verified, false);
  assert.equal(result.browser_python_estimator_parity_validated, false);
});

test('M37 detects missing real rep events instead of matching only total counts', () => {
  const input = trace(
    [170,170,100,100,170,170,170,170,170,170],
    [0,20,20,36]
  );
  const result = evaluateTrace(input);
  assert.equal(result.website_observed_cycles, 1);
  assert.equal(result.event_tp, 1);
  assert.equal(result.event_fp, 0);
  assert.equal(result.event_fn, 1);
  assert.equal(result.event_precision, 1);
  assert.equal(result.event_recall, .5);
  assert.ok(result.event_f1 < 1);
});


test('M37 zero-ground-truth Squat clip is an explicit negative control, not a data error', () => {
  const still = trace(Array(10).fill(170), []);
  const clean = evaluateTrace(still);
  assert.equal(clean.annotated_rep_count, 0);
  assert.equal(clean.website_observed_cycles, 0);
  assert.equal(clean.event_tp, 0);
  assert.equal(clean.event_fp, 0);
  assert.equal(clean.event_fn, 0);
  assert.equal(clean.absolute_count_error, 0);
  assert.equal(clean.zero_annotated_rep_negative_control, true);
  assert.equal(clean.negative_control_spurious_cycles, 0);
  assert.equal(clean.event_precision, null);
  assert.equal(clean.event_recall, null);
  assert.equal(clean.event_f1, null);

  const falseCount = trace([170,170,100,100,170,170,170,170], []);
  const result = evaluateTrace(falseCount);
  assert.equal(result.website_observed_cycles, 1);
  assert.equal(result.event_tp, 0);
  assert.equal(result.event_fp, 1);
  assert.equal(result.event_fn, 0);
  assert.equal(result.absolute_count_error, 1);
  assert.equal(result.negative_control_spurious_cycles, 1);
  assert.equal(result.event_precision, 0);
  assert.equal(result.event_recall, null);
  assert.equal(result.event_f1, null);
});


test('M37.1 records original M34 sample-gate and knee-angle phase observability only', () => {
  const input = trace([170,170,100,100,170,170,170,170], [0,20]);
  const full = evaluateTrace(input);
  const d = full.m37_observability;
  assert.equal(d.protocol, 'm37-squat-counter-observability-v1');
  assert.equal(d.input_frame_samples, 8);
  assert.equal(d.counter_accepted_ticks, 8);
  assert.equal(d.sampling_gate_skips, 0);
  assert.equal(d.knee_angle_usable_ticks, 8);
  assert.equal(d.knee_angle_high_ticks, 6);
  assert.equal(d.knee_angle_low_ticks, 2);
  assert.equal(d.confirmed_initial_high_transitions, 1);
  assert.equal(d.confirmed_low_transitions, 1);
  assert.equal(d.completed_return_transitions, 1);
  assert.equal(full.website_observed_cycles, 1);
  assert.equal(full.m34_counter_parameters_modified, false);
  assert.equal('landmarks' in d, false);
  assert.equal('predicted_event_timestamps' in d, false);

  const fastInput = trace([170,170,100,100,170,170,170,170], [0,20]);
  fastInput.frames.forEach((frame, index) => { frame.time_ms = index * 200; });
  fastInput.frames[2].landmarks = null;
  const fast = evaluateTrace(fastInput);
  const summary = fast.m37_observability;
  assert.equal(summary.input_frame_samples, 8);
  assert.equal(summary.counter_accepted_ticks, 4);
  assert.equal(summary.sampling_gate_skips, 4);
  assert.equal(summary.knee_angle_usable_ticks, 3);
  assert.equal(summary.knee_angle_unavailable_ticks, 1);
  assert.equal(summary.knee_angle_high_ticks + summary.knee_angle_low_ticks +
    summary.knee_angle_middle_ticks, summary.knee_angle_usable_ticks);
  assert.equal(summary.knee_angle_usable_ticks + summary.knee_angle_unavailable_ticks,
    summary.counter_accepted_ticks);
});

test('M37 prevents one detected event from matching two ground truths', () => {
  const actual = matchEvents([2000], [
    {endMs: 1900, toleranceMs: 300},
    {endMs: 2100, toleranceMs: 300}
  ]);
  assert.equal(actual.tp, 1);
  assert.equal(actual.fp, 0);
  assert.equal(actual.fn, 1);
  const falseAlarm = matchEvents([2000, 2001], [
    {endMs: 100, toleranceMs: 200}
  ]);
  assert.equal(falseAlarm.tp, 0);
  assert.equal(falseAlarm.fp, 2);
  assert.equal(falseAlarm.fn, 1);
});

test('M37 rejects annotations not aligned to decoded source clip indices', () => {
  const input = trace([170,170,100,100,170,170,100,100], [0,100]);
  assert.throws(() => evaluateTrace(input), /Invalid or overlapping/);
  const mismatch = trace([170,170,100,100,170,170,100,100], [0,20]);
  mismatch.annotated_count = 2;
  assert.throws(() => evaluateTrace(mismatch), /published count/);
  const missingPairs = trace([170,170,100,100,170,170,100,100], [0,20]);
  missingPairs.cycle_bounds_frames.push(22);
  assert.throws(() => evaluateTrace(missingPairs), /paired Squat/);
  const wrongAction = trace([170,170,100,100,170,170,100,100], [0,20]);
  wrongAction.action = 'barbell_row';
  assert.throws(() => evaluateTrace(wrongAction), /paired Squat/);
});

test('M37 rejects out-of-order trace timestamps and malformed poses', () => {
  const sample = trace([170,170,100,100,170,170,170,170], [0,20]);
  sample.frames[4].time_ms = sample.frames[3].time_ms;
  assert.throws(() => evaluateTrace(sample), /Invalid or unsorted/);
  const bad = trace([170,170,100,100,170,170,170,170], [0,20]);
  bad.frames[0].landmarks = Array(20).fill({});
  assert.throws(() => evaluateTrace(bad), /source pose landmark count/);
});

test('M37 evaluator is a strictly offline browser-replay command', () => {
  const source = readFileSync(new URL('../scripts/m37_eval_repcount_trace.js', import.meta.url), 'utf8');
  assert.match(source, /new RepCycleCounter\(\)/);
  assert.doesNotMatch(source, /\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|MediaRecorder/);
});
