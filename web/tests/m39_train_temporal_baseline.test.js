import test from 'node:test';
import assert from 'node:assert/strict';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import { compareTrainTrace, detectLowToHighEvents, FROZEN_RULE,
  M39_PROTOCOL } from '../scripts/m39_compare_train_trace.js';

function source(sequence, bounds, {fps=10, totalFrames=50} = {}) {
  return {
    action: 'squat', fps, total_frames: totalFrames,
    annotated_count: bounds.length / 2,
    cycle_bounds_frames: bounds,
    frames: sequence.map((v,i) => ({
      time_ms: i * 400,
      landmarks: v === null ? null : poseFromAngle('squat', v)
    }))
  };
}

test('M39 compares only immutable same-pose train baselines', () => {
  const input = source([170,170,100,170,170,170,170,170], [0,12]);
  const output = compareTrainTrace(input);
  assert.equal(output.protocol, M39_PROTOCOL);
  assert.equal(output.source_split, 'train');
  assert.equal(output.m34.predicted_count, 0);
  assert.equal(output.low_to_high.predicted_count, 1);
  assert.equal(output.low_to_high.event_tp, 1);
  assert.equal(output.always_zero.predicted_count, 0);
  assert.equal(output.always_zero.event_fn, 1);
  assert.equal(output.always_zero.event_precision, null);
  assert.equal(output.always_zero.event_f1, null);
  assert.equal(output.low_to_high.event_recall, 1);
  assert.equal(FROZEN_RULE.lowDeg, 120);
  assert.equal(FROZEN_RULE.highDeg, 155);
  assert.equal(FROZEN_RULE.minimumExcursionDeg, 35);
  assert.equal('frames' in output, false);
  assert.equal('cycle_bounds_frames' in output, false);
  assert.equal('event_times_ms' in output.low_to_high, false);
});

test('M39 does not count high-only oscillations or repeats while standing', () => {
  const result = compareTrainTrace(source(
    [170,170,160,165,170,155,170,170], []
  ));
  assert.equal(result.low_to_high.predicted_count, 0);
  assert.equal(result.always_zero.absolute_count_error, 0);
  assert.equal(result.low_to_high.event_recall, null);
  assert.equal(result.low_to_high.event_f1, null);
});

test('M39 rejects cross-limb stitched motion and a long loss of geometry', () => {
  const frames = [
    {time_ms:0, landmarks:poseFromAngle('squat',100,{side:'LEFT'})},
    {time_ms:400, landmarks:poseFromAngle('squat',170,{side:'RIGHT'})},
    {time_ms:800, landmarks:poseFromAngle('squat',100,{side:'RIGHT'})},
    {time_ms:1200, landmarks:poseFromAngle('squat',170,{side:'RIGHT'})}
  ];
  const side = detectLowToHighEvents(frames);
  assert.deepEqual(side.event_times_ms, [1200]);
  assert.equal(side.side_resets_pending, 1);
  const gap = detectLowToHighEvents([
    {time_ms:0, landmarks:poseFromAngle('squat',100)},
    {time_ms:1600, landmarks:null},
    {time_ms:2000, landmarks:poseFromAngle('squat',170)}
  ]);
  assert.deepEqual(gap.event_times_ms, []);
  assert.equal(gap.tracking_gap_resets_pending, 1);
});

test('M39 requires M37 label/frame validation before predicting', () => {
  const x = source([170,170,100,170,170,170,170,170], [0,12]);
  x.action = 'barbell_row';
  assert.throws(() => compareTrainTrace(x), /paired Squat/);
  const y = source([170,170,100,170,170,170,170,170], [0,12]);
  y.annotated_count = 2;
  assert.throws(() => compareTrainTrace(y), /count and paired event/);
});
