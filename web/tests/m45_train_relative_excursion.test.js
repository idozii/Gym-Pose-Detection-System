import test from 'node:test';
import assert from 'node:assert/strict';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import {
  compareTrainRelativeExcursion, detectRelativeExcursionEvents,
  RULE, PROTOCOL
} from '../scripts/m45_compare_train_relative.js';

function trace(angles, bounds = [0, 15]) {
  return {
    action: 'squat', fps: 10, total_frames: 50,
    annotated_count: bounds.length / 2,
    cycle_bounds_frames: bounds,
    frames: angles.map((v, i) => ({
      time_ms: i * 200,
      landmarks: v === null ? null : poseFromAngle('squat', v)
    }))
  };
}

test('M45 nondeployed relative excursion recovers synthetic GT without low <=120deg', () => {
  const seq = Array(20).fill(170);
  seq[1] = 132; // only an odd-index sample shows the trough
  seq[2] = 170; // 400ms subseries sees a high pose here
  seq[3] = 140;
  seq[4] = 169;
  const result = compareTrainRelativeExcursion(trace(seq));
  assert.equal(result.protocol, PROTOCOL);
  assert.equal(result.source_split, 'train');
  assert.equal(result.annotated_rep_count, 1);
  assert.equal(result.original_m44.m43.m41.dense_200ms.low_to_high.event_tp, 0);
  assert.equal(result.conditions.dense_200ms.relative_excursion.event_tp, 1);
  assert.equal(result.conditions.dense_200ms.relative_excursion.event_fp, 0);
  assert.equal(result.conditions.dense_200ms.relative_excursion.predicted_count, 1);
  assert.deepEqual(result.conditions.dense_200ms.paired_gt_vs_original_low_to_high, {
    both_matched: 0, new_only_matched: 1,
    original_only_matched: 0, neither_matched: 0
  });
  const sparse = result.conditions.subsample_400ms;
  assert.equal(sparse.relative_excursion.event_tp, 0);
  assert.equal(result.original_m44.m43.m41.dense_200ms.m34.predicted_count, 0);
  assert.equal(RULE.minDropDeg, 35);
  assert.equal(RULE.highDeg, 155);
  assert.equal(RULE.minReturnMs, 400);
  assert.equal('frames' in result, false);
  assert.equal('landmarks' in result, false);
  assert.equal('event_times_ms' in result.conditions.dense_200ms, false);
});

test('M45 does not falsely count flat poses or movements <35deg', () => {
  const a = detectRelativeExcursionEvents(trace(Array(20).fill(170)).frames);
  assert.deepEqual(a.event_times_ms, []);
  const b = Array(20).fill(170);
  b[1] = 140; b[4] = 170;
  assert.deepEqual(detectRelativeExcursionEvents(trace(b).frames).event_times_ms, []);
  const c = Array(20).fill(145);
  c[1] = 100; c[4] = 145; // never observed a >=155deg peak
  assert.deepEqual(detectRelativeExcursionEvents(trace(c).frames).event_times_ms, []);
});

test('M45 prevents event stitching across tracking gaps and side switches', () => {
  const angles = [
    {time_ms: 0, landmarks: poseFromAngle('squat',170,{side:'LEFT'})},
    {time_ms: 200, landmarks: poseFromAngle('squat',130,{side:'LEFT'})},
    {time_ms: 600, landmarks: poseFromAngle('squat',170,{side:'RIGHT'})},
    {time_ms: 800, landmarks: poseFromAngle('squat',170,{side:'RIGHT'})}
  ];
  const switched = detectRelativeExcursionEvents(angles);
  assert.deepEqual(switched.event_times_ms, []);
  assert.equal(switched.pending_side_resets, 1);
  const gap = detectRelativeExcursionEvents([
    {time_ms: 0, landmarks: poseFromAngle('squat',170)},
    {time_ms: 200, landmarks: poseFromAngle('squat',130)},
    {time_ms: 1700, landmarks: null},
    {time_ms: 2000, landmarks: poseFromAngle('squat',170)}
  ]);
  assert.equal(gap.event_times_ms.length, 0);
  assert.equal(gap.pending_gap_resets, 1);
});

test('M45 retains existing GT matcher and accounts for candidate false positives', () => {
  const angles = Array(20).fill(170);
  angles[1] = 132; angles[4] = 170;
  angles[7] = 130; angles[10] = 170; // second rep after source GT end
  const result = compareTrainRelativeExcursion(trace(angles, [0,9]));
  const d = result.conditions.dense_200ms;
  assert.equal(d.relative_excursion.predicted_count, 2);
  assert.equal(d.relative_excursion.event_tp, 1);
  assert.equal(d.relative_excursion.event_fp, 1);
  assert.equal(
    Object.values(d.candidate_fp_location).reduce((x,y)=>x+y,0), 1
  );
  assert.equal(
    Object.values(d.paired_gt_vs_original_low_to_high).reduce((x,y)=>x+y,0), 1
  );
});

test('M45 does not invent ground truth or metrics on zero-rep synthetic source', () => {
  const result = compareTrainRelativeExcursion(trace(Array(20).fill(170), []));
  assert.equal(result.conditions.dense_200ms.relative_excursion.event_recall, null);
  assert.equal(result.conditions.dense_200ms.relative_excursion.event_f1, null);
  assert.equal(result.conditions.dense_200ms.relative_excursion.predicted_count, 0);
  const corrupted = trace(Array(20).fill(170), [0,15]);
  corrupted.annotated_count = 2;
  assert.throws(() => compareTrainRelativeExcursion(corrupted), /count and paired/);
});
