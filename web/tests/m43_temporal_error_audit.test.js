import test from 'node:test';
import assert from 'node:assert/strict';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import {
  analyzeTrainTrace, detailedMatch, FN_KEYS, FP_KEYS, PROTOCOL
} from '../scripts/m43_analyse_train_errors.js';

function trace(angles, bounds = [0,8]) {
  return {
    action: 'squat', fps: 10, total_frames: 50,
    annotated_count: bounds.length / 2,
    cycle_bounds_frames: bounds,
    frames: angles.map((a, i) => ({
      time_ms: i * 200,
      landmarks: a === null ? null : poseFromAngle('squat', a)
    }))
  };
}

test('M43 matched GT indices retain original M37 one-to-one tie ordering', () => {
  const gt = [
    {endMs:1000, toleranceMs:200},
    {endMs:1100, toleranceMs:200}
  ];
  const match = detailedMatch([1050, 1050], gt);
  assert.deepEqual([...match.matchedGroundTruth].sort(), [0,1]);
  assert.deepEqual([...match.matchedPredictions].sort(), [0,1]);
  const occupied = detailedMatch([700,800], [{endMs:750,toleranceMs:100}]);
  assert.equal(occupied.matchedGroundTruth.size, 1);
  assert.equal(occupied.matchedPredictions.size, 1);
});

test('M43 partitions a missed GT event recovered by extra dense pose sample', () => {
  const angles = Array(20).fill(170);
  angles[1] = 100; // 200ms, never included in even-index subsequence
  angles[2] = 140;
  angles[3] = 170; // 600ms return, at least 400ms after the low
  const outcome = analyzeTrainTrace(trace(angles));
  assert.equal(outcome.protocol, PROTOCOL);
  assert.equal(outcome.source_split, 'train');
  assert.equal(outcome.annotated_rep_count, 1);
  assert.equal(outcome.m41.dense_200ms.low_to_high.event_tp, 1);
  assert.equal(outcome.m41.subsample_400ms.low_to_high.event_fn, 1);
  assert.deepEqual(outcome.diagnostics.paired_gt_outcomes, {
    both_matched: 0, dense_only_matched: 1,
    subsample_only_matched: 0, neither_matched: 0
  });
  assert.equal(
    outcome.diagnostics.subsample_400ms.unmatched_gt_window_observations
      .usable_but_no_low_in_gt_window, 1
  );
  assert.equal(
    outcome.diagnostics.dense_200ms.all_gt_windows_with_low_then_high, 1
  );
  assert.equal('frames' in outcome, false);
  assert.equal('cycle_bounds_frames' in outcome, false);
  assert.equal('landmarks' in outcome, false);
  assert.equal('event_times_ms' in outcome.diagnostics.dense_200ms, false);
  assert.equal('_matchedGt' in outcome.diagnostics.dense_200ms, false);
});

test('M43 recognizes both kinds of unmatched FP under frozen M37 tolerance', () => {
  const angles = Array(20).fill(170);
  angles[1] = 100; angles[2] = 140; angles[3] = 170; // TP at 600
  angles[4] = 100; angles[5] = 140; angles[6] = 170; // FP at 1200 inside tolerance
  angles[13] = 100; angles[14] = 140; angles[15] = 170; // distant FP at 3000
  const out = analyzeTrainTrace(trace(angles));
  const dense = out.diagnostics.dense_200ms;
  assert.equal(dense.tp, 1);
  assert.equal(dense.fp, 2);
  assert.equal(dense.fn, 0);
  assert.deepEqual(dense.unmatched_predictions_by_tolerance, {
    unmatched_within_some_gt_end_tolerance: 1,
    unmatched_outside_all_gt_end_tolerances: 1
  });
  assert.equal(FP_KEYS.length, 2);
});

test('M43 exhaustive FN observation categories on half-open publisher windows', () => {
  const cases = [
    {sequence: Array(20).fill(null), key: 'no_usable_angle_in_gt_window'},
    {sequence: Array(20).fill(170), key: 'usable_but_no_low_in_gt_window'},
    {sequence: Array(20).fill(140), key: 'low_but_no_later_high_in_gt_window',
      changes: {1:100}},
    {sequence: Array(20).fill(140), key: 'low_then_high_in_gt_window_still_unmatched',
      changes: {1:100,2:170}}
  ];
  for (const item of cases) {
    const seq = item.sequence.slice();
    for (const [key, value] of Object.entries(item.changes || {})) seq[Number(key)] = value;
    const out = analyzeTrainTrace(trace(seq));
    const d = out.diagnostics.dense_200ms;
    assert.equal(d.fn, 1, item.key);
    assert.equal(d.unmatched_gt_window_observations[item.key], 1, item.key);
    assert.equal(Object.values(d.unmatched_gt_window_observations)
      .reduce((a,b)=>a+b, 0), 1);
  }
  assert.equal(FN_KEYS.length, 4);
});

test('M43 retains null event recall on an actually zero-rep source', () => {
  const out = analyzeTrainTrace(trace(Array(20).fill(170), []));
  assert.equal(out.diagnostics.dense_200ms.gt_events, 0);
  assert.equal(out.diagnostics.dense_200ms.fn, 0);
  assert.deepEqual(out.diagnostics.paired_gt_outcomes, {
    both_matched: 0, dense_only_matched: 0,
    subsample_only_matched: 0, neither_matched: 0
  });
  assert.equal(out.m41.dense_200ms.low_to_high.event_recall, null);
});

test('M43 refuses incorrect source annotations rather than infer event bounds', () => {
  const invalid = trace(Array(20).fill(170));
  invalid.annotated_count = 4;
  assert.throws(() => analyzeTrainTrace(invalid), /count and paired/);
});
