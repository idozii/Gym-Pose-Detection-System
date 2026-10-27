import test from 'node:test';
import assert from 'node:assert/strict';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import { pairedCadenceComparison, PROTOCOL } from '../scripts/m41_compare_train_cadence.js';

function makeTrace(angles, bounds = [0,8]) {
  return {
    action: 'squat',
    fps: 10,
    total_frames: 50,
    annotated_count: bounds.length / 2,
    cycle_bounds_frames: bounds,
    frames: angles.map((angle, i) => ({
      time_ms: i * 200,
      landmarks: angle === null ? null : poseFromAngle('squat',angle)
    }))
  };
}

test('M41 sees a temporal excursion unavailable in paired every-other downsample', () => {
  const angle = Array(20).fill(170);
  angle[1] = 100; // low pose absent from all even-numbered time samples
  const report = pairedCadenceComparison(makeTrace(angle));
  assert.equal(report.protocol, PROTOCOL);
  assert.equal(report.source_split, 'train');
  assert.equal(report.dense_200ms.input_frame_samples, 20);
  assert.equal(report.subsample_400ms.input_frame_samples, 10);
  assert.equal(report.dense_200ms.low_to_high.predicted_count, 1);
  assert.equal(report.subsample_400ms.low_to_high.predicted_count, 0);
  assert.equal(report.dense_200ms.low_to_high.event_tp, 1);
  assert.equal(report.subsample_400ms.low_to_high.event_fn, 1);
  assert.equal(report.dense_200ms.m34.predicted_count, 0);
  assert.equal(report.subsample_400ms.m34.predicted_count, 0);
  assert.equal(report.dense_200ms.always_zero.predicted_count, 0);
  assert.equal(report.design.m34_internal_sample_gate_unchanged_ms, 400);
  assert.equal(report.design.exact_independent_400ms_pose_estimator_parity, false);
  assert.equal('frames' in report, false);
  assert.equal('cycle_bounds_frames' in report, false);
  assert.equal('event_times_ms' in report, false);
  assert.equal('event_times_ms' in report.dense_200ms.low_to_high, false);
});

test('M41 handles publisher zero-rep negative controls without invented recall', () => {
  const report = pairedCadenceComparison(makeTrace(Array(20).fill(170), []));
  for (const name of ['dense_200ms', 'subsample_400ms']) {
    assert.equal(report[name].annotated_rep_count, 0);
    assert.equal(report[name].low_to_high.predicted_count, 0);
    assert.equal(report[name].low_to_high.event_recall, null);
    assert.equal(report[name].low_to_high.event_f1, null);
  }
});

test('M41 rejects too-short or mismatched annotation metadata', () => {
  assert.throws(() => pairedCadenceComparison(makeTrace(Array(15).fill(170))), /16/);
  const wrong = makeTrace(Array(20).fill(170));
  wrong.action = 'ohp';
  assert.throws(() => pairedCadenceComparison(wrong), /paired Squat/);
  const inconsistent = makeTrace(Array(20).fill(170));
  inconsistent.annotated_count = 0;
  assert.throws(() => pairedCadenceComparison(inconsistent), /count and paired/);
});
