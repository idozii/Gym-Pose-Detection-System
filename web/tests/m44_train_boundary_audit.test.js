import test from 'node:test';
import assert from 'node:assert/strict';
import { poseFromAngle } from '../scripts/m36_auto_cycle_audit.js';
import {
  analyzeTrainBoundaryTrace, ANGLE_BIN_KEYS, CONTEXT_KEYS,
  FP_LOCATION_KEYS, PROTOCOL, CONTEXT_MS
} from '../scripts/m44_train_boundary_audit.js';

function trace(angles, bounds=[5,14]) {
  return {
    action:'squat', fps:10, total_frames:50,
    annotated_count:bounds.length / 2,
    cycle_bounds_frames:bounds,
    frames:angles.map((a,i)=>({
      time_ms:i*200,
      landmarks:a===null ? null : poseFromAngle('squat',a)
    }))
  };
}

test('M44 detects low before source GT start without inventing in-window low', () => {
  const angles=Array(20).fill(170);
  angles[1]=100; // 200ms: low BEFORE GT [500ms, 1400ms)
  angles[2]=140; // 400ms
  angles[3]=170; // 600ms; predicted event; 800ms from GT end >750 tolerance
  const result=analyzeTrainBoundaryTrace(trace(angles));
  assert.equal(result.protocol,PROTOCOL);
  assert.equal(result.source_split,'train');
  assert.equal(result.context_ms,CONTEXT_MS);
  const d=result.conditions.dense_200ms;
  assert.equal(result.m43.diagnostics.dense_200ms.fn,1);
  assert.equal(result.m43.diagnostics.dense_200ms.fp,1);
  assert.equal(d.unmatched_gt_no_low_inside,1);
  assert.equal(d.min_angle_bins_for_no_low_gt.min_gt_angle_above165,1);
  assert.equal(d.low_in_750ms_context_for_no_low_gt.low_before_only,1);
  assert.equal(d.no_low_gt_with_usable_angle_before,1);
  assert.equal(d.no_low_gt_with_usable_angle_after,1);
  assert.equal(d.unmatched_prediction_location.unmatched_inside_any_gt_interval,1);
  assert.equal(d.unmatched_prediction_location.unmatched_outside_all_gt_intervals,0);
  const s=result.conditions.subsample_400ms;
  assert.equal(s.unmatched_gt_no_low_inside,1);
  assert.equal(s.low_in_750ms_context_for_no_low_gt.low_neither_side,1);
  assert.equal(s.unmatched_prediction_location.unmatched_inside_any_gt_interval,0);
  assert.equal('frames' in result,false);
  assert.equal('cycle_bounds_frames' in result,false);
  assert.equal('event_times_ms' in result,false);
  assert.equal('landmarks' in result,false);
});

test('M44 min angle diagnostic intervals are exhaustive and do not alter detector', () => {
  const checks=[
    [130,'min_gt_angle_gt120_to135'],
    [140,'min_gt_angle_gt135_to150'],
    [160,'min_gt_angle_gt150_to165'],
    [170,'min_gt_angle_above165']
  ];
  for (const [degrees,key] of checks) {
    const result=analyzeTrainBoundaryTrace(trace(Array(20).fill(degrees)));
    assert.equal(result.m43.m41.dense_200ms.low_to_high.predicted_count,0);
    assert.equal(result.conditions.dense_200ms.unmatched_gt_no_low_inside,1);
    assert.equal(result.conditions.dense_200ms.min_angle_bins_for_no_low_gt[key],1);
    assert.equal(Object.values(result.conditions.dense_200ms
      .min_angle_bins_for_no_low_gt).reduce((a,v)=>a+v,0),1);
    assert.equal(result.conditions.dense_200ms
      .low_in_750ms_context_for_no_low_gt.low_neither_side,1);
  }
  assert.equal(ANGLE_BIN_KEYS.length,4);
  assert.equal(CONTEXT_KEYS.length,4);
  assert.equal(FP_LOCATION_KEYS.length,2);
});

test('M44 low on both sides is not misclassified as low inside window', () => {
  const seq=Array(20).fill(140);
  seq[1]=100; // outside before
  seq[8]=100; // outside after end=1400ms
  const result=analyzeTrainBoundaryTrace(trace(seq));
  assert.equal(result.conditions.dense_200ms.unmatched_gt_no_low_inside,1);
  assert.equal(result.conditions.dense_200ms
    .low_in_750ms_context_for_no_low_gt.low_both_sides,1);
  assert.equal(result.m43.diagnostics.dense_200ms.fn,1);
});

test('M44 retains zero rep null recall, rejects invalid annotations', () => {
  const zero=analyzeTrainBoundaryTrace(trace(Array(20).fill(170),[]));
  assert.equal(zero.annotated_rep_count,0);
  assert.equal(zero.conditions.dense_200ms.unmatched_gt_no_low_inside,0);
  assert.equal(zero.m43.m41.dense_200ms.low_to_high.event_recall,null);
  const mismatch=trace(Array(20).fill(170));
  mismatch.annotated_count=2;
  assert.throws(()=>analyzeTrainBoundaryTrace(mismatch),/count and paired/);
});
