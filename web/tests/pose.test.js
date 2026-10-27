import test from 'node:test';
import assert from 'node:assert/strict';
import { TASKS, VISIBILITY_THRESHOLD, jointVisibility, readiness, requiredJointIndices, taskReadiness } from '../src/pose.js';

function landmarks(low = []) {
  return Array.from({ length: 33 }, (_, index) => ({
    x: index / 100, y: index / 100, z: 0,
    visibility: low.includes(index) ? .2 : .95
  }));
}

test('four exercises, only three source-label categories, exactly six honest source slots', () => {
  assert.deepEqual(Object.keys(TASKS), ['squat', 'ohp', 'pushup', 'barbell_row']);
  assert.deepEqual(Object.values(TASKS).map((x) => x.targets.length), [2, 2, 0, 2]);
  assert.deepEqual(TASKS.squat.targets.map((t) => t.source),
    ['error_knees_forward', 'error_knees_inward']);
  assert.deepEqual(TASKS.ohp.targets.map((t) => t.source),
    ['error_elbows', 'error_knees']);
  assert.deepEqual(TASKS.barbell_row.targets.map((t) => t.source),
    ['torso_angle', 'lumbar']);
});

test('Squat exact 8 lower-body source-required joints and visible threshold .50', () => {
  assert.equal(VISIBILITY_THRESHOLD, .5);
  assert.deepEqual(requiredJointIndices('squat'), [11, 12, 23, 24, 25, 26, 27, 28]);
  const result = readiness('squat', landmarks([26, 27, 28]));
  assert.equal(result.total, 8);
  assert.equal(result.visibleCount, 5);
  assert.equal(result.basicVisibilityReady, false);
  assert.deepEqual(result.missing, ['Right knee', 'Left ankle', 'Right ankle']);
  assert.equal(jointVisibility(landmarks(), 27), 'VISIBLE');
  assert.equal(jointVisibility(landmarks([27]), 27), 'LOW');
});

test('OHP target-specific visibility can admit elbows but not knees', () => {
  const both = taskReadiness('ohp', landmarks([25, 26, 27, 28]));
  assert.deepEqual(both.map((t) => t.basicVisibilityReady), [true, false]);
  assert.deepEqual(requiredJointIndices('ohp'),
    [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);
});

test('Row torso4 ignores knee and ankle visibility without claiming lumbar measurement', () => {
  assert.deepEqual(requiredJointIndices('barbell_row'), [11, 12, 23, 24]);
  assert.deepEqual(taskReadiness('barbell_row', landmarks([25, 26, 27, 28]))
    .map((t) => t.basicVisibilityReady), [true, true]);
});

test('Push-up is geometry-only and exposes no backend source target', () => {
  assert.deepEqual(TASKS.pushup.targets, []);
  assert.deepEqual(requiredJointIndices('pushup'), [11, 12, 13, 14, 15, 16, 23, 24]);
  assert.deepEqual(taskReadiness('pushup', landmarks()), []);
  assert.equal(readiness('pushup', landmarks()).total, 8);
});

test('Missing, malformed and nonfinite landmarks never look visible', () => {
  assert.equal(jointVisibility(null, 11), 'NO POSE');
  assert.equal(jointVisibility([], 11), 'NO POSE');
  assert.equal(jointVisibility(landmarks().slice(1), 11), 'NO POSE');
  const v = landmarks();
  v[11].visibility = NaN;
  assert.equal(jointVisibility(v, 11), 'UNKNOWN');
  v[11].visibility = .49;
  assert.equal(jointVisibility(v, 11), 'LOW');
  v[11].visibility = .5;
  assert.equal(jointVisibility(v, 11), 'VISIBLE');
  v[11].x = Infinity;
  assert.equal(jointVisibility(v, 11), 'UNKNOWN');
});

test('Unknown exercise is rejected, no silent auto exercise classification', () => {
  assert.throws(() => requiredJointIndices('deadlift'), /Unknown exercise/);
  assert.throws(() => taskReadiness('deadlift', landmarks()), /Unknown exercise/);
});
