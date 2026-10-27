import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EXERCISE_FEEDBACK, ExerciseFeedbackWindow, measureExercise } from '../src/exercise_feedback.js';

function joints() {
  const data = Array.from({ length: 33 }, () => ({
    x: .1, y: .1, z: 0, visibility: .05
  }));
  // Squat left hip-knee-ankle forms 90 degrees in camera plane.
  data[23] = { x: .3, y: .25, z: 0, visibility: .98 };
  data[25] = { x: .3, y: .5, z: 0, visibility: .98 };
  data[27] = { x: .5, y: .5, z: 0, visibility: .98 };
  // OHP left shoulder-elbow-wrist, also 90 degrees.
  data[11] = { x: .3, y: .2, z: 0, visibility: .98 };
  data[13] = { x: .3, y: .4, z: 0, visibility: .98 };
  data[15] = { x: .5, y: .4, z: 0, visibility: .98 };
  // Row requires both shoulders and both hips.
  data[12] = { x: .4, y: .2, z: 0, visibility: .98 };
  data[24] = { x: .4, y: .25, z: 0, visibility: .98 };
  return data;
}

test('M33 defines only three research observation tasks, no source verdicts', () => {
  assert.deepEqual(Object.keys(EXERCISE_FEEDBACK), ['squat', 'ohp', 'barbell_row']);
  for (const policy of Object.values(EXERCISE_FEEDBACK)) {
    assert.match(policy.limitation, /cannot|does not/);
    assert.ok(policy.camera.length > 30);
    assert.equal('safe' in policy, false);
    assert.equal('correct' in policy, false);
  }
  assert.equal(measureExercise('pushup', joints()), null);
  assert.equal(measureExercise('deadlift', joints()), null);
});

test('Squat and OHP 2D joint angles are computed from visible specific joints', () => {
  const squat = measureExercise('squat', joints());
  const ohp = measureExercise('ohp', joints());
  assert.ok(squat && ohp);
  assert.equal(Math.round(squat.degrees), 90);
  assert.equal(Math.round(ohp.degrees), 90);
  assert.equal(squat.side, 'LEFT');
  assert.equal(ohp.side, 'LEFT');
  assert.equal('score' in squat, false);
  assert.equal('verdict' in ohp, false);
});

test('Barbell Row shows camera-plane torso tilt, not lumbar spinal bend', () => {
  const p = joints();
  assert.equal(Math.round(measureExercise('barbell_row', p).degrees), 0);
  p[11].x += .15;
  p[12].x += .15;
  const tilted = measureExercise('barbell_row', p);
  assert.ok(tilted.degrees > 0 && tilted.degrees < 90);
  assert.equal(tilted.side, 'CENTER');
  p[12].visibility = .2;
  assert.equal(measureExercise('barbell_row', p), null);
});

test('Missing, obscured, nonfinite or degenerate geometry always abstains', () => {
  for (const task of ['squat', 'ohp', 'barbell_row']) {
    assert.equal(measureExercise(task, null), null);
    assert.equal(measureExercise(task, []), null);
    assert.equal(measureExercise(task, joints().slice(1)), null);
  }
  let p = joints();
  p[25].visibility = .49;
  assert.equal(measureExercise('squat', p), null);
  p = joints();
  p[13].x = Infinity;
  assert.equal(measureExercise('ohp', p), null);
  p = joints();
  p[13].x = p[15].x = p[11].x;
  p[13].y = p[15].y = p[11].y;
  assert.equal(measureExercise('ohp', p), null);
  p = joints();
  p[11].x = p[12].x = p[23].x = p[24].x = .5;
  p[11].y = p[12].y = p[23].y = p[24].y = .5;
  assert.equal(measureExercise('barbell_row', p), null);
});

test('Rolling window shows observed range only after four valid samples', () => {
  const feedback = new ExerciseFeedbackWindow();
  assert.equal(feedback.observe('squat', null, 0).status, 'insufficient_tracking');
  const p = joints();
  for (const time of [500, 1000, 1500]) {
    const observation = feedback.observe('squat', p, time);
    assert.equal(observation.status, 'collecting');
    assert.equal(observation.range, null);
  }
  p[27].x = .3; p[27].y = .7; // straighten knee to 180 degrees
  const fourth = feedback.observe('squat', p, 2000);
  assert.equal(fourth.status, 'observing');
  assert.ok(fourth.range > 80);
  assert.equal(fourth.samples, 4);
  assert.match(fourth.note, /not a good\/bad score/);
  assert.equal(feedback.observe('squat', null, 3400).status, 'insufficient_tracking');
  assert.equal(feedback.observe('squat', null, 3400).angle, null);
  assert.equal(feedback.observe('squat', null, 3400).range, null);
  assert.equal(feedback.observe('squat', null, 14000).samples, 0);
});

test('Side selection resets history rather than mixing arm/leg angle trajectories', () => {
  const feedback = new ExerciseFeedbackWindow();
  const p = joints();
  for (const time of [0, 500, 1000, 1500]) feedback.observe('ohp', p, time);
  assert.equal(feedback.summary(1500).samples, 4);
  for (const index of [11, 13, 15]) p[index].visibility = .1;
  p[12] = { x: .6, y: .2, z: 0, visibility: .95 };
  p[14] = { x: .6, y: .4, z: 0, visibility: .95 };
  p[16] = { x: .7, y: .5, z: 0, visibility: .95 };
  const current = feedback.observe('ohp', p, 2000);
  assert.equal(current.side, 'RIGHT');
  assert.equal(current.samples, 1);
  assert.equal(current.range, null);
});

test('Exercise selection, camera stop and clock rollback cannot retain old trajectory', () => {
  const feedback = new ExerciseFeedbackWindow();
  const p = joints();
  for (const time of [0, 500, 1000, 1500]) feedback.observe('squat', p, time);
  assert.equal(feedback.summary(1500).samples, 4);
  const changed = feedback.observe('barbell_row', p, 2000);
  assert.equal(changed.exercise, 'barbell_row');
  assert.equal(changed.samples, 1);
  assert.equal(changed.range, null);
  feedback.reset();
  assert.equal(feedback.summary(2500), null);
  feedback.observe('squat', p, 3000);
  assert.equal(feedback.observe('squat', p, 1000).samples, 1);
  assert.throws(() => new ExerciseFeedbackWindow({ windowMs: 0 }), /Invalid geometry timing/);
});

test('M33 frontend feedback is on-device, and original opt-in six-model API stays separate', () => {
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const pose = readFileSync(new URL('../src/pose.js', import.meta.url), 'utf8');
  assert.match(main, /id="exerciseFeedback"/);
  assert.match(main, /id="feedbackAngle"/);
  assert.match(main, /id="feedbackRange"/);
  assert.match(main, /state\.lastExerciseFeedback = state\.exerciseFeedback\.observe/);
  assert.match(main, /ui\.exerciseFeedback\.hidden = isPushup/);
  assert.match(main, /state\.exerciseFeedback\.reset\(\)/);
  assert.match(main, /aria-live="polite"/);
  assert.match(main, /Only source-label research scoring is eligible for the remote API/);
  assert.match(pose, /torso_angle/);
  assert.match(pose, /error_knees_forward/);
  const local = readFileSync(new URL('../src/exercise_feedback.js', import.meta.url), 'utf8');
  assert.doesNotMatch(local, /\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|MediaRecorder/);
});
