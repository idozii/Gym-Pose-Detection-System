import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { explainMovement, sourceEvidenceNotice } from '../src/feedback_explainer.js';

const local = (exercise = 'squat', cycles = 0) => ({
  exercise, tracking: 'visible', cycles, meanDurationSec: 2.25
});
const motion = (exercise = 'squat') => ({
  exercise, status: 'observing', angle: 117, range: 55
});
const recent = [
  { durationMs: 2100, amplitudeDeg: 57 },
  { durationMs: 2200, amplitudeDeg: 62 },
  { durationMs: 2000, amplitudeDeg: 60 }
];

test('M35 explains every original exercise, excludes Push-up model claims', () => {
  for (const exercise of ['squat', 'ohp', 'barbell_row']) {
    const result = explainMovement(exercise, local(exercise, 3), motion(exercise),
      { recent });
    assert.equal(result.status, 'cycle_comparison');
    assert.match(result.title, /3 movement cycles/);
    assert.match(result.message, /2-2.2 seconds/);
    assert.match(result.detail, /NOT a technique score/);
    assert.equal('correct' in result, false);
    assert.equal('error' in result, false);
  }
  assert.equal(explainMovement('pushup', null, null), null);
  assert.equal(sourceEvidenceNotice('pushup'), null);
});

test('Offline and missing pose abstain and provide camera-specific hints', () => {
  const stopped = explainMovement('ohp', null, null, { cameraRunning: false });
  assert.equal(stopped.status, 'camera_offline');
  assert.match(stopped.message, /No cycle or technique quality/);
  const lost = explainMovement('barbell_row', {
    ...local('barbell_row', 5), tracking: 'not_visible'
  }, { ...motion('barbell_row'), status: 'insufficient_tracking' });
  assert.equal(lost.status, 'tracking_insufficient');
  assert.match(lost.detail, /side view/);
  assert.doesNotMatch(lost.message, /bad technique|danger|injury risk/i);
});

test('No complete cycle is not labeled as an incorrect repetition', () => {
  const result = explainMovement('squat', local('squat'), motion());
  assert.equal(result.status, 'observing');
  assert.match(result.message, /not automatically an incorrect repetition/);
  const one = explainMovement('squat', local('squat', 1), motion(),
    { recent: recent.slice(0, 1) });
  assert.equal(one.status, 'early_cycles');
  assert.match(one.title, /First movement cycle/);
  const two = explainMovement('squat', local('squat', 2), motion(),
    { recent: recent.slice(0, 2) });
  assert.equal(two.status, 'early_cycles');
  assert.match(two.message, /More continuous cycles/);
});

test('M35 describes within-set tempo variability without quality thresholds', () => {
  const consistent = explainMovement('ohp', local('ohp', 3), motion('ohp'),
    { recent });
  assert.match(consistent.message, /relatively similar/);
  assert.match(consistent.message, /Mean projected joint-angle excursion/);
  const varied = explainMovement('ohp', local('ohp', 4), motion('ohp'), {
    recent: [...recent, { durationMs: 4000, amplitudeDeg: 63 }]
  });
  assert.match(varied.message, /durations vary/);
  assert.match(varied.detail, /NOT a technique score or recommended tempo/);
  assert.doesNotMatch(varied.message, /safe|correct form|wrong form/i);
});

test('M35 rejects invalid cycle data and ignores malformed in-memory history', () => {
  const bad = explainMovement('squat', local('squat', NaN), motion());
  assert.equal(bad.status, 'tracking_insufficient');
  const result = explainMovement('squat', local('squat', 3), motion(), {
    recent: [
      { durationMs: -1, amplitudeDeg: 40 },
      { durationMs: Infinity, amplitudeDeg: 40 },
      recent[0], recent[1]
    ]
  });
  assert.equal(result.status, 'early_cycles');
  assert.match(result.message, /More continuous cycles/);
});

test('Research source scores never imply proper form even after explicit consent', () => {
  for (const exercise of ['squat', 'ohp', 'barbell_row']) {
    const off = sourceEvidenceNotice(exercise);
    assert.match(off, /research is OFF/);
    const pending = sourceEvidenceNotice(exercise, { consent: true });
    assert.match(pending, /unavailable or awaiting/);
    const response = {
      status: 'experimental_source_labels_only', exercise, targets: {
        dummy: { status: 'experimental_source_label_score', error_probability: .02 }
      }
    };
    const caption = sourceEvidenceNotice(exercise, {
      consent: true, report: response
    });
    assert.match(caption, /Low source score does NOT prove good form/);
    assert.match(caption, /NOT whether this repetition was correct/);
    assert.doesNotMatch(caption, /great job|safe to lift/i);
    assert.match(sourceEvidenceNotice(exercise, {
      consent: true, report: { ...response, exercise: 'not_same_task' }
    }), /unavailable or awaiting/);
  }
});

test('M35 browser UI remains local and does not modify original source model pipeline', () => {
  const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const module = readFileSync(new URL('../src/feedback_explainer.js', import.meta.url), 'utf8');
  assert.match(source, /id="feedbackExplainer"/);
  assert.match(source, /id="explainSource"/);
  assert.match(source, /explainMovement\(/);
  assert.match(source, /sourceEvidenceNotice\(/);
  assert.match(source, /ui\.feedbackExplainer\.hidden = isPushup/);
  assert.match(source, /recent: state\.repCounter\.recent/);
  assert.match(source, /sixModelReady\(health\)/);
  assert.match(source, /state\.repCounter\.reset\(\)/);
  assert.doesNotMatch(module, /\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|MediaRecorder/);
});
