import test from 'node:test';
import assert from 'node:assert/strict';
import { pushupElbowAngle, PushupMotionWindow } from '../src/pushup.js';
import { readFileSync } from 'node:fs';

function points() {
  const p = Array.from({ length: 33 }, () => ({ x: .1, y: .1, z: 0, visibility: .95 }));
  p[11] = { x: .3, y: .2, z: 0, visibility: .95 }; // left shoulder
  p[13] = { x: .3, y: .4, z: 0, visibility: .95 }; // left elbow
  p[15] = { x: .5, y: .4, z: 0, visibility: .95 }; // left wrist
  p[23] = { x: .3, y: .6, z: 0, visibility: .95 }; // left hip
  for (const j of [12,14,16,24]) p[j].visibility = .05;
  return p;
}

test('valid geometry measures angle only, never form correctness', () => {
  const measured = pushupElbowAngle(points());
  assert.equal(Math.round(measured.degrees), 90);
  assert.equal(measured.side, 'LEFT');
  assert.match(measured.evidence, /camera-plane/);
  assert.equal('correct' in measured, false);
  assert.equal('repCount' in measured, false);
});

test('missing or low-confidence landmarks abstain; never fake geometry', () => {
  assert.equal(pushupElbowAngle(null), null);
  assert.equal(pushupElbowAngle(points().slice(1)), null);
  let p = points();
  p[13].visibility = .1;
  assert.equal(pushupElbowAngle(p), null);
  p = points();
  p[15].x = p[13].x;
  p[15].y = p[13].y;
  assert.equal(pushupElbowAngle(p), null);
  p = points();
  p[11].x = Infinity;
  assert.equal(pushupElbowAngle(p), null);
});

test('10-second local sample window expires and reset removes prior movement', () => {
  const motion = new PushupMotionWindow({ durationMs: 10000, minIntervalMs: 500 });
  assert.equal(motion.observe(null, 0).angle, null);
  motion.observe(points(), 500);
  motion.observe(points(), 1000);
  motion.observe(points(), 1500);
  let x = motion.observe(points(), 2000);
  assert.equal(x.count, 4);
  assert.equal(x.range, 0);
  assert.equal(Math.round(x.angle), 90);
  x = motion.observe(null, 3001);
  assert.equal(x.angle, null);
  x = motion.observe(null, 14000);
  assert.equal(x.count, 0);
  motion.observe(points(), 14500);
  motion.reset();
  assert.equal(motion.summary(14500).count, 0);
});

test('web mode blocks Push-up backend calls and disables remote consent', () => {
  const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(source, /option value="pushup"/);
  assert.match(source, /id="pushupPreview"/);
  assert.match(source, /state\.exercise === 'pushup' \|\| !state\.inferenceEnabled/);
  assert.match(source, /if \(state\.exercise === 'pushup'\) \{\s*disableResearchInference\(\)/);
  assert.match(source, /ui\.inferenceToggle\.disabled = isPushup \|\| !API_URL/);
  assert.match(source, /state\.pushupMotion\.reset\(\)/);
  assert.match(source, /NO QUALITY SCORE/);
});
