import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RepCycleCounter, REP_POLICIES, measureCycleAngle } from '../src/rep_counter.js';

function pose(exercise, angle, side = 'LEFT') {
  const pts = Array.from({ length: 33 }, () => ({
    x: .2, y: .2, z: 0, visibility: .1
  }));
  const left = side === 'LEFT';
  const top = exercise === 'squat' ? (left ? 23 : 24) : left ? 11 : 12;
  const center = exercise === 'squat' ? (left ? 25 : 26) : left ? 13 : 14;
  const foot = exercise === 'squat' ? (left ? 27 : 28) : left ? 15 : 16;
  const x = .4;
  const y = .5;
  const r = .22;
  const rad = angle * Math.PI / 180;
  pts[top] = { x, y: y - r, z: 0, visibility: .98 };
  pts[center] = { x, y, z: 0, visibility: .98 };
  pts[foot] = {
    x: x + r * Math.sin(rad),
    y: y - r * Math.cos(rad), z: 0, visibility: .98
  };
  return pts;
}

function runAngles(exercise, degrees, startMs = 0, counter = new RepCycleCounter()) {
  let result;
  for (const [index, angle] of degrees.entries()) {
    result = counter.observe(exercise, angle === null ? null : pose(exercise, angle),
      startMs + index * 400);
  }
  return { counter, result };
}

test('M34 three explicit heuristics count joint cycles, not validated reps', () => {
  assert.deepEqual(Object.keys(REP_POLICIES), ['squat', 'ohp', 'barbell_row']);
  assert.equal(measureCycleAngle('pushup', pose('ohp', 160)), null);
  for (const exercise of Object.keys(REP_POLICIES)) {
    const geometry = measureCycleAngle(exercise, pose(exercise, 160));
    assert.ok(geometry);
    assert.ok(Math.abs(geometry.degrees - 160) < .001);
    assert.equal(geometry.side, 'LEFT');
    assert.match(REP_POLICIES[exercise].hint, /Count is|does not|may be counted|NOT used/);
  }
  assert.match(REP_POLICIES.barbell_row.source, /NOT torso tilt/);
});

test('Complete high-low-high cycles count once and retain tempo/excursion', () => {
  for (const ex of ['squat', 'ohp', 'barbell_row']) {
    const { counter, result } = runAngles(ex,
      [170, 170, 100, 100, 170, 170, 170, 170]);
    assert.equal(result.cycles, 1);
    assert.equal(result.phase, 'await_low');
    assert.ok(result.meanDurationSec > 1.5 && result.meanDurationSec < 4);
    assert.ok(result.meanExcursionDeg >= 69);
    assert.equal(result.recentCycles, 1);
    assert.equal(result.side, 'LEFT');
    assert.match(result.limitation, /NOT verified repetitions/);
    const snapshot = counter.summary();
    assert.equal(snapshot.cycles, 1);
    assert.equal('goodForm' in snapshot, false);
    assert.equal('errorProbability' in snapshot, false);
  }
});

test('Requires two reliable observations at both angle endpoints', () => {
  const highOnly = runAngles('squat', [170, 170, 175, 160, 170, 170]);
  assert.equal(highOnly.result.cycles, 0);
  const spike = runAngles('squat', [170, 170, 100, 175, 170, 170]);
  assert.equal(spike.result.cycles, 0);
  const noReturn = runAngles('squat', [170, 170, 100, 100, 120, 135]);
  assert.equal(noReturn.result.cycles, 0);
  const oneHighReturn = runAngles('squat', [170, 170, 100, 100, 170, 125]);
  assert.equal(oneHighReturn.result.cycles, 0);
});

test('No duplicate cycles during standing still and two real cycles count twice', () => {
  const { result } = runAngles('ohp', [
    170, 170, 100, 100, 170, 170,
    170, 170, 170, 170,
    100, 100, 170, 170
  ]);
  assert.equal(result.cycles, 2);
  assert.equal(result.recentCycles, 2);
  assert.ok(result.meanDurationSec > 1);
});

test('Occlusion, expiry, changing arms and explicit reset do not count phantom cycles', () => {
  const c = new RepCycleCounter();
  runAngles('ohp', [170, 170, 100, 100], 0, c);
  assert.equal(c.summary().phase, 'await_return');
  c.observe('ohp', null, 1600);
  c.observe('ohp', null, 3200); // lost track > 1.25 sec
  assert.equal(c.summary().phase, 'await_high');
  assert.equal(c.summary().cycles, 0);
  c.observe('ohp', pose('ohp', 170), 3600);
  c.observe('ohp', pose('ohp', 170), 4000);
  assert.equal(c.summary().cycles, 0);
  c.observe('ohp', pose('ohp', 100, 'RIGHT'), 4400);
  c.observe('ohp', pose('ohp', 100, 'RIGHT'), 4800);
  c.observe('ohp', pose('ohp', 170, 'RIGHT'), 5200);
  c.observe('ohp', pose('ohp', 170, 'RIGHT'), 5600);
  assert.equal(c.summary().cycles, 0);
  c.reset();
  assert.equal(c.summary(), null);
});

test('Clock rollback / changing exercise clears previous set', () => {
  const { counter } = runAngles('squat', [170, 170, 100, 100, 170, 170]);
  assert.equal(counter.summary().cycles, 1);
  counter.observe('ohp', pose('ohp', 170), 3000);
  assert.equal(counter.summary().cycles, 0);
  assert.equal(counter.summary().exercise, 'ohp');
  runAngles('ohp', [170, 170, 100, 100, 170, 170], 3400, counter);
  assert.equal(counter.summary().cycles, 1);
  counter.observe('ohp', pose('ohp', 170), 1000);
  assert.equal(counter.summary().cycles, 0);
  assert.equal(counter.summary().phase, 'await_high');
});

test('Counters abstain with missing, low-confidence and degenerate landmarks', () => {
  const c = new RepCycleCounter();
  const p = pose('barbell_row', 170);
  p[13].visibility = .2;
  assert.equal(measureCycleAngle('barbell_row', p), null);
  assert.equal(measureCycleAngle('squat', []), null);
  assert.equal(measureCycleAngle('ohp', null), null);
  const first = c.observe('barbell_row', p, 0);
  assert.equal(first.tracking, 'not_visible');
  assert.equal(first.cycles, 0);
  const p2 = pose('ohp', 170);
  p2[15].x = Infinity;
  assert.equal(measureCycleAngle('ohp', p2), null);
  assert.throws(() => new RepCycleCounter({ sampleMs: 0 }), /Invalid M34/);
});

test('Temporary camera freezing rejects incomplete cycles, preserves counted history', () => {
  const c = new RepCycleCounter();
  runAngles('squat', [170, 170, 100, 100, 170, 170], 0, c);
  assert.equal(c.summary().cycles, 1);
  c.observe('squat', null, 10000);
  assert.equal(c.summary().cycles, 1);
  assert.equal(c.summary().phase, 'await_high');
  assert.equal(c.summary().tracking, 'not_visible');
});

test('Only a fixed bounded number of cycle durations is held in local memory', () => {
  const c = new RepCycleCounter({ maxRecentCycles: 2 });
  const sequence = [170, 170, 100, 100, 170, 170,
    100, 100, 170, 170, 100, 100, 170, 170];
  const { result } = runAngles('squat', sequence, 0, c);
  assert.equal(result.cycles, 3);
  assert.equal(result.recentCycles, 2);
  assert.equal(c.recent.length, 2);
});

test('Web integrates local cycle counter; original API and Push-up behavior unchanged', () => {
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const moduleCode = readFileSync(new URL('../src/rep_counter.js', import.meta.url), 'utf8');
  assert.match(main, /id="repAnalysis"/);
  assert.match(main, /id="repCount"/);
  assert.match(main, /id="resetSet"/);
  assert.match(main, /state\.lastRepAnalysis = state\.repCounter\.observe/);
  assert.match(main, /state\.repCounter\.reset\(\)/);
  assert.match(main, /ui\.repAnalysis\.hidden = isPushup/);
  assert.match(main, /state\.exercise === 'pushup' \|\| !state\.inferenceEnabled/);
  assert.match(main, /state\.pushupMotion\.observe/);
  assert.match(main, /sixModelReady\(health\)/);
  assert.doesNotMatch(moduleCode, /\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|MediaRecorder/);
});
