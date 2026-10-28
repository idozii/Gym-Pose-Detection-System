import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RepCycleCounter } from '../src/rep_counter.js';
import {
  LiveRelativeExcursionCounter, LIVE_M45_RULE, LIVE_M45_SAMPLE_MS,
  isResearchSquat, RESEARCH_DEMO_MODES
} from '../src/live_relative_excursion.js';
import {
  detectRelativeExcursionEvents, RULE as ORIGINAL_M45_RULE
} from '../scripts/m45_compare_train_relative.js';

function pose(angle, side = 'LEFT') {
  if (angle === null) return null;
  const points = Array.from({ length: 33 }, () => ({
    x: .2, y: .2, z: 0, visibility: .1
  }));
  const left = side === 'LEFT';
  const hip = left ? 23 : 24;
  const knee = left ? 25 : 26;
  const ankle = left ? 27 : 28;
  const radians = angle * Math.PI / 180;
  points[hip] = { x: .4, y: .28, z: 0, visibility: .98 };
  points[knee] = { x: .4, y: .5, z: 0, visibility: .98 };
  points[ankle] = {
    x: .4 + .22 * Math.sin(radians),
    y: .5 - .22 * Math.cos(radians),
    z: 0, visibility: .98
  };
  return points;
}

function parity(cases) {
  for (const frames of cases) {
    const reference = detectRelativeExcursionEvents(frames);
    const candidate = new LiveRelativeExcursionCounter();
    const observed = [];
    let n = 0;
    for (const item of frames) {
      const result = candidate.consumeSample(item.landmarks, item.time_ms);
      if (result.cycles > n) {
        assert.equal(result.cycles, n + 1);
        observed.push(item.time_ms);
      }
      n = result.cycles;
    }
    assert.deepEqual(observed, reference.event_times_ms);
    assert.equal(candidate.summary().cycles, reference.event_times_ms.length);
  }
}

test('W2 fixed live rule matches frozen M45 TRAIN comparator exactly', () => {
  assert.deepEqual(LIVE_M45_RULE, ORIGINAL_M45_RULE);
  assert.equal(LIVE_M45_SAMPLE_MS, 200);
});

test('W2 incremental M45 events equal frozen comparator for complete, partial and repeated cycles', () => {
  const sequences = [
    [170, 168, 133, 132, 148, 169],
    [170, 170, 133, 132, 165, 169, 170, 132, 125, 150, 170, 172],
    [170, 134, 110, 120, 135, 145, 170, 160, 160],
    [160, 130, 125, 150, 160, 129, 158],
    [170, 169, 168, 170, 170, 170, 170]
  ];
  parity(sequences.map((s) => s.map((v, i) => ({
    time_ms: 200 * i,
    landmarks: pose(v)
  }))));
});

test('W2 parity holds across missing poses, side changes, gap resets and >12s timeout', () => {
  parity([
    [
      {time_ms:0,landmarks:pose(170)},
      {time_ms:200,landmarks:pose(120)},
      {time_ms:400,landmarks:null},
      {time_ms:1600,landmarks:null},
      {time_ms:1800,landmarks:pose(170)},
      {time_ms:2000,landmarks:pose(120)},
      {time_ms:2400,landmarks:pose(170)}
    ],
    [
      {time_ms:0,landmarks:pose(170,'LEFT')},
      {time_ms:200,landmarks:pose(120,'LEFT')},
      {time_ms:400,landmarks:pose(170,'RIGHT')},
      {time_ms:600,landmarks:pose(125,'RIGHT')},
      {time_ms:1000,landmarks:pose(170,'RIGHT')}
    ],
    [
      {time_ms:0,landmarks:pose(170)},
      {time_ms:200,landmarks:pose(130)},
      {time_ms:13000,landmarks:pose(160)},
      {time_ms:13200,landmarks:pose(130)},
      {time_ms:13600,landmarks:pose(170)}
    ]
  ]);
});

test('W2 live 200ms sampler skips denser frames; M34 original defaults stay unchanged', () => {
  const candidate = new LiveRelativeExcursionCounter();
  const original = new RepCycleCounter();
  const events = [170, 168, 130, 127, 145, 170];
  for (let i = 0; i < events.length; i++) {
    for (let jitter = 0; jitter < 4; jitter++) {
      const t = i * 200 + jitter * 50;
      candidate.observe(pose(events[i]), t);
      original.observe('squat', pose(events[i]), t);
    }
  }
  assert.equal(candidate.summary().sampledFrames, events.length);
  assert.equal(candidate.summary().cycles, 1);
  assert.equal(original.sampleMs, 400);
  assert.equal(original.summary().cycles, 0);
  assert.equal(candidate.events.length, 1);
  assert.ok(candidate.events[0] <= 12000);
});

test('W2 reset, timestamp rollback and memory bounds are privacy-safe', () => {
  const counter = new LiveRelativeExcursionCounter();
  for (let index = 0; index < 15; index++) {
    const t = index * 1200;
    for (const [offset, value] of [[0,170],[200,130],[600,170]]) {
      counter.observe(pose(value),t+offset);
    }
  }
  assert.equal(counter.summary().cycles, 15);
  assert.equal(counter.events.length, 12);
  counter.observe(pose(160), -100);
  assert.equal(counter.summary().cycles, 0);
  assert.equal(counter.events.length, 0);
  counter.reset();
  assert.equal(counter.summary().sampledFrames, 0);
  assert.deepEqual(counter.events, []);
  assert.equal('landmarks' in counter, false);
  assert.throws(() => new LiveRelativeExcursionCounter({sampleMs:190}), />=200ms/);
  assert.throws(() => counter.consumeSample(pose(170),Number.NaN), /chronological/);
});

test('W2 is Squat-only and mode names cannot authorize training scores', () => {
  assert.equal(isResearchSquat('squat'), true);
  for (const e of ['pushup', 'ohp', 'barbell_row', 'unknown']) {
    assert.equal(isResearchSquat(e), false);
  }
  assert.deepEqual(RESEARCH_DEMO_MODES,['compare','m34','m45']);
  const source = readFileSync(new URL('../src/live_relative_excursion.js', import.meta.url),'utf8');
  assert.doesNotMatch(source, /\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|MediaRecorder|indexedDB/);
  assert.doesNotMatch(source, /countIsCorrect|formIsSafe|injuryRisk/);
});

test('W1+W2 site wires same frame to both detectors, supports reset/camera privacy, limits other exercises', () => {
  const code=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const styles=readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  assert.match(code,/id="researchComparison"/);
  assert.match(code,/id="researchMode"/);
  assert.match(code,/id="researchM34Count"/);
  assert.match(code,/id="researchM45Count"/);
  assert.match(code,/state\.lastRepAnalysis = state\.repCounter\.observe/);
  assert.match(code,/state\.lastM45 = state\.liveRelative\.observe/);
  assert.match(code,/ui\.researchComparison\.hidden = !isResearchSquat\(state\.exercise\)/);
  assert.match(code,/state\.liveRelative\.reset\(\)/);
  assert.match(code,/state\.repCounter\.reset\(\)/);
  assert.match(code,/role="status" aria-live="polite"/);
  assert.match(code,/stopCamera\(\);\s*setStatus\('CAMERA DISCONNECTED'/);
  assert.match(styles,/@media \(max-width: 470px\)/);
  assert.match(styles,/\.research-count-card\[hidden\]/);
});
