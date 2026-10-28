import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { cameraFrameIsChronological } from '../src/camera_frame_clock.js';
import { SessionDashboard } from '../src/session_dashboard.js';
import { RuntimeHealth } from '../src/runtime_health.js';

const ready = (frameTimestampMs, sessionStartedAtMs, lastAnalyzedFrameMs = null) =>
  cameraFrameIsChronological({
    frameTimestampMs, sessionStartedAtMs, lastAnalyzedFrameMs
  });

test('W7 camera: stale rAF timestamp does not kill newly started W3 session', () => {
  const w3 = new SessionDashboard();
  const w7 = new RuntimeHealth();
  w3.start(1000.25); // performance.now() inside an already active paint frame
  let last = null;
  for (const frame of [1000, 1000.2, 1000.25, 1016.67, 1016.67, 1033.34]) {
    if (!ready(frame, w3.startedAtMs, last)) continue;
    w3.observe(frame, {poseSeen:true, requiredReady:true});
    w7.observe({wallTimeMs: frame, poseDurationMs: 5});
    last = frame;
  }
  assert.equal(w3.summary(1040).sampleFrames,3);
  assert.equal(w7.summary().windowFrames,3);
  assert.equal(w3.summary(1040).status,'JOINTS VISIBLE');
});

test('W7 camera: equal ticks, backwards clocks and non-finite rAF times are skipped', () => {
  assert.equal(ready(500, 501),false);
  assert.equal(ready(501, 501),true);
  assert.equal(ready(512, 500, 512),false);
  assert.equal(ready(511, 500, 512),false);
  assert.equal(ready(513, 500, 512),true);
  for(const clock of [NaN, Infinity, -Infinity, -1])
    assert.equal(ready(clock, 500, null),false);
  assert.equal(ready(600, NaN),false);
  assert.equal(ready(600, -1),false);
  assert.equal(ready(600, 500, NaN),false);
});

test('W7 camera: reset requires post-reset rAF frame, never rewinds W3', () => {
  const s = new SessionDashboard();
  s.start(100);
  s.observe(120, {poseSeen:false,requiredReady:false});
  s.start(300.4); // exercise change or Reset set between animation frames
  assert.equal(ready(300, s.startedAtMs, s.lastFrameMs),false);
  assert.equal(ready(300.4, s.startedAtMs, s.lastFrameMs),true);
  s.observe(300.4, {poseSeen:true,requiredReady:true});
  assert.equal(ready(300.4, s.startedAtMs, s.lastFrameMs),false);
  assert.equal(ready(316, s.startedAtMs, s.lastFrameMs),true);
  assert.equal(s.summary(317).sampleFrames,1);
});

test('W7 wiring gates the camera before detectForVideo, leaves local media gate intact', () => {
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(main,/import \{ cameraFrameIsChronological \} from '\.\/camera_frame_clock\.js'/);
  const loop=main.slice(main.indexOf('function loop(token, timestamp)'),
    main.indexOf('function stopCamera()'));
  assert.match(loop,/const frameMs = local \? localVideoTimestampMs\(v\.currentTime \|\| 0\) : timestamp/);
  assert.match(loop,/const cameraFrameReady = local \|\| cameraFrameIsChronological\(/);
  assert.match(loop,/sessionStartedAtMs: state\.sessionDashboard\.startedAtMs/);
  assert.match(loop,/lastAnalyzedFrameMs: state\.sessionDashboard\.lastFrameMs/);
  assert.match(loop,/if \(cameraFrameReady &&[\s\S]*?state\.landmarker\.detectForVideo\(v, frameMs\)/);
  assert.match(loop,/localFrameIsNew\(\{/);
  assert.match(loop,/state\.sessionDashboard\.observe\(frameMs/);
  const guard=readFileSync(new URL('../src/camera_frame_clock.js',import.meta.url),'utf8');
  const active=guard.replace(/\/\*[\s\S]*?\*\//g,'');
  assert.doesNotMatch(active,/fetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|navigator\.|document\.|performance\.now\s*\(/);
});
