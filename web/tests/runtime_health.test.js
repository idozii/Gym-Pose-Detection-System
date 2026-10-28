import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RuntimeHealth, percentile, describePerformance, W7_WINDOW } from '../src/runtime_health.js';

test('W7 bounded pose latency percentiles are correct from synthetic data',()=>{
  assert.equal(percentile([2,3,5,7,10],.5),5);
  assert.ok(Math.abs(percentile([2,3,5,7,10],.95)-9.4)<1e-10);
  assert.equal(percentile([], .95),null);
  assert.equal(percentile([2,1],.95),null);
  assert.equal(percentile([1,3],Number.NaN),null);
  assert.equal(percentile([-1,1],.5),null);
  assert.equal(W7_WINDOW,120);
});

test('W7 telemetry counts processed frames per WALL second and bounded latency window',()=>{
  const tracker = new RuntimeHealth({maxSamples:20});
  assert.equal(tracker.summary().status,'WAITING FOR FRAMES');
  for(let i=0;i<25;i++){
    tracker.observe({wallTimeMs:1000+i*100,poseDurationMs:10+(i%5)});
  }
  const stats=tracker.summary();
  assert.equal(stats.totalFrames,25);
  assert.equal(stats.windowFrames,20);
  assert.ok(Math.abs(stats.throughputFps-10)<1e-9);
  assert.equal(stats.observedGapsOver750ms,0);
  assert.equal(stats.medianPoseMs,12);
  assert.equal(stats.maxPoseMs,14);
  assert.ok(stats.p95PoseMs<=14);
  tracker.reset();
  assert.equal(tracker.summary().windowFrames,0);
  assert.equal(tracker.summary().totalFrames,0);
  assert.equal(tracker.summary().throughputFps,null);
  assert.equal(tracker.summary().medianPoseMs,null);
});

test('W7 gaps reflect only observed pause/decode gaps, not inferred frame loss',()=>{
  const t=new RuntimeHealth();
  t.observe({wallTimeMs:10,poseDurationMs:15});
  t.observe({wallTimeMs:110,poseDurationMs:18});
  t.observe({wallTimeMs:1110,poseDurationMs:20});
  assert.equal(t.summary().observedGapsOver750ms,1);
  assert.equal(t.summary().status,'PROCESSING GAP OBSERVED');
  assert.match(t.summary().interpretation,/not sensor FPS/);
  assert.throws(()=>t.observe({wallTimeMs:1100,poseDurationMs:2}),/monotonic/);
  assert.throws(()=>t.observe({wallTimeMs:1500,poseDurationMs:-1}),/finite/);
  assert.throws(()=>new RuntimeHealth({maxSamples:200}),/bounded/);
});

test('W7 session health clearly labels slow inference and throughput as diagnostics',()=>{
  assert.equal(describePerformance({samples:30,p95Ms:99,rate:25,stalls:0}),
    'HIGH POSE LATENCY');
  assert.equal(describePerformance({samples:30,p95Ms:20,rate:3,stalls:0}),
    'LOW THROUGHPUT');
  assert.equal(describePerformance({samples:3,p95Ms:99,rate:3,stalls:0}),
    'HIGH POSE LATENCY');
  assert.equal(describePerformance({samples:3,p95Ms:20,rate:3,stalls:0}),
    'OBSERVING');
});

test('W7 main app observes model CPU duration on-device, resets on camera/clip lifecycle',()=>{
  const code=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const css=readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  for(const id of ['runtimeHealthPanel','runtimeHealthStatus','runtimeMedian',
    'runtimeP95','runtimeRate','runtimeGaps']) {
    assert.match(code,new RegExp('id="'+id+'"'));
  }
  assert.match(code,/const poseStart = performance\.now\(\)/);
  assert.match(code,/const poseEnd = performance\.now\(\)/);
  assert.match(code,/state\.runtimeHealth\.observe\(\{/);
  assert.match(code,/wallTimeMs: poseEnd, poseDurationMs: poseEnd - poseStart/);
  assert.match(code,/state\.runtimeHealth\.reset\(\)/);
  assert.match(code,/function updateRuntimeHealth\(\)/);
  assert.match(code,/updateRuntimeHealth\(\);\s*const data = readiness/);
  assert.match(css,/\.runtime-health-grid/);
  assert.match(css,/@media \(max-width: 460px\)/);
});

test('W7 has no upload, persisted pose, user/device fingerprint or video record',()=>{
  const model=readFileSync(new URL('../src/runtime_health.js',import.meta.url),'utf8');
  const executable=model.replace(/\/\*[\s\S]*?\*\//g,'');
  assert.doesNotMatch(executable,/\bfetch\s*\(|XMLHttpRequest|navigator\.sendBeacon|localStorage|sessionStorage|indexedDB|MediaRecorder|getUserMedia|document\.cookie|navigator\.userAgent|deviceMemory/);
  const health=new RuntimeHealth();
  health.observe({wallTimeMs:1000,poseDurationMs:17});
  assert.equal(JSON.stringify(health.summary()).includes('landmarks'),false);
  assert.equal(JSON.stringify(health.summary()).includes('filename'),false);
});
