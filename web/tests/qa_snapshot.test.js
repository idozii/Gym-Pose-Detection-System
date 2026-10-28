import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RuntimeHealth } from '../src/runtime_health.js';
import { createQaSnapshot, QA_PROTOCOL, MIN_QA_FRAMES } from '../src/qa_snapshot.js';

function freshSummary() {
  const w=new RuntimeHealth();
  for(let i=0;i<40;i++)w.observe({wallTimeMs:1000+i*100,poseDurationMs:10+(i%7)});
  return w.summary();
}
test('W7b reports minimum 30 real processed frames as anonymous local JSON',()=>{
  const data=createQaSnapshot(freshSummary(),{sourceMode:'camera',exercise:'squat'});
  assert.equal(MIN_QA_FRAMES,30);
  assert.equal(data.protocol,QA_PROTOCOL);
  assert.equal(data.recent_processed_frames,40);
  assert.equal(data.session_processed_frames,40);
  assert.equal(data.source_type,'camera');
  assert.equal(data.exercise_type,'squat');
  assert.equal(data.pose_inference_median_ms,13);
  assert.equal(data.analyzed_frames_per_wall_second,10);
  assert.equal(data.network_upload_performed,false);
  assert.equal(data.raw_video_or_landmarks_saved,false);
  assert.equal(data.device_or_browser_fingerprint_saved,false);
  assert.doesNotMatch(JSON.stringify(data),/filename|timestamp|cameraStream|landmark_list|location|userAgent|deviceMemory|clipName/i);
});
test('W7b rejects unobserved, corrupted or wrong-source metrics',()=>{
  const s=freshSummary();
  for(const item of [
    {...s,windowFrames:29},
    {...s,medianPoseMs:NaN},
    {...s,p95PoseMs:-1},
    {...s,throughputFps:Infinity},
    {...s,observedGapsOver750ms:40},
    {...s,totalFrames:4}
  ])assert.throws(()=>createQaSnapshot(item,{sourceMode:'camera',exercise:'squat'}));
  assert.throws(()=>createQaSnapshot(s,{sourceMode:'upload',exercise:'squat'}));
  assert.throws(()=>createQaSnapshot(s,{sourceMode:'camera',exercise:'dance'}));
  assert.equal(createQaSnapshot(s,{sourceMode:'video',exercise:'row'}).source_type,'video');
});
test('W7 recent window does not carry permanent old startup stalls',()=>{
  const w=new RuntimeHealth({maxSamples:10});
  w.observe({wallTimeMs:0,poseDurationMs:5});
  w.observe({wallTimeMs:1000,poseDurationMs:5});
  assert.equal(w.summary().observedGapsOver750ms,1);
  for(let i=1;i<=20;i++)w.observe({wallTimeMs:1000+i*100,poseDurationMs:10});
  assert.equal(w.summary().windowFrames,10);
  assert.equal(w.summary().observedGapsOver750ms,0);
  assert.notEqual(w.summary().status,'PROCESSING GAP OBSERVED');
});
test('W7b frontend capture is opt-in, no camera start, local textarea, clears on reset',()=>{
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const css=readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  for(const id of ['qaCapture','qaReport','qaCaptureHint','qaReportLabel'])
    assert.match(main,new RegExp('id="'+id+'"'));
  assert.match(main,/ui\.qaCapture\.addEventListener\('click'/);
  assert.match(main,/createQaSnapshot\(state\.runtimeHealth\.summary\(\)/);
  assert.match(main,/ui\.qaReport\.value = JSON\.stringify\(summary, null, 2\)/);
  assert.match(main,/ui\.qaCapture\.disabled = !state\.running/);
  assert.match(main,/ui\.qaReport\.value = ''/);
  assert.match(main,/ui\.resetSet\.addEventListener\('click', \(\) => \{\s*state\.runtimeHealth\.reset\(\);\s*clearQaSnapshot\(\)/);
  assert.match(main,/ui\.exercise\.addEventListener\('change', \(\) => \{\s*if \(!TASKS\[ui\.exercise\.value\]\) return;\s*state\.runtimeHealth\.reset\(\);\s*clearQaSnapshot\(\)/);
  assert.match(css,/\.w7-qa-actions/);
  assert.match(css,/\.w7-qa-report-label\[hidden\]/);
});
test('W7b helper never uploads, uses cookies, or fingerprints device',()=>{
  const src=readFileSync(new URL('../src/qa_snapshot.js',import.meta.url),'utf8');
  const active=src.replace(/\/\*[\s\S]*?\*\//g,'');
  assert.doesNotMatch(active,/\bfetch\s*\(|XMLHttpRequest|sendBeacon|localStorage|sessionStorage|indexedDB|navigator\.|document\.cookie|getUserMedia|MediaRecorder|performance\.getEntries/);
});
