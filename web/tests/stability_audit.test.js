import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {
  StabilityAudit,createStabilitySnapshot,W8_PROTOCOL,
  W8_FRAMES_PER_WINDOW,W8_RETAINED_WINDOWS
} from '../src/stability_audit.js';

const observe=(a,n,{start=0,step=20,ms=10}={})=>{
  for(let i=0;i<n;i++)a.observe({wallTimeMs:start+i*step,poseDurationMs:ms});
};

test('W8 multi-window stable session produces aggregate ranges after 240 frames',()=>{
 const a=new StabilityAudit();
 observe(a,240);
 const s=a.summary();
 assert.equal(s.completedWindows,2);
 assert.equal(s.sessionFrames,240);
 assert.equal(s.retainedWindows,2);
 assert.deepEqual(s.medianRangeMs,[10,10]);
 assert.deepEqual(s.p95RangeMs,[10,10]);
 assert.equal(s.sessionObservedGapsOver750ms,0);
 const data=createStabilitySnapshot(s,{sourceMode:'camera',exercise:'ohp'});
 assert.equal(data.protocol,W8_PROTOCOL);
 assert.equal(data.completed_120_frame_windows,2);
 assert.deepEqual(data.analyzed_throughput_range_fps,[50,50]);
 assert.equal(data.network_upload_performed,false);
});

test('W8 retains at most 16 completed windows; full-session counts do not disappear',()=>{
 const a=new StabilityAudit();
 observe(a,120*22);
 const s=a.summary();
 assert.equal(s.completedWindows,22);
 assert.equal(s.retainedWindows,W8_RETAINED_WINDOWS);
 assert.equal(s.sessionFrames,2640);
 assert.equal(a.windows.length,16);
 assert.equal(a.windowDurations.length,0);
 assert.equal(W8_FRAMES_PER_WINDOW,120);
});

test('W8 reports cross-window pauses even when W7 recent window forgets them',()=>{
 const a=new StabilityAudit();
 observe(a,120,{start:0});
 observe(a,120,{start:4000});
 const s=a.summary();
 assert.equal(s.completedWindows,2);
 assert.equal(s.sessionObservedGapsOver750ms,1);
 assert.equal(s.retainedWindowGaps,1);
 a.reset();
 assert.equal(a.summary().sessionFrames,0);
 assert.equal(a.summary().completedWindows,0);
 assert.equal(a.summary().sessionObservedGapsOver750ms,0);
});

test('W8 fails closed on insufficient frames, corrupt metrics and source types',()=>{
 const a=new StabilityAudit();
 observe(a,239);
 assert.throws(()=>createStabilitySnapshot(a.summary(),{sourceMode:'camera',exercise:'squat'}));
 a.observe({wallTimeMs:239*20,poseDurationMs:10});
 const s=a.summary();
 for (const bad of [
  {...s,p95RangeMs:[NaN,11]},
  {...s,medianRangeMs:[12,10]},
  {...s,throughputRangeFps:[Infinity,20]},
  {...s,retainedWindows:17},
  {...s,completedWindows:1},
  {...s,sessionObservedGapsOver750ms:-1}
 ]) assert.throws(()=>createStabilitySnapshot(bad,{sourceMode:'camera',exercise:'squat'}));
 assert.throws(()=>createStabilitySnapshot(s,{sourceMode:'upload',exercise:'squat'}));
 assert.throws(()=>createStabilitySnapshot(s,{sourceMode:'camera',exercise:'unknown'}));
 assert.equal(createStabilitySnapshot(s,{sourceMode:'video',exercise:'barbell_row'}).source_type,'video');
 assert.throws(()=>a.observe({wallTimeMs:1,poseDurationMs:10}));
 assert.throws(()=>a.observe({wallTimeMs:9999,poseDurationMs:NaN}));
});

test('W8 UI capture is opt-in, lifecycle reset and local only',()=>{
 const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
 const src=readFileSync(new URL('../src/stability_audit.js',import.meta.url),'utf8');
 for(const id of ['w8Capture','w8Report','w8ReportLabel','w8Hint'])
   assert.match(main,new RegExp('id="'+id+'"'));
 assert.match(main,/state\.stabilityAudit\.observe\(\{\s*wallTimeMs: poseEnd/);
 assert.match(main,/state\.stabilityAudit\.reset\(\)/);
 assert.match(main,/ui\.w8Capture\.addEventListener\('click'/);
 assert.match(main,/createStabilitySnapshot\(state\.stabilityAudit\.summary\(\)/);
 assert.match(main,/ui\.w8Report\.value = ''/);
 assert.doesNotMatch(src.replace(/\/\*[\s\S]*?\*\//g,''),/\bfetch\s*\(|XMLHttpRequest|localStorage|sessionStorage|indexedDB|navigator\.|document\.cookie|MediaRecorder/);
});
