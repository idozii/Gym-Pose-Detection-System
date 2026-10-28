import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  SessionDashboard, clockLabel, percentage, tempoBars, tempoFromM34,
  tempoFromM45, W3_MAX_TEMPO_BARS, W3_STALE_FRAME_MS
} from '../src/session_dashboard.js';

test('W3 correctly formats 00:00 to HH:MM:SS with stable seconds', () => {
  assert.equal(clockLabel(0),'00:00');
  assert.equal(clockLabel(999),'00:00');
  assert.equal(clockLabel(1000),'00:01');
  assert.equal(clockLabel(60999),'01:00');
  assert.equal(clockLabel(3599999),'59:59');
  assert.equal(clockLabel(3600000),'01:00:00');
  assert.equal(clockLabel(90061000),'25:01:01');
  assert.equal(clockLabel(Number.NaN),'00:00');
  assert.equal(clockLabel(-1),'00:00');
});

test('W3 visibility coverage counts observed analyzed frames, not classifier confidence', () => {
  const s=new SessionDashboard();
  assert.equal(s.summary(100).running,false);
  assert.equal(s.summary(100).status,'CAMERA OFFLINE');
  assert.equal(s.summary(100).poseSeenPct,null);
  s.start(100);
  assert.equal(s.summary(100).status,'WAITING FOR VIDEO');
  assert.equal(s.summary(100).elapsedLabel,'00:00');
  s.observe(120,{poseSeen:true,requiredReady:true});
  assert.equal(s.summary(150).status,'JOINTS VISIBLE');
  s.observe(160,{poseSeen:true,requiredReady:false});
  assert.equal(s.summary(180).status,'LIMITED JOINT VIEW');
  s.observe(200,{poseSeen:false,requiredReady:false});
  const current=s.summary(1300);
  assert.equal(current.elapsedLabel,'00:01');
  assert.equal(current.sampleFrames,3);
  assert.equal(current.poseSeenPct,67);
  assert.equal(current.requiredReadyPct,33);
  assert.equal(current.status,'NO POSE VISIBLE');
  assert.equal(s.summary(200+W3_STALE_FRAME_MS+1).status,'VIDEO STALLED');
  assert.equal(W3_STALE_FRAME_MS,1500);
  assert.deepEqual(Object.keys(current).sort(),[
    'elapsedLabel','elapsedMs','poseSeenPct','requiredReadyPct',
    'running','sampleFrames','status'
  ].sort());
});

test('W3 reset and restart discard counters, elapsed time and frame observations', () => {
  const s=new SessionDashboard();
  s.start(1000);
  s.observe(1001,{poseSeen:true,requiredReady:true});
  assert.equal(s.summary(5000).elapsedLabel,'00:04');
  s.start(5000);
  assert.equal(s.summary(5000).sampleFrames,0);
  assert.equal(s.summary(5000).poseSeenPct,null);
  assert.equal(s.summary(5000).elapsedLabel,'00:00');
  s.observe(5200,{poseSeen:false,requiredReady:false});
  assert.equal(s.summary(5500).status,'NO POSE VISIBLE');
  s.reset();
  assert.equal(s.summary(9000).running,false);
  assert.equal(s.summary(9000).sampleFrames,0);
  assert.equal(s.summary(9000).elapsedLabel,'00:00');
});

test('W3 resists invalid clock order and fake required-joint readiness', () => {
  const s=new SessionDashboard();
  assert.throws(()=>s.start(-1), /finite monotone clock/);
  s.start(300);
  s.observe(400,{poseSeen:true,requiredReady:true});
  assert.throws(()=>s.observe(400,{poseSeen:true,requiredReady:true}), /chronological/);
  assert.throws(()=>s.observe(250,{poseSeen:true,requiredReady:true}), /chronological/);
  assert.throws(()=>s.observe(500,{poseSeen:false,requiredReady:true}), /honest pose/);
  assert.throws(()=>s.observe(500,{poseSeen:true,requiredReady:'yes'}), /honest pose/);
  assert.equal(s.summary(600).sampleFrames,1);
  assert.throws(()=>percentage(2,1), /nonnegative/);
  assert.throws(()=>percentage(-1,10), /nonnegative/);
  assert.equal(percentage(0,0),null);
  assert.equal(percentage(2,3),67);
});

test('W3 builds bounded relative tempo bars, filters invalid data, and keeps original source units distinct', () => {
  const m34=tempoFromM34([
    {durationMs:900},{durationMs:1200},{durationMs:0},
    {durationMs:2100},{durationMs:NaN},null
  ]);
  const m45=tempoFromM45([350,580,900,NaN,0]);
  assert.deepEqual(m34,[.9,1.2,2.1]);
  assert.deepEqual(m45,[.35,.58,.9]);
  const a=tempoBars(m34);
  assert.deepEqual(a.map(x=>x.durationSec),m34);
  assert.equal(a[2].heightPercent,100);
  assert.equal(a[0].heightPercent,43);
  assert.equal(W3_MAX_TEMPO_BARS,12);
  assert.deepEqual(tempoBars([]),[]);
  assert.deepEqual(tempoBars([NaN,-1,0,Infinity,61]),[]);
  const large=tempoBars(Array.from({length:29},(_,i)=>.2+i*.1));
  assert.equal(large.length,12);
  assert.ok(Math.abs(large[0].durationSec-1.9)<1e-10);
  assert.ok(Math.abs(large[11].durationSec-3)<1e-10);
  assert.ok(large.every(x=>x.heightPercent>=9&&x.heightPercent<=100));
  assert.throws(()=>tempoBars([1],0), /bounded/);
});

test('W3 site wires start, frame updates, clock even on frame stall, and all privacy reset paths', () => {
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const css=readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  for (const id of [
    'sessionDashboard','sessionState','sessionElapsed','sessionPoseSeen',
    'sessionJointReady','sessionFrames','sessionM34Mean','sessionM34Bars',
    'sessionM45Panel','sessionM45Mean','sessionM45Bars'
  ]) {
    assert.match(main,new RegExp('id="'+id+'"'));
  }
  assert.match(main,/state\.sessionDashboard\.start\(performance\.now\(\)\)/);
  assert.match(main,/state\.sessionDashboard\.observe\(timestamp/);
  assert.match(main,/state\.sessionDashboard\.reset\(\)/);
  assert.match(main,/ui\.sessionDashboard\.hidden = isPushup/);
  assert.match(main,/ui\.sessionM45Panel\.hidden = !squat/);
  assert.match(main,/tempoFromM34\(state\.repCounter\.recent\)/);
  assert.match(main,/tempoFromM45\(state\.liveRelative\.events\)/);
  assert.match(main,/Clock and stalled-video status keep updating even if decoded frames stop/);
  assert.match(main,/aria-label="Local experimental repetition session dashboard"/);
  assert.match(main,/role="img" aria-label="M34 no completed movements yet"/);
  assert.match(css,/\.session-stat-grid/);
  assert.match(css,/@media \(max-width: 470px\)/);
  assert.match(css,/\.session-dashboard\[hidden\]/);
  const model=readFileSync(new URL('../src/session_dashboard.js',import.meta.url),'utf8');
  assert.doesNotMatch(model,/\bfetch\s*\(|localStorage|sessionStorage|IndexedDB|MediaRecorder|XMLHttpRequest/);
  assert.doesNotMatch(model,/formQuality|injuryRisk|modelConfidence/);
  assert.match(main,/state\.repCounter\.reset\(\)/);
  assert.match(main,/state\.liveRelative\.reset\(\)/);
});

test('W3 clocked session stays observational without exposure of source RF scores or GT labels', () => {
  const dash=new SessionDashboard();
  dash.start(0);
  for(let t=200;t<=1600;t+=200) {
    dash.observe(t,{poseSeen:t%400===0,requiredReady:t%800===0});
  }
  const summary=dash.summary(1800);
  assert.equal(summary.sampleFrames,8);
  assert.equal(summary.poseSeenPct,50);
  assert.equal(summary.requiredReadyPct,25);
  assert.equal(summary.status,'JOINTS VISIBLE');
  assert.equal('accuracy' in summary,false);
  assert.equal('confidence' in summary,false);
  assert.equal('gt' in summary,false);
  assert.equal('pose' in summary,false);
});
