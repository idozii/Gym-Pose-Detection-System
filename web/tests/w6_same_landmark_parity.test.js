import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { detectRelativeExcursionEvents, RULE } from '../scripts/m45_compare_train_relative.js';
import { LIVE_M45_RULE, LIVE_M45_SAMPLE_MS } from '../src/live_relative_excursion.js';
import {
  W6_PROTOCOL, evaluateSameLandmarksParity, auditTrace
} from '../scripts/w6_same_landmark_parity.js';

function pose(angle, side='LEFT') {
  if(angle === null)return null;
  const points=Array.from({length:33},()=>({
    x:.2,y:.2,z:0,visibility:.1
  }));
  const hip=side==='LEFT'?23:24,knee=side==='LEFT'?25:26,
    ankle=side==='LEFT'?27:28;
  const theta=angle*Math.PI/180;
  points[hip]={x:.4,y:.28,z:0,visibility:.98};
  points[knee]={x:.4,y:.5,z:0,visibility:.98};
  points[ankle]={x:.4+.22*Math.sin(theta),
    y:.5-.22*Math.cos(theta),z:0,visibility:.98};
  return points;
}
const frames=(a,gap=200)=>a.map((v,i)=>({
  time_ms:i*gap,landmarks:typeof v==='object'&&v!==null?pose(v.angle,v.side):
    pose(v)
}));

test('W6 exact original M45 and website relative rule constants match',()=>{
  assert.equal(W6_PROTOCOL,'w6-same-landmarks-live-m45-algorithmic-parity-v1');
  assert.equal(LIVE_M45_SAMPLE_MS,200);
  assert.deepEqual(LIVE_M45_RULE,RULE);
});

test('W6 streaming M45 reproduces event timestamps on same frame trace',()=>{
  for(const values of [
    [170,165,133,126,134,149,169],
    [171,135,116,137,170,170,130,114,129,170],
    [170,170,169,167,170,169,170],
    [170,160,120,110,112,120,130,145,154,172],
    [170,130,125,170,170,130,130,170]
  ]){
    const f=frames(values);
    const result=evaluateSameLandmarksParity(f);
    assert.equal(result.source_frames,f.length);
    assert.equal(result.direct_same_input_event_disagreements,0);
    assert.equal(result.gate_same_input_event_disagreements,0);
    assert.equal(result.original_m45_event_count,
      detectRelativeExcursionEvents(f).event_times_ms.length);
    assert.equal(result.direct_exact_event_parity,true);
    assert.equal(result.gated_exact_event_parity,true);
    assert.equal(result.pose_estimator_parity_verified,false);
    assert.equal(result.webpage_runtime_parity_verified,false);
  }
});

test('W6 gate drops frames and compares against correct gated reference',()=>{
  const times=[0,80,160,200,320,400,480,600,650,800,1000,1200,1400,1600];
  const angles=[170,168,160,165,145,120,119,128,135,142,167,170,169,170];
  const frames1=times.map((t,i)=>({time_ms:t,landmarks:pose(angles[i])}));
  const out=evaluateSameLandmarksParity(frames1);
  assert.ok(out.gate_dropped_frames>0);
  assert.equal(out.gate_dropped_frames+
    out.gate_selected_frames,out.source_frames);
  assert.equal(out.direct_same_input_event_disagreements,0);
  assert.equal(out.gate_same_input_event_disagreements,0);
});

test('W6 tracking gaps and side switches do not create false parity',()=>{
  const f=[
    {time_ms:0,landmarks:pose(170)},
    {time_ms:200,landmarks:pose(130)},
    {time_ms:400,landmarks:null},
    {time_ms:1700,landmarks:null},
    {time_ms:1800,landmarks:pose(170,'RIGHT')},
    {time_ms:2000,landmarks:pose(120,'RIGHT')},
    {time_ms:2500,landmarks:pose(170,'RIGHT')},
    {time_ms:2700,landmarks:pose(170,'LEFT')},
    {time_ms:2900,landmarks:pose(120,'LEFT')},
    {time_ms:3400,landmarks:pose(170,'LEFT')}
  ];
  const out=evaluateSameLandmarksParity(f);
  assert.equal(out.direct_same_input_event_disagreements,0);
  assert.equal(out.gate_same_input_event_disagreements,0);
  assert.ok(out.original_m45_event_count>=1);
});

test('W6 emits only two-cadence aggregate proof, never GT or source IDs',()=>{
  const f=frames([170,133,120,142,170,170,130,128,169,170]);
  const results=auditTrace({frames:f,private_source_name:'do-not-persist.mp4',
    gtEvents:[{private:true}],source_split:'train'});
  assert.equal(results.protocol,W6_PROTOCOL);
  assert.equal(results.source_split,'train');
  assert.deepEqual(Object.keys(results.conditions).sort(),
    ['dense_200ms','subsample_400ms']);
  assert.equal(results.validation_or_test_accessed,false);
  assert.equal(results.raw_frames_or_event_times_saved,false);
  assert.equal(results.browser_or_python_pose_parity_claimed,false);
  const serialized=JSON.stringify(results);
  assert.doesNotMatch(serialized,/do-not-persist|\"landmarks\"\s*:|private|gtEvents|\"event_times_ms\"\s*:/);
  assert.equal(results.conditions.dense_200ms.direct_exact_event_parity,true);
});

test('W6 rejects invalid or nonchronological source traces',()=>{
  const f=frames([170,130,170,120,170]);
  assert.throws(()=>evaluateSameLandmarksParity([f[0]]),/2..20000/);
  assert.throws(()=>evaluateSameLandmarksParity([f[0],f[0]]),/chronological/);
  assert.throws(()=>evaluateSameLandmarksParity([
    {time_ms:0,landmarks:[{x:0}]},
    {time_ms:200,landmarks:pose(170)}
  ]),/chronological/);
  assert.throws(()=>auditTrace({source_split:'test'}),/original frozen/);
  assert.throws(()=>auditTrace(null),/original frozen/);
});

test('W6 process has no browser camera capture, remote API or persistent landmark array in output',()=>{
  const code=readFileSync(new URL('../scripts/w6_same_landmark_parity.js',import.meta.url),'utf8');
  assert.doesNotMatch(code, /getUserMedia|PoseLandmarker|detectForVideo|XMLHttpRequest|localStorage|indexedDB|fetch\s*\(/);
  assert.match(code,/detectRelativeExcursionEvents\(frames\)/);
  assert.match(code,/counter\.consumeSample/);
  assert.match(code,/counter\.observe/);
  assert.match(code,/writeFileSync\(resolve\(output\)/);
  assert.match(code,/\{flag:'wx'\}/);
});
