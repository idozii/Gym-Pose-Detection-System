/** W6: same-input algorithmic parity audit, NOT browser detector accuracy.
 *
 * Compare the exact unchanged M45 TRAIN comparator with the W2 incremental
 * browser logic on identical landmark arrays and media timestamps.
 * Also compare the W2 observe() gate with the M45 comparator on the
 * EXACT gated inputs (not with a different 200ms trace).
 *
 * IMPORTANT: W6 does not instantiate MediaPipe Tasks Vision, decode media,
 * perform browser camera inference, or establish browser vs Python pose
 * model parity. The original offline Python pose frames are the SAME input.
 *
 * Never save frame-level landmarks, names, filenames, events or GT data.
 */
import { detectRelativeExcursionEvents } from './m45_compare_train_relative.js';
import { LiveRelativeExcursionCounter, LIVE_M45_SAMPLE_MS } from '../src/live_relative_excursion.js';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const W6_PROTOCOL = 'w6-same-landmarks-live-m45-algorithmic-parity-v1';

function countMismatches(expected, actual) {
  const max = Math.max(expected.length,actual.length);
  let disagreements=0;
  for(let i=0;i<max;i++) {
    if(expected[i]!==actual[i]) disagreements++;
  }
  return disagreements;
}

function validateFrames(frames) {
  if(!Array.isArray(frames) || frames.length<2 || frames.length>20000) {
    throw new TypeError('W6 requires 2..20000 chronological source pose frames');
  }
  let last=-Infinity;
  for(const f of frames) {
    if(!f || !Number.isFinite(f.time_ms) ||
       f.time_ms<0 || f.time_ms<=last ||
       !Object.hasOwn(f,'landmarks') ||
       (f.landmarks !== null &&
        (!Array.isArray(f.landmarks) || f.landmarks.length!==33))) {
      throw new TypeError('W6 invalid chronological landmark trace');
    }
    last=f.time_ms;
  }
}

function replayIncremental(frames, gated=false) {
  const counter=new LiveRelativeExcursionCounter();
  const used=[];
  const events=[];
  let prev=0;
  for(const f of frames) {
    const before=counter.sampledFrames;
    const out=gated
      ? counter.observe(f.landmarks,f.time_ms)
      : counter.consumeSample(f.landmarks,f.time_ms);
    if(counter.sampledFrames>before)used.push(f);
    if(out.cycles>prev) {
      if(out.cycles!==prev+1)throw new Error('W6 incremental counter skipped a cycle');
      events.push(f.time_ms);
    }
    prev=out.cycles;
  }
  return {events,used};
}

export function evaluateSameLandmarksParity(frames) {
  validateFrames(frames);
  const reference=detectRelativeExcursionEvents(frames).event_times_ms;
  const manual=replayIncremental(frames,false);
  const gated=replayIncremental(frames,true);
  const gatedReference=detectRelativeExcursionEvents(gated.used).event_times_ms;
  const directMismatch=countMismatches(reference,manual.events);
  const gatedMismatch=countMismatches(gatedReference,gated.events);
  return {
    source_frames:frames.length,
    gate_sample_ms:LIVE_M45_SAMPLE_MS,
    gate_selected_frames:gated.used.length,
    original_m45_event_count:reference.length,
    browser_incremental_event_count:manual.events.length,
    direct_same_input_event_disagreements:directMismatch,
    gate_reference_event_count:gatedReference.length,
    gate_browser_event_count:gated.events.length,
    gate_same_input_event_disagreements:gatedMismatch,
    gate_dropped_frames:frames.length-gated.used.length,
    direct_exact_event_parity:directMismatch===0,
    gated_exact_event_parity:gatedMismatch===0,
    pose_estimator_parity_verified:false,
    webpage_runtime_parity_verified:false
  };
}

export function auditTrace(trace) {
  if(!trace || typeof trace!=='object' || !Array.isArray(trace.frames)) {
    throw new TypeError('W6 expects original frozen M45 JSON trace with frames');
  }
  const views={
    dense_200ms:trace.frames,
    subsample_400ms:trace.frames.filter((_,i)=>i%2===0)
  };
  const result={};
  for(const [name,frames] of Object.entries(views))result[name]=evaluateSameLandmarksParity(frames);
  return {
    protocol:W6_PROTOCOL,
    source_split:'train',
    experiment:'identical_offline_pose_input_algorithm_logic_only',
    conditions:result,
    validation_or_test_accessed:false,
    raw_frames_or_event_times_saved:false,
    browser_or_python_pose_parity_claimed:false
  };
}

function main() {
  const [, , input,output]=process.argv;
  if(!input || !output) throw new Error('Usage: node web/scripts/w6_same_landmark_parity.js <private_train_trace.json> <private_result.json>');
  const trace=JSON.parse(readFileSync(resolve(input),'utf8'));
  const report=auditTrace(trace);
  writeFileSync(resolve(output), JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log('W6 identical-input algorithmic parity audit complete (no browser/runtime claim)');
}
if(process.argv[1] && resolve(process.argv[1])===fileURLToPath(import.meta.url))main();
