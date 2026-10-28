import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  W4_MAX_VIDEO_BYTES, W4_MAX_VIDEO_SECONDS,
  validateLocalVideoFile, validateVideoMetadata,
  localVideoTimestampMs, localFrameIsNew, canRequestRemoteInference
} from '../src/local_video.js';

const mp4 = (name = 'research.mp4', size = 1234567, type = 'video/mp4') =>
  ({ name, size, type });

test('W4 accepts bounded local MP4/WebM and rejects oversized or malformed source', () => {
  assert.equal(validateLocalVideoFile(mp4()).ok, true);
  assert.equal(validateLocalVideoFile(mp4('SQUAT.MP4', W4_MAX_VIDEO_BYTES, '')).ok, true);
  assert.equal(validateLocalVideoFile(mp4('movement.webm', 99999, 'video/webm')).ok, true);
  assert.equal(W4_MAX_VIDEO_BYTES, 250 * 1024 * 1024);
  for (const file of [
    null, undefined, {}, mp4('video.mov'), mp4('frames.png', 12345, 'image/png'),
    mp4('gym.mp4', 2345, 'image/jpeg'), mp4('gym.mp4', 0),
    mp4('gym.mp4', -1), mp4('gym.mp4', W4_MAX_VIDEO_BYTES + 1),
    mp4('gym.mp4', 12.5), mp4('gym.mp4', Number.POSITIVE_INFINITY)
  ]) {
    assert.equal(validateLocalVideoFile(file).ok, false);
  }
  const invalid = validateLocalVideoFile(mp4('personal-private-name.mp4', 123, 'application/octet-stream'));
  assert.equal(invalid.ok, false);
  assert.doesNotMatch(invalid.reason, /personal-private-name/);
});

test('W4 only accepts bounded playable video timeline and picture dimensions', () => {
  assert.equal(W4_MAX_VIDEO_SECONDS, 300);
  assert.equal(validateVideoMetadata({ duration: 120, width: 1920, height: 1080 }).ok, true);
  assert.equal(validateVideoMetadata({ duration: 300, width: 3840, height: 2160 }).ok, true);
  for (const obj of [
    {duration:0,width:100,height:100},
    {duration:301,width:100,height:100},
    {duration:Infinity,width:100,height:100},
    {duration:NaN,width:100,height:100},
    {duration:30,width:0,height:200},
    {duration:30,width:200,height:0},
    {duration:30,width:99999,height:200},
    {duration:30,width:1920.5,height:1080}
  ]) {
    assert.equal(validateVideoMetadata(obj).ok, false);
  }
});

test('W4 video pose timestamps are tied to decoded media position, not RAF clock', () => {
  assert.equal(localVideoTimestampMs(0), 0);
  assert.equal(localVideoTimestampMs(.1999), 200);
  assert.equal(localVideoTimestampMs(1.25), 1250);
  assert.equal(localVideoTimestampMs(120), 120000);
  assert.throws(() => localVideoTimestampMs(-.1), /nonnegative/);
  assert.throws(() => localVideoTimestampMs(Infinity), /nonnegative/);
  assert.throws(() => localVideoTimestampMs(NaN), /nonnegative/);
  assert.equal(localFrameIsNew({
    currentTime:0,previousTime:-1,timestampMs:0,previousTimestampMs:null
  }), true);
  assert.equal(localFrameIsNew({
    currentTime:0.1,previousTime:0.1,timestampMs:100,previousTimestampMs:100
  }), false);
  assert.equal(localFrameIsNew({
    currentTime:0.1001,previousTime:0.1,timestampMs:100,previousTimestampMs:100
  }), false);
  assert.equal(localFrameIsNew({
    currentTime:0.2,previousTime:0.1,timestampMs:200,previousTimestampMs:100
  }), true);
  assert.equal(localFrameIsNew({
    currentTime:0.1,previousTime:0.2,timestampMs:100,previousTimestampMs:200
  }), false);
});

test('W4 local video disables remote inference regardless of stale user consent', () => {
  assert.equal(canRequestRemoteInference({
    sourceMode:'video', enabled:true, running:true
  }),false);
  assert.equal(canRequestRemoteInference({
    sourceMode:'video', enabled:true, running:false
  }),false);
  assert.equal(canRequestRemoteInference({
    sourceMode:'camera', enabled:true, running:true
  }),true);
  assert.equal(canRequestRemoteInference({
    sourceMode:'camera', enabled:false, running:true
  }),false);
  assert.equal(canRequestRemoteInference({
    sourceMode:'camera', enabled:true, running:false
  }),false);
});

test('W4 UI is local-file opt-in and maintains camera-only model API gate', () => {
  const main = readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const css = readFileSync(new URL('../src/style.css',import.meta.url),'utf8');
  for (const element of [
    'chooseCamera','chooseVideo','localVideoActions','localVideoFile',
    'localPlay','localReplay','localClear','localProgress','localVideoPosition'
  ]) assert.match(main, new RegExp('id="' + element + '"'));
  assert.match(main,/accept="\.mp4,\.webm,video\/mp4,video\/webm"/);
  assert.match(main,/URL\.createObjectURL\(file\)/);
  assert.match(main,/URL\.revokeObjectURL\(state\.localUrl\)/);
  assert.match(main,/state\.localLoadAbort\?\.abort\(\)/);
  assert.match(main,/ui\.video\.removeAttribute\('src'\)/);
  assert.match(main,/function waitForLocalMetadata/);
  assert.match(main,/validateVideoMetadata\(/);
  assert.match(main,/state\.sessionDashboard\.start\(0\)/);
  assert.match(main,/localVideoTimestampMs\(v\.currentTime \|\| 0\)/);
  assert.match(main,/state\.landmarker\.detectForVideo\(v, frameMs\)/);
  assert.match(main,/state\.liveRelative\.observe\(state\.landmarks, frameMs\)/);
  assert.match(main,/state\.repCounter\.observe/);
  assert.match(main,/function toggleLocalPlayback/);
  assert.match(main,/function selectSourceMode/);
  assert.match(main,/sourceMode: state\.sourceMode, enabled: state\.inferenceEnabled/);
  assert.match(main,/if \(!local\) void requestResearchInference\(timestamp\)/);
  assert.match(main,/if \(state\.sourceMode === 'video' \|\| state\.exercise === 'pushup'\)/);
  assert.match(main,/state\.sourceMode === 'camera' &&/);
  assert.match(main,/state\.localNeedsReplay = true/);
  assert.match(main,/function drawFrame/);
  assert.match(main,/ui\.video\.addEventListener\('ended'/);
  assert.match(main,/document\.addEventListener\('visibilitychange'/);
  assert.match(css, /\.source-mode-choice\[aria-pressed="true"\]/);
  assert.match(css, /\.local-video-actions\[hidden\]/);
  assert.match(css, /@media \(max-width: 470px\)/);
});

test('W4 no upload, model inference or local storage APIs in media file input helper', () => {
  const source = readFileSync(new URL('../src/local_video.js',import.meta.url),'utf8');
  assert.doesNotMatch(source,
    /fetch\s*\(|XMLHttpRequest|navigator\.sendBeacon|localStorage|indexedDB|sessionStorage|MediaRecorder|FormData/);
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const loading=main.split('async function loadLocalVideo(file)')[1]
    .split('async function toggleLocalPlayback()')[0];
  assert.doesNotMatch(loading, /\bfetch\s*\(|ResearchInferenceClient|\.score\s*\(/);
  assert.doesNotMatch(loading, /\blocalStorage\b|sessionStorage|MediaRecorder|FormData/);
});
