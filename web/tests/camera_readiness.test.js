import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  cameraErrorMessage, shouldRevokeOnHidden, shouldIgnoreEndedTrack
} from '../src/camera_readiness.js';

test('W1 camera permission and device failures have actionable non-leaking help', () => {
  for (const name of ['NotAllowedError','PermissionDeniedError','SecurityError']) {
    assert.match(cameraErrorMessage({name}), /permission|Allow camera/i);
  }
  assert.match(cameraErrorMessage({name:'NotFoundError'}), /No available camera/);
  assert.match(cameraErrorMessage({name:'NotReadableError'}), /busy/);
  assert.match(cameraErrorMessage({name:'OverconstrainedError'}), /resolution/);
  assert.match(cameraErrorMessage({name:'AbortError'}), /interrupted/);
  assert.match(cameraErrorMessage({message:'private data',name:'UnexpectedError'}), /Check browser support/);
  assert.doesNotMatch(cameraErrorMessage({message:'private data'}), /private data/);
});

test('W1 hidden tabs revoke camera while both active AND still initializing', () => {
  assert.equal(shouldRevokeOnHidden({hidden:true,running:true,starting:false}),true);
  assert.equal(shouldRevokeOnHidden({hidden:true,running:false,starting:true}),true);
  assert.equal(shouldRevokeOnHidden({hidden:true,running:false,starting:false}),false);
  assert.equal(shouldRevokeOnHidden({hidden:false,running:true,starting:true}),false);
});

test('W1 stale ended-track events never interfere with replacement camera session', () => {
  assert.equal(shouldIgnoreEndedTrack({
    cancelled:false,currentToken:2,startedToken:2
  }),false);
  assert.equal(shouldIgnoreEndedTrack({
    cancelled:true,currentToken:2,startedToken:2
  }),true);
  assert.equal(shouldIgnoreEndedTrack({
    cancelled:false,currentToken:3,startedToken:2
  }),true);
});

test('W1 camera UI is opt-in, HTTPS-gated, cancels stale camera startup and reacts to ended tracks', () => {
  const main = readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(main,/navigator\.mediaDevices\?\.getUserMedia/);
  assert.match(main,/window\.isSecureContext/);
  assert.match(main,/if \(document\.hidden\)/);
  assert.match(main,/shouldRevokeOnHidden/);
  assert.match(main,/tempStream\.getVideoTracks\(\)\.forEach/);
  assert.match(main,/track\.addEventListener\('ended'/);
  assert.match(main,/cancelAnimationFrame\(state\.raf\)/);
  assert.match(main,/tempStream\?\.getTracks\(\)\.forEach/);
  assert.match(main,/state\.inferenceAbort\?\.abort\(\)/);
  assert.match(main,/state\.liveRelative\.reset\(\)/);
  assert.match(main,/cameraErrorMessage\(error\)/);
  assert.match(main,/const token = \+\+state\.token/);
  assert.match(main,/if \(token !== state\.token\) throw new Error\('Camera start cancelled'\)/);
  assert.match(main,/window\.addEventListener\('pagehide', stopCamera\)/);
});
