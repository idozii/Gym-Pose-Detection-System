import test from 'node:test';
import assert from 'node:assert/strict';
import { landmarkPayload, scoreState, ResearchInferenceClient } from '../src/inference.js';

const landmarks = () => Array.from({ length: 33 }, (_, i) => ({
  x: i / 100, y: i / 150, z: -i / 120, visibility: .91, privateNote: 'never sent'
}));

test('landmark payload strictly strips extra fields; only normalized pose leaves browser', () => {
  const p = landmarkPayload('squat', landmarks());
  assert.equal(p.landmarks.length, 33);
  assert.deepEqual(Object.keys(p.landmarks[0]), ['x', 'y', 'z', 'visibility']);
  assert.equal(JSON.stringify(p).includes('privateNote'), false);
  assert.equal(JSON.stringify(p).includes('video'), false);
});

test('invalid landmarks and unknown exercises refuse any outgoing request', () => {
  for (const bad of [null, [], landmarks().slice(0, 32)]) {
    assert.throws(() => landmarkPayload('ohp', bad), /landmarks|required/i);
  }
  const invalid = landmarks();
  invalid[0].visibility = NaN;
  assert.throws(() => landmarkPayload('squat', invalid), /Invalid pose/);
  invalid[0].visibility = .99;
  invalid[0].z = Infinity;
  assert.throws(() => landmarkPayload('squat', invalid), /Invalid pose/);
  assert.throws(() => landmarkPayload('deadlift', landmarks()), /Unsupported/);
});

test('score display only when returned probability and threshold pass validation', () => {
  const p = scoreState({
    status: 'experimental_source_label_score',
    error_probability: .734, threshold: .30
  });
  assert.equal(p.status, 'scored');
  assert.match(p.text, /Source score 73.4%/);
  assert.doesNotMatch(p.text, /safe|wrong|correct/i);
  assert.equal(scoreState({ status: 'abstain' }).text, 'ABSTAIN');
  for (const value of [NaN, 2, -1, null]) {
    assert.notEqual(scoreState({
      status: 'experimental_source_label_score',
      error_probability: value, threshold: .3
    }).status, 'scored');
  }
});

test('client does not use cookies, sends no camera frames or secrets', async () => {
  const calls = [];
  const client = new ResearchInferenceClient({
    apiUrl: 'https://research-api.example.com',
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      if (url.endsWith('/health')) {
        return { ok: true, json: async () => ({
          research_only: true, pose_estimator_source_parity_validated: false,
          models: { 'squat/error_knees_forward': 'loaded_experimental_only' }
        }) };
      }
      return { ok: true, json: async () => ({
        status: 'experimental_source_labels_only', exercise: 'squat',
        targets: {
          error_knees_forward: { status: 'experimental_source_label_score',
                                 error_probability: .5, threshold: .4 }
        }
      }) };
    }
  });
  await client.health();
  const report = await client.score('squat', landmarks());
  assert.equal(report.targets.error_knees_forward.error_probability, .5);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].init.credentials, 'omit');
  assert.equal(calls[1].init.cache, 'no-store');
  assert.equal(calls[1].init.method, 'POST');
  assert.deepEqual(Object.keys(JSON.parse(calls[1].init.body)), ['exercise', 'landmarks']);
  assert.equal(calls[1].init.body.includes('privateNote'), false);
});

test('backend config rejects insecure nonlocal HTTP and missing URL', () => {
  assert.throws(() => new ResearchInferenceClient({ apiUrl: '' }), /not configured/);
  assert.throws(() => new ResearchInferenceClient({ apiUrl: 'http://unsafe.example.com' }), /HTTPS/);
  assert.doesNotThrow(() => new ResearchInferenceClient({ apiUrl: 'http://localhost:8000' }));
});
