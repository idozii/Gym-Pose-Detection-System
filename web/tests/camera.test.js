import test from 'node:test';
import assert from 'node:assert/strict';
import { CAMERA_FACING, cameraConstraints, mirrorPreview, cameraFallbackMessage } from '../src/camera.js';

test('mobile front and rear use correct MediaDevices facing modes', () => {
  assert.deepEqual(cameraConstraints(CAMERA_FACING.FRONT), {
    audio: false, video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } }
  });
  assert.equal(cameraConstraints(CAMERA_FACING.REAR).video.facingMode.ideal, 'environment');
  assert.throws(() => cameraConstraints('unknown'), /Unsupported/);
});

test('mirroring applies only to front preview', () => {
  assert.equal(mirrorPreview('user', true), true);
  assert.equal(mirrorPreview('user', false), false);
  assert.equal(mirrorPreview('environment', true), false);
  assert.equal(mirrorPreview('environment', false), false);
});

test('camera fallback explicitly disclosed instead of promising rear camera', () => {
  assert.match(cameraFallbackMessage('environment', 'user'), /Rear camera not available/);
  assert.match(cameraFallbackMessage('user', 'environment'), /Front camera not available/);
  assert.equal(cameraFallbackMessage('environment', 'environment'), '');
  assert.equal(cameraFallbackMessage('environment', undefined), '');
});
