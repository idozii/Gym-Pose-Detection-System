import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf-8');
const css = readFileSync(new URL('../src/style.css', import.meta.url), 'utf-8');

test('mobile camera controls are accessible and wired to physical track switching', () => {
  assert.match(source, /id="cameraFacing"/);
  assert.match(source, /id="mirrorButton"/);
  assert.match(source, /aria-pressed="true"/);
  assert.match(source, /cameraFacing\.addEventListener\('change'/);
  assert.match(source, /stopCamera\(\);\s*state\.facingMode = next;/);
  assert.match(source, /getUserMedia\(cameraConstraints\(state\.facingMode\)\)/);
});

test('selfie mirroring is restricted to display canvas, not AI landmark payload', () => {
  assert.match(source, /if \(state\.sourceMode === 'camera' &&\s*mirrorPreview\(state\.facingMode, state\.mirrorSelfie\)\)/);
  assert.match(source, /context\.translate\(canvas\.width, 0\)/);
  assert.match(source, /context\.scale\(-1, 1\)/);
  assert.doesNotMatch(source, /state\.landmarks\s*=\s*state\.landmarks\.map/);
  assert.match(source, /disableResearchInference\(\);[\s\S]*?state\.token\+\+/);
});

test('design ships responsive camera action layout, focus styles and reduced motion', () => {
  assert.match(css, /\.camera-actions/);
  assert.match(css, /\.mirror-btn\[aria-pressed=true\]/);
  assert.match(css, /@media \(max-width: 640px\)/);
  assert.match(css, /:focus-visible/);
  assert.match(css, /prefers-reduced-motion: reduce/);
});
