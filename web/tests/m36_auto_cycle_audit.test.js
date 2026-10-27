import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { poseFromAngle, runSyntheticAudit, AUDIT_VERSION, EXERCISES,
  PROJECTIONS } from '../scripts/m36_auto_cycle_audit.js';
import { measureCycleAngle } from '../src/rep_counter.js';

test('M36 automatic deterministic stress suite passes every generated case', () => {
  const report = runSyntheticAudit();
  assert.equal(report.audit, AUDIT_VERSION);
  assert.equal(report.candidate_cases, 216);
  assert.equal(report.matched_synthetic_expectations, 216);
  assert.equal(report.mismatched_synthetic_expectations, 0);
  assert.equal(report.all_synthetic_expectations_met, true);
  assert.deepEqual(report.failed_synthetic_cases, []);
  assert.deepEqual(Object.keys(report.by_exercise), EXERCISES);
  assert.equal(report.by_scenario.complete_clean.cases, 72);
  assert.equal(report.by_scenario.complete_small_jitter.cases, 72);
  for (const exercise of EXERCISES) {
    assert.equal(report.by_exercise[exercise].cases, 72);
    assert.equal(report.by_exercise[exercise].mean_absolute_synthetic_count_error, 0);
  }
});

test('M36 synthetic stress is deterministic and never claims real accuracy', () => {
  const a = runSyntheticAudit({ seed: 36 });
  const b = runSyntheticAudit({ seed: 36 });
  assert.deepEqual(a, b);
  assert.equal(a.human_annotated_videos_used, false);
  assert.equal(a.human_rep_count_accuracy_measured, false);
  assert.equal(a.real_world_rep_precision_recall_measured, false);
  assert.equal(a.raw_video_or_user_landmarks_accessed, false);
  assert.equal(a.m20_val_or_test_accessed, false);
  assert.equal(a.form_quality_assessed, false);
  assert.match(a.conclusion, /NOT evidence/);
  assert.match(a.projection_warning, /NOT real/);
});

test('M36 transformations preserve intended angle ranges without assuming actual camera generalization', () => {
  for (const exercise of EXERCISES) {
    for (const projection of PROJECTIONS) {
      for (const side of ['LEFT', 'RIGHT']) {
        const high = measureCycleAngle(exercise,
          poseFromAngle(exercise, 170, { side, ...projection }));
        const low = measureCycleAngle(exercise,
          poseFromAngle(exercise, 95, { side, ...projection }));
        assert.ok(high && low);
        assert.equal(high.side, side);
        assert.equal(low.side, side);
        assert.ok(high.degrees >= 155);
        assert.ok(low.degrees <= 115);
      }
    }
  }
});

test('M36 does not invent participant files, ground truth or network IO', () => {
  const source = readFileSync(new URL('../scripts/m36_auto_cycle_audit.js', import.meta.url), 'utf8');
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(pkg.scripts['validate:m36'], 'node scripts/m36_auto_cycle_audit.js');
  assert.doesNotMatch(source, /\bfetch\s*\(|XMLHttpRequest|VideoCapture|localStorage|MediaRecorder/);
  assert.match(source, /deterministic synthetic camera-plane landmark contract ONLY/);
  assert.match(source, /process\.exitCode = 1/);
});

test('M36 fails fast on malformed synthetic input and seed', () => {
  assert.throws(() => poseFromAngle('pushup', 170), /Invalid synthetic/);
  assert.throws(() => poseFromAngle('squat', Infinity), /Invalid synthetic/);
  assert.throws(() => poseFromAngle('ohp', 170, { side: 'BOTH' }), /Invalid synthetic/);
  assert.throws(() => poseFromAngle('squat', 90, { visibility: -1 }), /Invalid synthetic/);
  assert.throws(() => runSyntheticAudit({ seed: -1 }), /Seed must/);
});
