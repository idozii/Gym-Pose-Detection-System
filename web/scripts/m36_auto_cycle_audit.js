/** M36 automatic deterministic stress audit of the M34 browser angle-cycle counter.
 *
 * NO real participant/video data, manual CSV, camera, server API or publisher
 * test sets. Generated joint triplets have deliberately known cycles.
 * A correct synthetic counter run is engineering verification only, NOT
 * human/video repetition precision/recall or exercise form accuracy.
 */
import { RepCycleCounter } from '../src/rep_counter.js';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const AUDIT_VERSION = 'm36-synthetic-counter-stress-v1';
export const EXERCISES = Object.freeze(['squat', 'ohp', 'barbell_row']);
export const PROJECTIONS = Object.freeze([
  { name: 'image_neutral', radians: 0, horizontalScale: 1 },
  { name: 'image_rotated', radians: 12 * Math.PI / 180, horizontalScale: 1 },
  { name: 'image_horizontal_compressed', radians: -8 * Math.PI / 180, horizontalScale: .82 }
]);

export function poseFromAngle(exercise, angle, {
  side = 'LEFT', radians = 0, horizontalScale = 1, visibility = .98
} = {}) {
  if (!EXERCISES.includes(exercise) || !['LEFT', 'RIGHT'].includes(side) ||
    !Number.isFinite(angle) || angle < 0 || angle > 180 ||
    !Number.isFinite(radians) || !Number.isFinite(horizontalScale) ||
    horizontalScale <= 0 || !Number.isFinite(visibility) ||
    visibility < 0 || visibility > 1) {
    throw new Error('Invalid synthetic 2D joint pose request');
  }
  const landmarks = Array.from({ length: 33 }, () => ({
    x: .1, y: .1, z: 0, visibility: .01
  }));
  const left = side === 'LEFT';
  const isSquat = exercise === 'squat';
  const ids = isSquat
    ? (left ? [23, 25, 27] : [24, 26, 28])
    : (left ? [11, 13, 15] : [12, 14, 16]);
  const r = .18;
  const rad = angle * Math.PI / 180;
  const projected = (x, y) => {
    const scaledX = x * horizontalScale;
    return {
      x: .5 + scaledX * Math.cos(radians) - y * Math.sin(radians),
      y: .5 + scaledX * Math.sin(radians) + y * Math.cos(radians),
      z: 0, visibility
    };
  };
  landmarks[ids[0]] = projected(0, -r);
  landmarks[ids[1]] = projected(0, 0);
  landmarks[ids[2]] = projected(r * Math.sin(rad), -r * Math.cos(rad));
  return landmarks;
}

function randomGenerator(seed) {
  if (!Number.isInteger(seed) || seed < 0) throw new Error('Seed must be nonnegative integer');
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

function normalCycles(number) {
  const frames = [170, 170];
  for (let n = 0; n < number; n++) {
    frames.push(150, 125, 100, 95, 95, 125, 150, 170, 170);
  }
  if (number === 0) frames.push(...Array(20).fill(170));
  return frames;
}

function evaluate(exercise, scenario, projection, side, frames, expected, rng) {
  const counter = new RepCycleCounter();
  let at = 0;
  for (const frame of frames) {
    const angle = typeof frame === 'number' ? frame : frame?.angle;
    const cameraSide = (frame && typeof frame === 'object' && frame.side) || side;
    const isMissing = frame === null || frame?.missing === true;
    const jitter = scenario === 'complete_small_jitter' ? (rng() - .5) * 5 : 0;
    const landmarks = isMissing ? null : poseFromAngle(exercise, angle + jitter, {
      side: cameraSide, ...projection
    });
    counter.observe(exercise, landmarks, at);
    at += 400;
  }
  const actual = counter.summary()?.cycles ?? 0;
  return {
    exercise,
    scenario,
    projection: projection.name,
    side,
    expected_cycles: expected,
    counted_cycles: actual,
    absolute_count_error: Math.abs(actual - expected),
    passed: actual === expected
  };
}

export function runSyntheticAudit({ seed = 36 } = {}) {
  const rng = randomGenerator(seed);
  const cases = [];
  const add = (exercise, scenario, projection, side, frames, expected) => {
    cases.push(evaluate(exercise, scenario, projection, side,
      frames, expected, rng));
  };
  for (const exercise of EXERCISES) {
    for (const projection of PROJECTIONS) {
      for (const side of ['LEFT', 'RIGHT']) {
        for (const count of [0, 1, 3, 5]) {
          add(exercise, 'complete_clean', projection, side,
            normalCycles(count), count);
          add(exercise, 'complete_small_jitter', projection, side,
            normalCycles(count), count);
        }
        // A half cycle may be legitimate user motion, but is not a
        // completed high-low-high event according to the M34 contract.
        add(exercise, 'partial_cycle', projection, side,
          [170, 170, 140, 100, 95, 95, 120, 130], 0);
        add(exercise, 'stationary_camera', projection, side,
          Array(30).fill(167), 0);
        // Loss of tracking in the middle of one pending cycle must
        // invalidate that candidate, not fabricate a completed cycle.
        add(exercise, 'occluded_before_return', projection, side,
          [170, 170, 100, 95, 95, null, null, null, null, null, 170, 170], 0);
        const alternate = side === 'LEFT' ? 'RIGHT' : 'LEFT';
        add(exercise, 'switch_body_side', projection, side,
          [170, 170, 100, 95, 95,
            { angle: 170, side: alternate },
            { angle: 170, side: alternate }], 0);
      }
    }
  }
  const perScenario = {};
  const perExercise = {};
  for (const entry of cases) {
    for (const [bucket, key] of [
      [perScenario, entry.scenario], [perExercise, entry.exercise]
    ]) {
      if (!bucket[key]) bucket[key] = {
        cases: 0, passed: 0, absolute_count_error_sum: 0
      };
      bucket[key].cases++;
      bucket[key].passed += Number(entry.passed);
      bucket[key].absolute_count_error_sum += entry.absolute_count_error;
    }
  }
  for (const bucket of [perScenario, perExercise]) {
    for (const value of Object.values(bucket)) {
      value.exact_synthetic_case_fraction = value.passed / value.cases;
      value.mean_absolute_synthetic_count_error =
        value.absolute_count_error_sum / value.cases;
    }
  }
  const passed = cases.filter((item) => item.passed).length;
  return {
    audit: AUDIT_VERSION,
    run_seed: seed,
    evidence_level: 'deterministic synthetic camera-plane landmark contract ONLY',
    original_counter_unmodified: true,
    exercises: EXERCISES,
    synthetic_projections: PROJECTIONS.map((p) => p.name),
    projection_warning: 'Only in-plane rotation and simple x-axis compression are simulated. They are NOT real front/side/diagonal camera video evaluations.',
    candidate_cases: cases.length,
    matched_synthetic_expectations: passed,
    mismatched_synthetic_expectations: cases.length - passed,
    all_synthetic_expectations_met: passed === cases.length,
    by_scenario: perScenario,
    by_exercise: perExercise,
    failed_synthetic_cases: cases.filter((item) => !item.passed),
    human_annotated_videos_used: false,
    human_rep_count_accuracy_measured: false,
    real_world_rep_precision_recall_measured: false,
    form_quality_assessed: false,
    m20_val_or_test_accessed: false,
    raw_video_or_user_landmarks_accessed: false,
    conclusion: 'Passing means the M34 counting state machine satisfies selected artificial angle-sequence expectations. It is NOT evidence of reliable repetition counting on unseen participants or reliable technique diagnosis.'
  };
}

function main() {
  const result = runSyntheticAudit();
  const output = resolve('reports/m36_synthetic_cycle_audit.json');
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(result, null, 2) + '\n', {
    encoding: 'utf8', flag: 'w'
  });
  console.log('M36 synthetic stress audit:', result.matched_synthetic_expectations,
    '/', result.candidate_cases, 'cases matched.');
  console.log('Report:', output);
  console.log('Only deterministic artificial joint motions, NOT real rep accuracy.');
  if (!result.all_synthetic_expectations_met) {
    console.error(JSON.stringify(result.failed_synthetic_cases, null, 2));
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
