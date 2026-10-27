/** M33 local exercise movement observations. Research preview, never a form verdict.
 *
 * Geometry is computed on 2D, camera-plane coordinates. It cannot observe
 * lumbar curvature, bar path, loading, injury risk or true 3D joint motion.
 * No publisher source-error model scores, remote calls, video or landmarks
 * are stored here. The window holds only ephemeral degree measurements.
 */
const VISIBILITY = 0.5;
const LEFT = {
  squat: [23, 25, 27], ohp: [11, 13, 15],
};
const RIGHT = {
  squat: [24, 26, 28], ohp: [12, 14, 16],
};

export const EXERCISE_FEEDBACK = Object.freeze({
  squat: Object.freeze({
    name: 'Squat', measure: 'Knee flexion (2D)',
    camera: 'A side view with shoulder, hip, knee and ankle visible helps interpret knee motion. Perspective changes measured angles.',
    limitation: 'Knee angle alone cannot determine squat depth, knee safety or overall form.'
  }),
  ohp: Object.freeze({
    name: 'Overhead Press', measure: 'Elbow angle (2D)',
    camera: 'Keep the shoulder, elbow and wrist visible on one side through the press. Camera view changes the measured angle.',
    limitation: 'Elbow angle does not verify bar path, overhead stability or safe technique.'
  }),
  barbell_row: Object.freeze({
    name: 'Barbell Row', measure: 'Torso inclination (2D)',
    camera: 'A side view showing both shoulders and hips helps estimate torso orientation. Keep the camera still.',
    limitation: 'Shoulder/hip landmarks cannot measure spinal curvature or certify lumbar safety.'
  })
});

function usable(landmarks, ids) {
  if (!Array.isArray(landmarks) || landmarks.length !== 33) return false;
  return ids.every((id) => {
    const p = landmarks[id];
    return p && [p.x, p.y, p.z, p.visibility].every(Number.isFinite)
      && p.visibility >= VISIBILITY && p.visibility <= 1;
  });
}

function planeAngle(a, vertex, c) {
  const u = [a.x - vertex.x, a.y - vertex.y];
  const v = [c.x - vertex.x, c.y - vertex.y];
  const denom = Math.hypot(...u) * Math.hypot(...v);
  if (!Number.isFinite(denom) || denom < 1e-6) return null;
  const cos = Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1]) / denom));
  const angle = Math.acos(cos) * 180 / Math.PI;
  return Number.isFinite(angle) ? angle : null;
}

function oneSideAngle(landmarks, exercise) {
  const sides = [
    { name: 'LEFT', ids: LEFT[exercise] },
    { name: 'RIGHT', ids: RIGHT[exercise] },
  ].filter(({ ids }) => usable(landmarks, ids))
    .map((side) => ({
      ...side, quality: Math.min(...side.ids.map((id) => landmarks[id].visibility))
    }))
    .sort((a, b) => b.quality - a.quality);
  for (const side of sides) {
    const angle = planeAngle(...side.ids.map((id) => landmarks[id]));
    if (angle !== null) return { degrees: angle, side: side.name };
  }
  return null;
}

function rowInclination(landmarks) {
  const ids = [11, 12, 23, 24];
  if (!usable(landmarks, ids)) return null;
  const sh = {
    x: (landmarks[11].x + landmarks[12].x) / 2,
    y: (landmarks[11].y + landmarks[12].y) / 2
  };
  const hip = {
    x: (landmarks[23].x + landmarks[24].x) / 2,
    y: (landmarks[23].y + landmarks[24].y) / 2
  };
  const dx = sh.x - hip.x;
  const dy = sh.y - hip.y;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.hypot(dx, dy) < 0.03) {
    return null;
  }
  // Absolute camera-plane tilt relative to image vertical, [0, 90].
  return { degrees: Math.atan2(Math.abs(dx), Math.abs(dy)) * 180 / Math.PI,
    side: 'CENTER' };
}

export function measureExercise(exercise, landmarks) {
  if (!Object.hasOwn(EXERCISE_FEEDBACK, exercise)) return null;
  if (exercise === 'barbell_row') return rowInclination(landmarks);
  return oneSideAngle(landmarks, exercise);
}

export class ExerciseFeedbackWindow {
  constructor({ windowMs = 10000, sampleMs = 500, staleMs = 1250 } = {}) {
    if (!Number.isFinite(windowMs) || windowMs < 2000 ||
      !Number.isFinite(sampleMs) || sampleMs < 100 ||
      !Number.isFinite(staleMs) || staleMs <= sampleMs || staleMs > windowMs) {
      throw new Error('Invalid geometry timing policy');
    }
    this.windowMs = windowMs;
    this.sampleMs = sampleMs;
    this.staleMs = staleMs;
    this.reset();
  }

  reset() {
    this.exercise = null;
    this.samples = [];
    this.lastTick = -Infinity;
  }

  observe(exercise, landmarks, timeMs) {
    if (!Object.hasOwn(EXERCISE_FEEDBACK, exercise)) return null;
    if (this.exercise !== exercise) {
      this.reset();
      this.exercise = exercise;
    }
    if (!Number.isFinite(timeMs)) return this.summary(timeMs);
    // Do not carry motion across a backwards clock or inactive page.
    if (this.samples.length && timeMs < this.samples.at(-1).time) this.samples = [];
    this.samples = this.samples.filter((sample) =>
      sample.time <= timeMs && timeMs - sample.time <= this.windowMs);
    if (timeMs - this.lastTick >= this.sampleMs) {
      this.lastTick = timeMs;
      const value = measureExercise(exercise, landmarks);
      if (value) {
        // Never mix LEFT/RIGHT into a false movement amplitude.
        if (this.samples.length && this.samples.at(-1).side !== value.side) {
          this.samples = [];
        }
        this.samples.push({ time: timeMs, degrees: value.degrees, side: value.side });
      }
    }
    return this.summary(timeMs);
  }

  summary(timeMs) {
    const policy = EXERCISE_FEEDBACK[this.exercise];
    if (!policy || !Number.isFinite(timeMs)) return null;
    const valid = this.samples.filter((sample) =>
      sample.time <= timeMs && timeMs - sample.time <= this.windowMs);
    const recent = valid.length && (timeMs - valid.at(-1).time <= this.staleMs)
      ? valid.at(-1) : null;
    const values = valid.map((sample) => sample.degrees);
    const range = recent && values.length >= 4 ? Math.max(...values) - Math.min(...values) : null;
    return {
      exercise: this.exercise,
      title: policy.measure,
      angle: recent?.degrees ?? null,
      range,
      side: recent?.side ?? null,
      samples: valid.length,
      status: !recent ? 'insufficient_tracking' : range === null ? 'collecting' : 'observing',
      // These states characterize measurement availability, not form correctness.
      note: !recent
        ? 'Insufficient current landmarks. Reposition the camera until the listed joints are visible.'
        : range === null
          ? 'Angle observed. Keep moving within view to collect four clear samples (about 2 seconds).'
          : 'Movement range observed locally. It is not a good/bad score or a rep count.',
      camera: policy.camera,
      limitation: policy.limitation
    };
  }
}
