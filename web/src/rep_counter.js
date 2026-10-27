/** M34 - local, heuristic angle-cycle counter, NOT validated exercise coaching.
 *
 * Detects a complete high-low-high excursion with two sample confirmation
 * at both ends, stable visible arm/leg side, and plausible duration.
 * Thresholds are deliberately FIXED and not marketed as good-form targets.
 * Camera-plane angle measurements are viewpoint dependent. Counts are
 * observed candidate cycles, not exercise-valid reps or medical advice.
 *
 * No backend requests, video or landmark arrays are retained. Metrics
 * exist in volatile JS memory and are cleared on stop/camera/exercise change.
 */
import { measureExercise } from './exercise_feedback.js';

export const REP_POLICIES = Object.freeze({
  squat: Object.freeze({
    name: 'Squat',
    source: 'camera-plane knee angle',
    highDeg: 155, lowDeg: 120,
    minimumExcursionDeg: 35,
    hint: 'Side view: keep one hip, knee and ankle visible. Begin with your knee relatively straight; lower and rise. Count is an observed angle cycle, not squat depth or quality.'
  }),
  ohp: Object.freeze({
    name: 'Overhead Press',
    source: 'camera-plane elbow angle',
    highDeg: 155, lowDeg: 115,
    minimumExcursionDeg: 40,
    hint: 'Keep shoulder, elbow and wrist of one arm visible. A relatively straight-to-bent-to-straight elbow cycle may be counted; this does not validate a press or its loading.'
  }),
  barbell_row: Object.freeze({
    name: 'Barbell Row',
    source: 'camera-plane elbow angle, NOT torso tilt',
    highDeg: 155, lowDeg: 115,
    minimumExcursionDeg: 40,
    hint: 'Use a side/diagonal view with one shoulder, elbow and wrist visible during the pull. Torso inclination alone is NOT used as a repetition detector.'
  })
});

export function measureCycleAngle(exercise, landmarks) {
  if (!Object.hasOwn(REP_POLICIES, exercise)) return null;
  // Row repetitions are driven by elbow flexion, not back/torso tilt.
  // The OH Press and Row arm segment both use shoulder-elbow-wrist 2D geometry.
  const geometry = measureExercise(exercise === 'barbell_row' ? 'ohp' : exercise, landmarks);
  return geometry && { degrees: geometry.degrees, side: geometry.side };
}

export class RepCycleCounter {
  constructor({ sampleMs = 400, trackingGapMs = 1250,
    minCycleMs = 900, maxCycleMs = 12000, maxRecentCycles = 12 } = {}) {
    if (![sampleMs, trackingGapMs, minCycleMs, maxCycleMs, maxRecentCycles].every(Number.isFinite) ||
      sampleMs < 200 || trackingGapMs <= sampleMs ||
      minCycleMs < sampleMs * 2 || maxCycleMs <= minCycleMs ||
      maxRecentCycles < 1 || maxRecentCycles > 100) {
      throw new Error('Invalid M34 cycle-monitoring policy');
    }
    this.sampleMs = sampleMs;
    this.trackingGapMs = trackingGapMs;
    this.minCycleMs = minCycleMs;
    this.maxCycleMs = maxCycleMs;
    this.maxRecentCycles = maxRecentCycles;
    this.reset();
  }

  reset() {
    this.exercise = null;
    this.total = 0;
    this.recent = [];
    this.side = null;
    this.lastTime = null;
    this.lastTick = -Infinity;
    this.lastGood = null;
    this.lastAngle = null;
    this.#clearPending();
  }

  #clearPending() {
    this.phase = 'await_high';
    this.highConfirm = 0;
    this.lowConfirm = 0;
    this.returnConfirm = 0;
    this.startedAt = null;
    this.minimum = Infinity;
    this.maximum = -Infinity;
  }

  observe(exercise, landmarks, timestampMs) {
    if (!Object.hasOwn(REP_POLICIES, exercise)) return null;
    if (this.exercise !== exercise) {
      this.reset();
      this.exercise = exercise;
    }
    if (!Number.isFinite(timestampMs)) return this.summary();
    if (this.lastTime !== null && timestampMs < this.lastTime) {
      this.reset();
      this.exercise = exercise;
    }
    this.lastTime = timestampMs;
    // If one person leaves camera or a phone freezes, discard ONLY the
    // incomplete cycle, not a previously completed set.
    if (this.lastGood !== null && timestampMs - this.lastGood > this.trackingGapMs) {
      this.#clearPending();
      this.side = null;
      this.lastAngle = null;
      this.lastGood = null;
    }
    if (timestampMs - this.lastTick < this.sampleMs) return this.summary();
    this.lastTick = timestampMs;
    const measurement = measureCycleAngle(exercise, landmarks);
    if (!measurement) {
      this.lastAngle = null;
      return this.summary();
    }
    if (this.side !== null && this.side !== measurement.side) this.#clearPending();
    this.side = measurement.side;
    this.lastGood = timestampMs;
    this.lastAngle = measurement.degrees;

    const cfg = REP_POLICIES[exercise];
    const high = measurement.degrees >= cfg.highDeg;
    const low = measurement.degrees <= cfg.lowDeg;
    if (this.phase === 'await_high') {
      this.highConfirm = high ? this.highConfirm + 1 : 0;
      if (this.highConfirm >= 2) {
        this.phase = 'await_low';
        this.startedAt = timestampMs;
        this.maximum = measurement.degrees;
        this.minimum = measurement.degrees;
        this.highConfirm = 0;
      }
    } else if (this.phase === 'await_low') {
      if (this.startedAt !== null && timestampMs - this.startedAt > this.maxCycleMs) {
        this.#clearPending();
        return this.summary();
      }
      this.minimum = Math.min(this.minimum, measurement.degrees);
      this.maximum = Math.max(this.maximum, measurement.degrees);
      this.lowConfirm = low ? this.lowConfirm + 1 : 0;
      if (this.lowConfirm >= 2) {
        this.phase = 'await_return';
        this.returnConfirm = 0;
      }
    } else if (this.phase === 'await_return') {
      if (this.startedAt !== null && timestampMs - this.startedAt > this.maxCycleMs) {
        this.#clearPending();
        return this.summary();
      }
      this.minimum = Math.min(this.minimum, measurement.degrees);
      this.maximum = Math.max(this.maximum, measurement.degrees);
      this.returnConfirm = high ? this.returnConfirm + 1 : 0;
      if (this.returnConfirm >= 2) {
        const durationMs = timestampMs - this.startedAt;
        const amplitudeDeg = this.maximum - this.minimum;
        if (durationMs >= this.minCycleMs && durationMs <= this.maxCycleMs &&
            amplitudeDeg >= cfg.minimumExcursionDeg) {
          this.total++;
          this.recent.push({ durationMs, amplitudeDeg });
          if (this.recent.length > this.maxRecentCycles) this.recent.shift();
        }
        // A full high-low-high cycle has completed. Current high endpoint is
        // the start of the next cycle; it cannot count again without a new low.
        this.phase = 'await_low';
        this.startedAt = timestampMs;
        this.minimum = measurement.degrees;
        this.maximum = measurement.degrees;
        this.lowConfirm = 0;
        this.returnConfirm = 0;
      }
    }
    return this.summary();
  }

  summary() {
    if (!this.exercise) return null;
    const n = this.recent.length;
    const mean = (key) => n
      ? this.recent.reduce((total, rep) => total + rep[key], 0) / n : null;
    return {
      exercise: this.exercise,
      source: REP_POLICIES[this.exercise].source,
      cycles: this.total,
      phase: this.phase,
      tracking: this.lastAngle === null ? 'not_visible' : 'visible',
      angle: this.lastAngle,
      side: this.lastAngle === null ? null : this.side,
      lastDurationSec: n ? this.recent[n - 1].durationMs / 1000 : null,
      meanDurationSec: n ? mean('durationMs') / 1000 : null,
      meanExcursionDeg: mean('amplitudeDeg'),
      recentCycles: n,
      note: REP_POLICIES[this.exercise].hint,
      limitation: 'Approximate 2D movement cycles only. Counts may be missed or wrong due to viewpoint, occlusion, tempo or limited 400ms sampling. NOT verified repetitions, good/bad form, or injury advice.'
    };
  }
}
