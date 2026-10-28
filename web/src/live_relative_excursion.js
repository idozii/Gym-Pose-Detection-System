/** W2: experimental live replay of the exact frozen M45 Squat rule.
 *
 * Mirrors M45 detectRelativeExcursionEvents incrementally using the same
 * original measureCycleAngle and M45 constants. NO GT, classifier,
 * backend requests, video history, landmark recording or score claims.
 *
 * The W2 website sampler nominally uses >=200ms between observations.
 * It is NOT parity evidence for Python Solutions vs JS Tasks Vision poses.
 * M34 stays separately running and unchanged at its 400ms default.
 */
import { measureCycleAngle, REP_POLICIES } from './rep_counter.js';

export const LIVE_M45_RULE = Object.freeze({
  highDeg: REP_POLICIES.squat.highDeg,
  minDropDeg: REP_POLICIES.squat.minimumExcursionDeg,
  minRiseDeg: REP_POLICIES.squat.minimumExcursionDeg,
  minReturnMs: 400,
  maxReturnMs: 12000,
  maxTrackingGapMs: 1250
});

export const LIVE_M45_SAMPLE_MS = 200;

export class LiveRelativeExcursionCounter {
  constructor({ sampleMs = LIVE_M45_SAMPLE_MS } = {}) {
    if (!Number.isFinite(sampleMs) || sampleMs < 200) {
      throw new RangeError('W2 requires an experimental sample cadence >=200ms');
    }
    this.sampleMs = sampleMs;
    this.reset();
  }

  reset() {
    this.cycles = 0;
    this.events = []; // bounded ephemeral recent event durations ONLY
    this.lastInput = null;
    this.lastSample = null;
    this.lastGood = null;
    this.lastSide = null;
    this.peak = null;
    this.lowAt = null;
    this.minDeg = Infinity;
    this.currentAngle = null;
    this.tracking = 'not_visible';
    this.sampledFrames = 0;
  }

  clearPending() {
    this.peak = null;
    this.lowAt = null;
    this.minDeg = Infinity;
  }

  /** Process a frame already selected by an external fixed sampler.
   * This method is semantically equivalent to ONE iteration of the
   * frozen M45 TRAIN comparator's original iteration through frames.
   */
  consumeSample(landmarks, timestampMs) {
    if (!Number.isFinite(timestampMs) ||
        (this.lastSample !== null && timestampMs <= this.lastSample)) {
      throw new TypeError('W2 needs chronological finite sample times');
    }
    this.lastInput = timestampMs;
    this.lastSample = timestampMs;
    this.sampledFrames += 1;
    if (this.lastGood !== null &&
        timestampMs - this.lastGood > LIVE_M45_RULE.maxTrackingGapMs) {
      this.clearPending();
      this.lastSide = null;
      this.lastGood = null;
    }
    const angle = measureCycleAngle('squat', landmarks);
    if (!angle) {
      this.tracking = 'not_visible';
      this.currentAngle = null;
      return this.summary();
    }
    if (this.lastSide !== null && angle.side !== this.lastSide) {
      this.clearPending();
    }
    this.lastGood = timestampMs;
    this.lastSide = angle.side;
    this.tracking = 'visible';
    this.currentAngle = angle.degrees;
    const v = angle.degrees;

    if (this.lowAt !== null && timestampMs - this.lowAt > LIVE_M45_RULE.maxReturnMs) {
      this.clearPending();
    }
    if (this.lowAt === null) {
      if (v >= LIVE_M45_RULE.highDeg) {
        this.peak = Math.max(this.peak ?? -Infinity, v);
      }
      if (this.peak !== null && this.peak - v >= LIVE_M45_RULE.minDropDeg) {
        this.lowAt = timestampMs;
        this.minDeg = v;
      }
      return this.summary();
    }

    this.minDeg = Math.min(this.minDeg, v);
    if (timestampMs - this.lowAt >= LIVE_M45_RULE.minReturnMs &&
        v >= LIVE_M45_RULE.highDeg &&
        v - this.minDeg >= LIVE_M45_RULE.minRiseDeg) {
      const observedMs = timestampMs - this.lowAt;
      this.cycles += 1;
      this.events.push(observedMs);
      if (this.events.length > 12) this.events.shift();
      this.clearPending();
      this.peak = v; // same next-cycle high anchor as frozen M45
    }
    return this.summary();
  }

  /** Live input sampling; no queues or frame uploads. */
  observe(landmarks, timestampMs) {
    if (!Number.isFinite(timestampMs)) return this.summary();
    if (this.lastInput !== null && timestampMs < this.lastInput) {
      // Camera/clock rollback must never add events to an old session.
      this.reset();
    }
    this.lastInput = timestampMs;
    if (this.lastSample !== null &&
        timestampMs - this.lastSample < this.sampleMs) {
      return this.summary();
    }
    return this.consumeSample(landmarks, timestampMs);
  }

  summary() {
    const n = this.events.length;
    return {
      cycles: this.cycles,
      tracking: this.tracking,
      angle: this.currentAngle,
      phase: this.lowAt === null ? 'await_drop' : 'await_return',
      sampleMs: this.sampleMs,
      sampledFrames: this.sampledFrames,
      meanRecoverySec: n
        ? this.events.reduce((sum, ms) => sum + ms, 0) / n / 1000
        : null,
      limitation: 'Experimental same-pose Squat cycle count; no GT, form validation, or browser/Python parity claim.'
    };
  }
}

export const RESEARCH_DEMO_MODES = Object.freeze(['compare', 'm34', 'm45']);

export function isResearchSquat(exercise) {
  return exercise === 'squat';
}
