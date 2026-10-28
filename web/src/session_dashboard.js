/** W3 - volatile local workout-session telemetry; no video/pose history.
 *
 * A "ready" frame means required image-plane joints were sufficiently
 * visible for the existing pose readiness gate, NOT that a repetition
 * is correct or that a classifier has high confidence.
 * Sampling statistics count analyzed video frames, NOT all camera FPS.
 */

export const W3_MAX_TEMPO_BARS = 12;
export const W3_STALE_FRAME_MS = 1500;

export function clockLabel(elapsedMs) {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return '00:00';
  const total = Math.floor(elapsedMs / 1000);
  const seconds = String(total % 60).padStart(2, '0');
  const minutes = String(Math.floor(total / 60) % 60).padStart(2, '0');
  if (total < 3600) return minutes + ':' + seconds;
  return String(Math.floor(total / 3600)).padStart(2, '0') + ':' + minutes + ':' + seconds;
}

export function percentage(numerator, denominator) {
  if (!Number.isSafeInteger(numerator) || numerator < 0 ||
      !Number.isSafeInteger(denominator) || denominator < 0 ||
      numerator > denominator) {
    throw new RangeError('W3 frame totals must be nonnegative and consistent');
  }
  return denominator === 0 ? null : Math.round(numerator / denominator * 100);
}

/** Display-only bars; never passes raw pose frames to chart or storage.
 * Each series uses its OWN maximum, not a shared comparable scale.
 */
export function tempoBars(durationsSec, maxBars = W3_MAX_TEMPO_BARS) {
  if (!Number.isSafeInteger(maxBars) || maxBars < 1 || maxBars > 40 ||
      !Array.isArray(durationsSec)) {
    throw new TypeError('W3 requires a bounded list of cycle-time values');
  }
  const recent = durationsSec.filter((v) =>
    typeof v === 'number' && Number.isFinite(v) && v > 0 && v <= 60
  ).slice(-maxBars);
  if (!recent.length) return [];
  const maximum = Math.max(...recent);
  return recent.map((durationSec, i) => ({
    index: i + 1,
    durationSec,
    heightPercent: Math.max(9, Math.round(durationSec / maximum * 100))
  }));
}

export function tempoFromM34(recent) {
  if (!Array.isArray(recent)) return [];
  return recent.map((x) =>
    x && Number.isFinite(x.durationMs) ? x.durationMs / 1000 : NaN
  ).filter((x) => Number.isFinite(x) && x > 0);
}

export function tempoFromM45(events) {
  if (!Array.isArray(events)) return [];
  return events.map((ms) => Number.isFinite(ms) ? ms / 1000 : NaN)
    .filter((x) => Number.isFinite(x) && x > 0);
}

export class SessionDashboard {
  constructor() { this.reset(); }

  reset() {
    this.running = false;
    this.startedAtMs = null;
    this.lastFrameMs = null;
    this.sampleFrames = 0;
    this.poseSeenFrames = 0;
    this.requiredReadyFrames = 0;
    this.lastPoseSeen = false;
    this.lastReady = false;
  }

  start(timestampMs) {
    if (!Number.isFinite(timestampMs) || timestampMs < 0) {
      throw new TypeError('W3 session start must use a finite monotone clock');
    }
    this.reset();
    this.running = true;
    this.startedAtMs = timestampMs;
  }

  observe(timestampMs, { poseSeen, requiredReady } = {}) {
    if (!this.running || this.startedAtMs === null) return;
    if (!Number.isFinite(timestampMs) || timestampMs < this.startedAtMs ||
        (this.lastFrameMs !== null && timestampMs <= this.lastFrameMs)) {
      throw new RangeError('W3 requires chronological video frame timestamps');
    }
    if (typeof poseSeen !== 'boolean' || typeof requiredReady !== 'boolean' ||
        (requiredReady && !poseSeen)) {
      throw new TypeError('W3 requires honest pose and required-joint visibility flags');
    }
    this.sampleFrames += 1;
    if (poseSeen) this.poseSeenFrames += 1;
    if (requiredReady) this.requiredReadyFrames += 1;
    this.lastPoseSeen = poseSeen;
    this.lastReady = requiredReady;
    this.lastFrameMs = timestampMs;
  }

  summary(timestampMs) {
    if (!this.running || this.startedAtMs === null) {
      return {
        running: false, elapsedMs: 0, elapsedLabel: '00:00',
        sampleFrames: 0, poseSeenPct: null, requiredReadyPct: null,
        status: 'CAMERA OFFLINE'
      };
    }
    const now = Number.isFinite(timestampMs)
      ? Math.max(this.startedAtMs, timestampMs) : this.startedAtMs;
    let status = 'WAITING FOR VIDEO';
    if (this.lastFrameMs !== null) {
      if (now - this.lastFrameMs > W3_STALE_FRAME_MS) status = 'VIDEO STALLED';
      else if (!this.lastPoseSeen) status = 'NO POSE VISIBLE';
      else if (!this.lastReady) status = 'LIMITED JOINT VIEW';
      else status = 'JOINTS VISIBLE';
    }
    const elapsedMs = now - this.startedAtMs;
    return {
      running: true, elapsedMs, elapsedLabel: clockLabel(elapsedMs),
      sampleFrames: this.sampleFrames,
      poseSeenPct: percentage(this.poseSeenFrames, this.sampleFrames),
      requiredReadyPct: percentage(this.requiredReadyFrames, this.sampleFrames),
      status
    };
  }
}
