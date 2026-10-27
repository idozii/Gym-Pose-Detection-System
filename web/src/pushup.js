/** Push-up local *geometry preview* - NOT a form classifier or rep counter.
 * Pixels and landmarks are processed in-browser only. No API calls.
 * Angular values are camera-plane estimates and change with camera viewpoint.
 */
export const PUSHUP_SIDE_INDICES = Object.freeze([
  Object.freeze({ name: 'LEFT', joints: Object.freeze([11, 13, 15, 23]) }),
  Object.freeze({ name: 'RIGHT', joints: Object.freeze([12, 14, 16, 24]) })
]);

export function pushupElbowAngle(landmarks, minVisibility = 0.5) {
  if (!Array.isArray(landmarks) || landmarks.length !== 33) return null;
  if (!Number.isFinite(minVisibility) || minVisibility < 0 || minVisibility > 1) return null;
  const checks = PUSHUP_SIDE_INDICES.map((side) => {
    const valid = side.joints.every((j) => {
      const p = landmarks[j];
      return p && [p.x, p.y, p.z, p.visibility].every(Number.isFinite)
        && p.visibility >= minVisibility && p.visibility <= 1;
    });
    return valid ? { ...side, confidence: Math.min(...side.joints.map((j) => landmarks[j].visibility)) } : null;
  }).filter(Boolean);
  if (!checks.length) return null;
  const side = checks.sort((a, b) => b.confidence - a.confidence)[0];
  const [shoulder, elbow, wrist] = side.joints.map((j) => landmarks[j]);
  const u = { x: shoulder.x - elbow.x, y: shoulder.y - elbow.y };
  const v = { x: wrist.x - elbow.x, y: wrist.y - elbow.y };
  const denominator = Math.hypot(u.x, u.y) * Math.hypot(v.x, v.y);
  if (!Number.isFinite(denominator) || denominator < 1e-6) return null;
  const cosine = Math.max(-1, Math.min(1, (u.x * v.x + u.y * v.y) / denominator));
  const angle = Math.acos(cosine) * 180 / Math.PI;
  if (!Number.isFinite(angle)) return null;
  return { degrees: angle, side: side.name, evidence: 'camera-plane angle only' };
}

export class PushupMotionWindow {
  constructor({ durationMs = 10000, minIntervalMs = 500 } = {}) {
    if (durationMs < 1000 || minIntervalMs < 100) throw new Error('Invalid monitoring window');
    this.durationMs = durationMs;
    this.minIntervalMs = minIntervalMs;
    this.reset();
  }

  reset() {
    this.samples = [];
    this.lastSampleAt = -Infinity;
  }

  observe(landmarks, timestampMs) {
    if (!Number.isFinite(timestampMs)) return this.summary(timestampMs);
    // Drop expired samples even when the user's pose disappears.
    this.samples = this.samples.filter((entry) => timestampMs >= entry.t
      && timestampMs - entry.t <= this.durationMs);
    if (timestampMs - this.lastSampleAt < this.minIntervalMs) return this.summary(timestampMs);
    this.lastSampleAt = timestampMs;
    const angle = pushupElbowAngle(landmarks);
    if (angle) this.samples.push({ t: timestampMs, angle: angle.degrees, side: angle.side });
    return this.summary(timestampMs);
  }

  summary(timestampMs) {
    if (!Number.isFinite(timestampMs)) return { angle: null, range: null, count: 0, side: null };
    const recent = this.samples.filter((entry) => timestampMs >= entry.t
      && timestampMs - entry.t <= this.durationMs);
    const latest = recent.length ? recent[recent.length - 1] : null;
    // Do not show stale "live" angles if tracking is lost.
    const active = latest && timestampMs - latest.t < 1300 ? latest : null;
    const values = recent.map((entry) => entry.angle);
    return {
      angle: active?.angle ?? null,
      side: active?.side ?? null,
      range: values.length >= 4 ? Math.max(...values) - Math.min(...values) : null,
      count: recent.length,
    };
  }
}
