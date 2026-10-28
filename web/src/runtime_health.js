/** W7: privacy-preserving, bounded on-device browser pose performance diagnostic.
 *
 * Measures synchronous detectForVideo CPU wall duration from performance.now().
 * Rate is processed video frames PER ELAPSED WALL SECOND, never nominal video
 * FPS, camera sensor FPS, successful pose fraction or a model confidence.
 * No video/landmarks, timestamps from the source file, identity, URLs,
 * telemetry uploads, localStorage, cookies or persistence.
 */
export const W7_WINDOW = 120;
export const W7_MAX_INTERVAL_MS = 10_000;

export function percentile(sorted, fraction) {
  if (!Array.isArray(sorted) || !sorted.length ||
      fraction < 0 || fraction > 1 || !Number.isFinite(fraction) ||
      sorted.some((x,i) => !Number.isFinite(x) || x < 0 ||
        (i > 0 && x < sorted[i - 1]))) return null;
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index), upper = Math.ceil(index);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

export function describePerformance({samples,medianMs,p95Ms,rate,stalls}) {
  if (!samples) return 'WAITING FOR FRAMES';
  if (stalls > 0) return 'PROCESSING GAP OBSERVED';
  if (p95Ms !== null && p95Ms > 80) return 'HIGH POSE LATENCY';
  if (rate !== null && rate < 8 && samples >= 30) return 'LOW THROUGHPUT';
  return 'OBSERVING';
}

export class RuntimeHealth {
  constructor({maxSamples=W7_WINDOW}={}) {
    if (!Number.isSafeInteger(maxSamples) || maxSamples < 5 ||
        maxSamples > W7_WINDOW) throw new RangeError('W7 bounded samples');
    this.maxSamples=maxSamples;
    this.reset();
  }
  reset() {
    this.times=[];
    this.poseDurations=[];
    this.stalls=0;
    this.totalFrames=0;
  }

  observe({wallTimeMs,poseDurationMs}) {
    if (!Number.isFinite(wallTimeMs) || wallTimeMs < 0 ||
        !Number.isFinite(poseDurationMs) || poseDurationMs < 0 ||
        poseDurationMs > 30000 ||
        (this.times.length && wallTimeMs < this.times.at(-1))) {
      throw new RangeError('W7 requires monotonic wall clock and finite pose latency');
    }
    if (this.times.length &&
        wallTimeMs - this.times.at(-1) > 750) this.stalls += 1;
    this.times.push(wallTimeMs);
    this.poseDurations.push(poseDurationMs);
    if (this.times.length > this.maxSamples) {
      this.times.shift();
      this.poseDurations.shift();
    }
    this.totalFrames += 1;
  }

  summary() {
    const n=this.times.length;
    const sorted=[...this.poseDurations].sort((a,b)=>a-b);
    const elapsed=n>1?this.times.at(-1)-this.times[0]:0;
    const rate=elapsed>0?(n-1)*1000/elapsed:null;
    const medianMs=percentile(sorted,.5);
    const p95Ms=percentile(sorted,.95);
    // Only recent samples belong to the visible diagnostic window.
    // A historical startup stall must not poison later steady-state QA.
    const recentGaps = this.times.slice(1).reduce(
      (count, t, i) => count + Number(t - this.times[i] > 750), 0
    );
    return {
      windowFrames:n,
      totalFrames:this.totalFrames,
      throughputFps:rate,
      medianPoseMs:medianMs,
      p95PoseMs:p95Ms,
      maxPoseMs:n?sorted.at(-1):null,
      observedGapsOver750ms:recentGaps,
      status:describePerformance({samples:n,medianMs,p95Ms,rate,stalls:recentGaps}),
      interpretation:'Local on-device sampled CPU pose timing, not sensor FPS, MediaPipe accuracy, or dropped source frames.'
    };
  }
}
