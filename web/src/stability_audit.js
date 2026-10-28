/** W8 session-level, bounded performance QA. Pure on-device, no persistent state.
 * Runs only on already processed pose frames; never holds video or landmark data.
 * Each complete non-overlapping 120-frame window becomes four scalar metrics.
 * The oldest window summaries are evicted after 16; no per-frame history
 * survives completion. Session-wide gap count is an observation, NOT drops.
 */
import { percentile } from './runtime_health.js';

export const W8_FRAMES_PER_WINDOW = 120;
export const W8_RETAINED_WINDOWS = 16;
export const W8_MIN_COMPLETED_WINDOWS = 2;
export const W8_PROTOCOL = 'w8-local-session-stability-qa-v1';

export class StabilityAudit {
  constructor({windowSize=W8_FRAMES_PER_WINDOW, maxWindows=W8_RETAINED_WINDOWS}={}) {
    if (!Number.isSafeInteger(windowSize) || windowSize < 5 ||
        windowSize > W8_FRAMES_PER_WINDOW ||
        !Number.isSafeInteger(maxWindows) || maxWindows < 2 ||
        maxWindows > W8_RETAINED_WINDOWS) {
      throw new RangeError('W8 requires strictly bounded windows');
    }
    this.windowSize=windowSize;
    this.maxWindows=maxWindows;
    this.reset();
  }
  reset() {
    this.windowDurations=[];
    this.windowStartMs=null;
    this.windowLastMs=null;
    this.windowGaps=0;
    this.lastWallMs=null;
    this.sessionFrames=0;
    this.sessionGaps=0;
    this.completedWindowCount=0;
    this.windows=[];
  }
  observe({wallTimeMs,poseDurationMs}) {
    if (!Number.isFinite(wallTimeMs) || wallTimeMs < 0 ||
        !Number.isFinite(poseDurationMs) || poseDurationMs < 0 ||
        poseDurationMs > 30000 ||
        (this.lastWallMs !== null && wallTimeMs < this.lastWallMs)) {
      throw new RangeError('W8 requires finite monotonic observed timings');
    }
    const gap=this.lastWallMs !== null && wallTimeMs-this.lastWallMs > 750;
    if (gap) {
      this.sessionGaps++;
      this.windowGaps++;
    }
    this.lastWallMs=wallTimeMs;
    if (this.windowStartMs === null) this.windowStartMs=wallTimeMs;
    this.windowLastMs=wallTimeMs;
    this.windowDurations.push(poseDurationMs);
    this.sessionFrames++;
    if (this.windowDurations.length === this.windowSize) {
      const sorted=[...this.windowDurations].sort((a,b)=>a-b);
      const elapsed=this.windowLastMs-this.windowStartMs;
      const throughput=elapsed>0?(this.windowSize-1)*1000/elapsed:null;
      this.windows.push({
        medianMs:percentile(sorted,.5),
        p95Ms:percentile(sorted,.95),
        throughputFps:throughput,
        gaps:this.windowGaps
      });
      if (this.windows.length>this.maxWindows) this.windows.shift();
      this.completedWindowCount++;
      this.windowDurations=[];
      this.windowStartMs=null;
      this.windowLastMs=null;
      this.windowGaps=0;
    }
  }
  summary() {
    const range=key=>{
      const values=this.windows.map(w=>w[key]).filter(Number.isFinite);
      return values.length ? [Math.min(...values),Math.max(...values)] : null;
    };
    return {
      sessionFrames:this.sessionFrames,
      completedWindows:this.completedWindowCount,
      retainedWindows:this.windows.length,
      framesPerWindow:this.windowSize,
      sessionObservedGapsOver750ms:this.sessionGaps,
      retainedWindowGaps:this.windows.reduce((n,w)=>n+w.gaps,0),
      medianRangeMs:range('medianMs'),
      p95RangeMs:range('p95Ms'),
      throughputRangeFps:range('throughputFps')
    };
  }
}

export function createStabilitySnapshot(summary,{sourceMode,exercise}={}) {
  if (!['camera','video'].includes(sourceMode) ||
      !['squat','ohp','barbell_row','pushup'].includes(exercise) ||
      !summary || summary.framesPerWindow !== W8_FRAMES_PER_WINDOW ||
      !Number.isSafeInteger(summary.sessionFrames) ||
      !Number.isSafeInteger(summary.completedWindows) ||
      !Number.isSafeInteger(summary.retainedWindows) ||
      summary.completedWindows < W8_MIN_COMPLETED_WINDOWS ||
      summary.retainedWindows < W8_MIN_COMPLETED_WINDOWS ||
      summary.retainedWindows > W8_RETAINED_WINDOWS ||
      summary.retainedWindows > summary.completedWindows ||
      summary.sessionFrames < summary.completedWindows*W8_FRAMES_PER_WINDOW ||
      !Number.isSafeInteger(summary.sessionObservedGapsOver750ms) ||
      summary.sessionObservedGapsOver750ms < 0 ||
      summary.sessionObservedGapsOver750ms >= summary.sessionFrames ||
      !Number.isSafeInteger(summary.retainedWindowGaps) ||
      summary.retainedWindowGaps < 0 ||
      summary.retainedWindowGaps > summary.sessionObservedGapsOver750ms) {
    throw new RangeError('W8 needs at least two valid completed 120-frame windows');
  }
  for(const key of ['medianRangeMs','p95RangeMs','throughputRangeFps']) {
    const range=summary[key];
    if (!Array.isArray(range) || range.length !== 2 ||
        !range.every(v=>Number.isFinite(v) && v>=0) ||
        range[0]>range[1]) throw new RangeError('W8 invalid aggregate range');
  }
  const round=x=>Math.round(x*100)/100;
  const roundRange=x=>x.map(round);
  return {
    protocol:W8_PROTOCOL,
    source_type:sourceMode,
    exercise_type:exercise,
    session_processed_frames:summary.sessionFrames,
    completed_120_frame_windows:summary.completedWindows,
    retained_recent_windows:summary.retainedWindows,
    median_pose_latency_range_ms:roundRange(summary.medianRangeMs),
    p95_pose_latency_range_ms:roundRange(summary.p95RangeMs),
    analyzed_throughput_range_fps:roundRange(summary.throughputRangeFps),
    session_observed_processing_gaps_over_750ms:summary.sessionObservedGapsOver750ms,
    retained_window_gaps_over_750ms:summary.retainedWindowGaps,
    measurement_scope:'non-overlapping 120-processed-frame windows, most recent 16 retained',
    hardware_or_browser_identity_collected:false,
    persistent_storage_used:false,
    network_upload_performed:false,
    caveat:'Observational browser processing only; not sensor FPS, long-run memory proof, dropped frames, repetition accuracy, or hardware certification.'
  };
}
