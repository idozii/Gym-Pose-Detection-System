/**
 * W7b: minimal privacy-preserving manual QA evidence snapshot.
 *
 * Snapshot is generated exclusively on a user click. It contains only
 * aggregates of the most recent processed video frames, a source TYPE,
 * and exercise type. No raw frames, GT, landmarks, device IDs, IPs,
 * URLs, timestamps, clip path, filenames, or persistent storage.
 * No request is made over the network.
 */
export const QA_PROTOCOL = 'w7-local-browser-pose-qa-v1';
export const MIN_QA_FRAMES = 30;

export function createQaSnapshot(summary, {sourceMode, exercise}={}) {
  if (!summary || !['camera','video'].includes(sourceMode) ||
      !['squat','ohp','barbell_row','pushup'].includes(exercise) ||
      !Number.isSafeInteger(summary.windowFrames) ||
      summary.windowFrames < MIN_QA_FRAMES ||
      summary.windowFrames > 120 ||
      !Number.isSafeInteger(summary.totalFrames) ||
      summary.totalFrames < summary.windowFrames) {
    throw new RangeError('Collect at least 30 recent processed frames first');
  }
  const keys=['medianPoseMs','p95PoseMs','throughputFps'];
  if (keys.some(k=>!Number.isFinite(summary[k]) || summary[k]<0) ||
      !Number.isSafeInteger(summary.observedGapsOver750ms) ||
      summary.observedGapsOver750ms < 0 ||
      summary.observedGapsOver750ms >= summary.windowFrames) {
    throw new RangeError('W7 requires finite aggregate-only metrics');
  }
  const round=x=>Math.round(x*100)/100;
  return {
    protocol:QA_PROTOCOL,
    source_type:sourceMode,
    exercise_type:exercise,
    recent_processed_frames:summary.windowFrames,
    session_processed_frames:summary.totalFrames,
    pose_inference_median_ms:round(summary.medianPoseMs),
    pose_inference_p95_ms:round(summary.p95PoseMs),
    analyzed_frames_per_wall_second:round(summary.throughputFps),
    gaps_over_750ms_in_recent_window:summary.observedGapsOver750ms,
    device_or_browser_fingerprint_saved:false,
    raw_video_or_landmarks_saved:false,
    network_upload_performed:false,
    caveat:'Observational on-device browser processing; not camera FPS, file source dropped frames, rep accuracy, model parity, or safety validation.'
  };
}
