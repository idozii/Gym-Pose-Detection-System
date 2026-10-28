/** 
 * W7 camera clock guard: ignore stale/coarsely quantized rAF callbacks.
 *
 * requestAnimationFrame timestamps describe the start of the browser paint
 * frame and can predate a session initialized with performance.now() in the
 * same paint frame. They also may repeat under timer precision reduction.
 * Never advance MediaPipe, W3, M34, M45 or W7 telemetry on such callbacks.
 *
 * This guard applies ONLY to live camera frames. Local MP4/WebM processing
 * retains its isolated media presentation time and W4 duplicate/seek gates.
 */
export function cameraFrameIsChronological({
  frameTimestampMs, sessionStartedAtMs, lastAnalyzedFrameMs
}) {
  return Number.isFinite(frameTimestampMs) &&
    Number.isFinite(sessionStartedAtMs) &&
    frameTimestampMs >= 0 && sessionStartedAtMs >= 0 &&
    frameTimestampMs >= sessionStartedAtMs &&
    (lastAnalyzedFrameMs === null ||
      (Number.isFinite(lastAnalyzedFrameMs) &&
        frameTimestampMs > lastAnalyzedFrameMs));
}
