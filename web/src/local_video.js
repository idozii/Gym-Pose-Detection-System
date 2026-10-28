/** W4 - file video input restrictions and timestamps, no network or storage.
 *
 * Uploaded video stays a browser-owned File via URL.createObjectURL;
 * W4 never sends File, frames, pose landmarks or file names to the API.
 */
export const W4_MAX_VIDEO_BYTES = 250 * 1024 * 1024;
export const W4_MAX_VIDEO_SECONDS = 300;
const ALLOWED_TYPES = Object.freeze(['video/mp4','video/webm']);
const ALLOWED_SUFFIXES = Object.freeze(['.mp4','.webm']);

export function validateLocalVideoFile(file) {
  if (!file || typeof file.name !== 'string' ||
      !Number.isSafeInteger(file.size) || file.size <= 0 ||
      file.size > W4_MAX_VIDEO_BYTES) {
    return { ok: false, reason: 'Select an MP4 or WebM file no larger than 250 MB.' };
  }
  const name = file.name.toLowerCase();
  const extOk = ALLOWED_SUFFIXES.some((ext) => name.endsWith(ext));
  const typeOk = file.type === '' || ALLOWED_TYPES.includes(file.type.toLowerCase());
  if (!extOk || !typeOk) {
    return { ok: false, reason: 'Only MP4 and WebM video files are supported.' };
  }
  return { ok: true, reason: '' };
}

export function validateVideoMetadata({ duration, width, height }) {
  if (!Number.isFinite(duration) || duration <= 0 ||
      duration > W4_MAX_VIDEO_SECONDS) {
    return { ok: false, reason: 'Video duration must be between 0 and 5 minutes.' };
  }
  if (!Number.isInteger(width) || !Number.isInteger(height) ||
      width <= 0 || height <= 0 || width > 7680 || height > 4320) {
    return { ok: false, reason: 'Cannot read a valid video picture size.' };
  }
  return { ok: true, reason: '' };
}

/** PTS seconds are from the selected HTMLVideoElement, NEVER from wall clock. */
export function localVideoTimestampMs(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new RangeError('W4 media time must be nonnegative and finite');
  }
  return Math.round(seconds * 1000);
}

export function localFrameIsNew({ currentTime, previousTime, timestampMs, previousTimestampMs }) {
  return Number.isFinite(currentTime) && currentTime >= 0 &&
    (previousTime === -1 || currentTime > previousTime) &&
    (previousTimestampMs === null || timestampMs > previousTimestampMs);
}

/** Remote model inference is exclusive to an explicitly consented live webcam. */
export function canRequestRemoteInference({ sourceMode, enabled, running }) {
  return sourceMode === 'camera' && enabled === true && running === true;
}
