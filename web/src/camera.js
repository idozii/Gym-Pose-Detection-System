/** Camera selection + display-only selfie mirroring. No inference transforms. */
export const CAMERA_FACING = Object.freeze({ FRONT: 'user', REAR: 'environment' });

export function cameraConstraints(facingMode = CAMERA_FACING.FRONT) {
  if (!Object.values(CAMERA_FACING).includes(facingMode)) throw new Error('Unsupported camera facing mode');
  return {
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };
}

export function mirrorPreview(facingMode, mirrorSelfie) {
  // Mirroring is DISPLAY-ONLY. Never flip model input landmark coordinates.
  return facingMode === CAMERA_FACING.FRONT && mirrorSelfie === true;
}

export function cameraFallbackMessage(requested, actual) {
  if (!actual || requested === actual) return '';
  return requested === CAMERA_FACING.REAR
    ? 'Rear camera not available. The browser selected a different camera.'
    : 'Front camera not available. The browser selected a different camera.';
}
