/** W1: human-readable camera failure guidance + privacy lifecycle rules.
 * No persistence, requests or hidden camera start.
 */
export function cameraErrorMessage(error) {
  const name = typeof error?.name === 'string' ? error.name : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError' ||
      name === 'SecurityError') {
    return 'Camera permission denied or blocked. Allow camera access in site settings, then retry on HTTPS or localhost.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No available camera was found. Connect a camera, check OS permissions and try again.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Camera appears busy. Close other camera applications or tabs, then retry.';
  }
  if (name === 'OverconstrainedError') {
    return 'Requested camera is not available at this resolution. Try the other camera or browser.';
  }
  if (name === 'AbortError') {
    return 'The camera was interrupted. Verify the device connection, then restart.';
  }
  return 'Camera or pose model could not start. Check browser support, HTTPS, permissions and the initial model download.';
}

export function shouldRevokeOnHidden({ hidden, running, starting }) {
  return Boolean(hidden && (running || starting));
}

export function shouldIgnoreEndedTrack({ cancelled, currentToken, startedToken }) {
  return Boolean(cancelled || startedToken !== currentToken);
}
