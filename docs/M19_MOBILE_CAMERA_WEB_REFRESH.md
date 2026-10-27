# M19 - Mobile camera controls and visual redesign

## New mobile and desktop controls

- **Front / Selfie**: asks MediaDevices for the front-facing user camera.
  Preview is mirrored by default and can be manually unmirrored.
- **Rear / Observer**: asks MediaDevices for the environment-facing camera.
  Preview is always unmirrored, suitable for filming another athlete with
  consent while the phone is positioned safely.
- **Mirror toggle**: affects the **canvas presentation only**. All 33
  MediaPipe landmark X/Y/Z/visibility values remain in raw estimator
  coordinate space; the RF features and scoring payload are unchanged.
- **Switch cameras**: closes the old video tracks before opening the new
  physical camera. iOS Safari often cannot hold front and rear streams
  concurrently. Any research inference consent is revoked on switch;
  the user has to opt in again.
- **Safe fallback**: facingMode is an ideal preference; browsers may
  silently choose a different device. If MediaStreamTrack.getSettings()
  reveals a mismatch, the UI states the fallback honestly rather than
  claiming the rear camera is active.
- **Privacy**: webcam pixels never upload. No recording or photos. Only
  33 numeric pose landmarks may be sent to the owner-controlled API
  following separate explicit consent. Switching cameras, stopping the
  preview or hiding the tab ends that consent.

## UI redesign

The page uses a new dark movement-studio style with volt-lime accent,
oversized two-line headline, annotated live canvas, responsive controls,
high-contrast data cards, branded status states, clearer model cards,
research disclaimer, keyboard focus rings, and reduced-motion preference.
On narrow screens the Start button is full width and the physical-camera
selector + selfie-mirror toggle are accessible thumb-sized controls.

This does NOT expand the supported inference exercises. The original six
source-label models still cover Squat, OHP, Barbell Row only.

## Acceptance checklist (physical device required)

1. Run `npm test --prefix web` and `npm run build --prefix web`.
2. On **iPhone Safari**, open site via HTTPS, allow camera. Select
   Front, start preview. Mirror on should match selfie expectations;
   Mirror off should flip **only the displayed canvas**, not alter
   the score JSON schema.
3. Switch to Rear while camera is running. Prior physical video track
   must stop, new track opens, canvas is **not** mirrored; no automatic
   research inference consent is retained. Confirm an unavailable rear
   camera shows a fallback message instead of false assurance.
4. Repeat on **Android Chrome** (front, rear, mirror on/off) and
   desktop Chrome/Firefox (single-camera graceful fallback).
5. Verify changing exercises clears old inference readings and new
   camera does not retain an older result.
6. Verify HTTPS, MediaPipe CDN/model availability, and privacy:
   camera frames are never uploaded; no inference POST until consent.
7. Recheck actual side/front/diagonal camera effects on model feature
   parity before inferring movement-quality stability.

CI tests cover deterministic camera helpers, frontend control wiring,
responsive CSS markers and typography. **CI cannot simulate mobile
hardware or replace manual browser acceptance**.

## ASCII hyphen normalization

All tracked plain-text Markdown docs and software source files are
checked for Unicode en dash (U+2013) and em dash (U+2014) and
normalized to simple ASCII hyphen. This is cosmetic prose cleanup,
**not** replacement of technical `--option` command flags, negative
numbers, double-minus operators, CSS `--custom-properties`, versioned
schema identifiers, or historical binary/notebook assets.

## Further exercise research

Read [M19 exercise dataset expansion](M19_EXERCISE_DATASET_EXPANSION.md)
for new publicly documented training sources, subject-separated splits,
licensing and explicit warnings about exercise-class versus form-error labels.
