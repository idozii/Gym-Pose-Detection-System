# W4 - Privacy-preserving Local Video Analysis (browser only)

## What W4 adds

The same camera-plane MediaPipe Tasks Vision workbench now accepts
one **user-selected local MP4 or WebM file**. The user chooses between
"Live camera" and "Local video file"; selecting a file never opens
the webcam or calls the Python model backend.

In local video mode the interface provides:
- Explicit local file picker: only MP4/WebM, <=250 MiB and
  video duration <=300 seconds, with finite readable dimensions.
  Browser codec support varies; file type alone is not a guarantee
  the browser can decode its video codec.
- "Play video", "Pause video", "Replay from start", "Clear video"
  buttons and non-seekable playback progress. Playback does NOT
  auto-start on file selection.
- Local video canvas pose overlay, same M34 counter and M45
  experimental Squat-only relative-excursion counter, same
  W3 observational visibility/tempo/dashboard. Push-up remains
  an unscored geometry preview.
- Pose inference uses the video's **own currentTime in ms**,
  never requestAnimationFrame wall time. Missed frames due to
  browser performance are not reconstructed or fabricated.
- On replay, W4 creates a new fresh MediaPipe tracker and resets
  M34, M45, W3, and the media playback timeline. On any backwards
  video seek (even if from an external script), the analysis
  pauses and requires a fresh replay; stale MediaPipe temporal
  tracking is never silently reused. Users can pause/resume
  without invalidating the forward-only tracking timeline.
- W3 session elapsed becomes "ANALYSIS ELAPSED" relative to the
  set reset point on the media clock, excluding wall-clock
  time spent paused. The dedicated playback progress indicator
  reports the actual current file position.
- Camera/video source switches tear down existing media/pose
  resources. Hidden tab and pagehide use the W1 privacy
  lifecycle. Choosing a different file, clearing or leaving
  the view revokes the temporary URL via URL.revokeObjectURL.

## Data privacy boundary

W4 creates a temporary local object URL (blob:) using
URL.createObjectURL(file) and assigns it to an HTML video
element. This is browser-local; the File bytes are never
posted to a server or committed to Git. W4 does not save
video blobs, filenames, landmark arrays or source
timestamps in localStorage, IndexedDB or sessionStorage.

Remote research model inference is **disabled in local video
mode** even if an earlier camera session had consent enabled:
switching input stops the previous camera, aborts pending
requests, clears consent, disables that checkbox and guards
the network-scoring function against any W4 media source.
MediaPipe model and WASM runtime are still downloaded from
the configured Google/CDN locations on first use.
The existing camera-only, explicitly consented six RF model
API remains unchanged.

**Scope limitation:** W4 is for *local* videos the user has
permission to analyze. It is not a video-upload service, nor
an external dataset evaluator. It must not be used to scan
official publisher TEST/validation clips outside the
frozen evaluation protocol. No RepCount archive, source
annotation/GT file or training data is bundled or accessed
by the public website.

No model is retrained. All 17 research source files frozen
at M48 remain unchanged. W4 does not claim browser
MediaPipe Tasks Vision equals the offline Python MediaPipe
Solutions pose estimator, does not claim true rep accuracy,
and does not assess safe/correct exercise form.

## Local verification on Ubuntu

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
cd web
npm install
npm test
npm run build
npm run dev
~~~

Visit localhost:5173. Manual acceptance:
1. Choose "Local video file". The camera must stop if it had
   been running; the Python research inference checkbox
   becomes disabled. No getUserMedia permission prompt.
2. Choose a SHORT video containing a consented test
   Squat movement, ideally H.264 MP4 or supported WebM.
   The app must show LOCAL VIDEO READY after metadata
   and model loading; no automatic playback.
3. Press Play. The video should appear with skeleton
   overlay, experimental M34/M45 Squat counts and
   W3 frame visibility and analysis-time measurements.
   All values are observational, not validated GT reps.
4. Press Pause; file progress, W3 analysis elapsed and
   counts remain unchanged. Press Play to resume.
5. Press Replay from start. The video and both counters
   restart at zero with a fresh MediaPipe task; the
   previous W3 metrics and pose frame history disappear.
6. Click Reset set during playback; live counting resets
   at the current video position, not at 0. Use Replay
   to start the entire file again.
7. Select OHP or Row; M45-only panel hides. Select
   Push-up; the original no-rep local preview remains.
8. Switch from local file to Camera; then camera access
   must require a user click. Test on mobile HTTPS.
9. Try a file >250 MiB, invalid extension, duration
   >5 minutes, corrupted file and unsupported codec.
   The app should reject without uploading any bytes.
10. Clear the file, switch modes and hide the browser
    tab; object URL/pose state must be released. Re-run
    in a narrow mobile viewport.

Do not use private training subjects, TEST videos or
publication-restricted media to test a public preview.
A generic short video you own is sufficient.

## CI and deployment

The Node tests cover:
- File constraints, video metadata boundaries and clock
  units;
- Monotonic sampled media time and rewind isolation;
- The explicit camera-only remote inference gate;
- DOM control wiring, temporary object URL revocation,
  source-switch lifecycle and no network/storage in
  local video loading;
- All previous W1-W3 and M34/M45 research tests.

Web CI also builds the Vite static site. This is not
browser E2E, GPU/CPU device compatibility or webcam
certification. Verify real file decode on your Ubuntu
machine/mobile browser before promising general support.

For Vercel production, deploy the actual reviewed
and merged W4 main commit SHA, not a stale preview
branch. Validate latest gitSource SHA, project
framework='vite', rootDirectory='web', build command
and deployment state. Never promote an old
pre-W4 preview or release an unsuccessful build.

## Known limitations and follow-up

Large videos and synchronous CPU pose estimation may
be slow on low-power phones; the browser only processes
the decoded frames it receives. Some codecs and files
require transcoding to H.264 MP4 or browser-supported
WebM. No seeking controls are provided because the
pose detector has temporal state; explicit replay
reinitializes it safely. Current W4 counts are
descriptive and not an accuracy/F1 measurement without
independently labeled live events.
