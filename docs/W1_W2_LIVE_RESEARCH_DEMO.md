# W1 + W2: Browser camera readiness and live M34/M45 research comparison

## Scope and project boundaries

W1 checks and improves readiness of the EXISTING browser Vite camera
workbench. W2 introduces a **new, explicitly experimental** incremental
counter for Squat based on the original M45 relative-excursion rule.

The 17 files in the M48 research freeze, including original
web/src/rep_counter.js and original
web/scripts/m45_compare_train_relative.js, are NOT changed. In
particular, M34's original website counter retains its 400ms sample
gate and original fixed high-low-high angle thresholds.

W2 does not deploy the Python offline evaluator, conduct a TEST study,
train a model, compare technique quality, claim injury prevention or
reinterpret M47 TRAIN evidence as live-webcam accuracy.

## W1 engineering checks and recovery

The existing website already has:
- An HTTPS/localhost secure-context check and a user-clicked camera
  start; front/rear selection and visual-only mirroring; a canvas
  overlay from on-device MediaPipe Tasks Vision.
- A separate explicit consent toggle for remote research RF inference.
  Six Python models are not automatically packaged in the static
  frontend. Without a configured and trusted backend, scores are
  unavailable rather than simulated.
- Stop-camera actions on pagehide and active hidden-tab transitions,
  with video tracks stopped, pose landmarker closed, frame canvas
  cleared and in-flight research inference aborted.

W1 adds:
- The same stop/cancellation protection if a tab becomes hidden
  WHILE camera initialization is still in progress.
- An explicit stop/revoke if an active MediaStream video track ends
  unexpectedly, with a recovery prompt. Stale asynchronous track
  end events cannot stop the replacement camera.
- Actionable, privacy-preserving messages for permission denied,
  missing camera, camera in use, unsupported requested constraints,
  interruption and generic pose-engine startup error.
- A guard against new camera startup in a hidden browser tab.
- Unit/static regression tests for those behaviors and the original
  HTTPS opt-in, camera cancellation and inference-consent rules.

This is code- and CI-level readiness. **No actual user phone/desktop
camera or production Vercel endpoint was accessed by CI.** A real
human device test and current hosting checks are still required.
Do not report a successful deployment or working iPhone camera
solely because the Vite build succeeds.

## W2 incremental M45 design

The website still instantiates original RepCycleCounter (M34) with
default 400ms sampling, for all original Squat/OHP/Row contexts.

When Squat is selected, a separate
LiveRelativeExcursionCounter receives the **same incoming Tasks
Vision pose landmark arrays and browser frame timestamp** as M34,
using an independent minimum 200ms sample interval.

W2's frozen relative rule is a direct incremental transcription
of M45's original TRAIN comparator in
web/scripts/m45_compare_train_relative.js:
- same original camera-plane knee-angle measurement function;
- starting peak at least 155 degrees;
- decline at least 35 degrees from the high peak;
- recovery at least 35 degrees from the low point AND final angle
  at least 155 degrees;
- recovery after at least 400ms from observed low, before 12000ms;
- invalidate pending incomplete cycles after a >1250ms visibility
  gap or a left/right tracking-side switch;
- after a valid recovery, use that high endpoint as the next
  candidate peak.

W2 has one new live view selector: Compare, M34 only, M45 only.
It is display-only. Both counters always advance simultaneously
on the same camera session regardless of the selected view,
so switching does not secretly restart a model or alter counts.
A single Reset set clears both counters; stopping camera or
switching exercise also clears BOTH histories. Only Squat
displays W2. Push-up stays local preview; OHP and Row continue
using original M34 behavior with no unsupported M45 expansion.

The status readout describes M45's current observations. It does
not call any completed movement a verified or safe repetition,
and it never publishes a ground-truth accuracy percentage from
unlabeled webcam sessions. No source video or landmark history
is recorded, uploaded, persisted in browser storage or sent to
a remote backend by W2. Existing *separately consented*
RF inference remains the one optional source of landmark
transmission and is not enabled by the research comparison.

**Not equivalent to browser vs offline parity:** Python MediaPipe
Solutions (used in original M45 TRAIN evaluation) and browser
MediaPipe Tasks Vision Lite can output different landmarks. An
original nominal 200ms Python video pipeline and a browser
RAF sampler are not identical tracking contexts; on slow devices,
actual browser observations can be further apart than 200ms.
Counts from M34 and M45 are not directly an F1 comparison without
published matching GT events. M47's frozen TRAIN F1 results
compare M45 with M39 low-to-high, NOT deployed M34.

## Regression tests

The new W2 tests import the exact original **unchanged** M45
TRAIN comparator at test time and compare timestamped
sample-by-sample predicted events on synthetic Squat pose
traces with complete cycles, partial cycles, occlusion,
side switches, tracking gaps and timeouts. They also verify
M34 default 400ms behavior and that the W2 new module has
no network, storage or video-recorder APIs.

The W1 tests verify camera startup error recovery messages,
hidden-tab revocation during active and pending camera start,
ended-track cancellation, secure-context permission gate,
inference cancellation, old/new stream isolation and reset.

Run locally:
~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
cd web
npm install
npm test
npm run build
npm run dev
~~~

Open the Vite local URL over localhost, e.g. localhost:5173.
Camera access in a public production browser requires HTTPS.
To run the browser-only local demo, do not enable research
inference; it needs no external private Python model host.

## Manual W1 desktop/mobile acceptance matrix

These steps require the OWNER and actual device/browser.
Passing CI alone does not pass them:

| Platform/action | Expected behavior |
|---|---|
| Desktop Chrome/Firefox on localhost | Camera starts on user click, skeleton and pose FPS visible |
| Android Chrome HTTPS | Front/rear fallback and mirrored view are distinguishable |
| iPhone Safari HTTPS | Camera starts only with permission; switching front/rear releases the old stream |
| Permission denied | Clear recovery hint; no stuck loading button or video stream |
| Browser tab hidden during startup | Initialization is cancelled and any temporary track is stopped |
| Browser tab hidden while live | Video/media track stopped, counts/pose state cleared |
| Camera unplugged or another app takes camera | Active track termination triggers a clear restart prompt |
| Squat Compare mode | Both M34 400ms and M45 200ms counters update from the SAME camera pose |
| Squat view switch | Only visible cards change; counters keep their session history |
| Reset set | Both M34 and M45 counts return to zero |
| Change exercise / Push-up | Both counts clear; M45 panel hidden outside Squat |
| No external RF opt-in | No 33-landmark API request; source RF scores unavailable |
| Opt-in API unavailable | ABSTAIN/unavailable rather than fake model probabilities |
| Small viewport / keyboard | Cards fit screen and selector is keyboard-focusable |
| Browser model download unavailable | Camera error is explained and can be retried |
| Stop camera and restart | No old counts, stale video or open camera streams |

Manual results should record browser/version, device, public site URL
(if deployed), local vs public, and outcome. Do not store images,
private pose traces or personal identities to satisfy W1.

## Deploy and hosting caveats

The original README notes earlier Vercel Hobby HTTP 402 quota issues
and a separate GitHub Pages static preview. Neither the presence
of a GitHub Actions green build nor merged web code establishes
that the public Vercel production site actually serves W1/W2.
Confirm current hosting deployment and frontend route manually.
Vite site root is web/, build command npm run build, dist/.
No private Python RF bundles are published with this UI.

This milestone creates no model-training, TEST evaluation,
dataset archive access, backend hosting change or live privacy
guarantee beyond the reviewed browser code. User consent,
camera/device compatibility, reliability and separate Python
model-provisioning still require real-world checks.
