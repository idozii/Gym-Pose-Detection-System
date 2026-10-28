# W3 - Local Rep Counter Session Dashboard

## User-facing W3 capabilities

W3 builds on W1 camera lifecycle and W2 Squat M34/M45
live experimental comparison, adding a responsive **local
session dashboard** without modifying the original frozen
M34 counter or M45 research comparator.

When the owner starts a supported Squat, Overhead Press
or Barbell Row camera session, the dashboard shows:

- Session elapsed clock, mm:ss or hh:mm:ss for longer
  sessions, beginning only AFTER the camera/model starts.
  It updates from the browser RAF clock even if new
  decoded video frames temporarily stop arriving.
- Pose detected proportion among analyzed video frames;
  this is an OBSERVATION RATE, never AI model confidence.
- Required joint visibility proportion among analyzed
  video frames using the existing pose.js readiness gate;
  this is NOT technique quality, per-rep correctness,
  statistical certainty, or model calibration.
- Number of analyzed pose frames since the current set.
- Status: CAMERA OFFLINE / WAITING FOR VIDEO /
  JOINTS VISIBLE / LIMITED JOINT VIEW /
  NO POSE VISIBLE / VIDEO STALLED. The latter indicates
  no newly processed video frame for >1500ms.
- Up to 12 recent descriptive cycle-time bars.
  M34 records the ORIGINAL high-low-high full-cycle
  durations in RepCycleCounter.recent. For Squat only,
  M45 adds separately labeled **low-to-return**
  durations already retained by the live M45 module.

Important: M34 full-cycle and M45 low-to-return
durations have different definitions and independent
vertical scale maxima. Their charts cannot establish
which algorithm is more accurate. Without manually
annotated live GT, predicted cycle count is not a
verified repetition count.

Push-up remains an unscored local geometry preview,
with W3 cycle dashboard hidden. M45 is NOT enabled
for Row or OHP; those exercises retain M34 only.

## Reset, lifecycle and privacy

W3 listens to existing on-device pose observations;
it does not request camera permission, create
an additional video stream, send data or modify
opt-in remote RF inference. It only keeps integer
frame counters, two boolean latest visibility
flags, a monotonic timer and a last-frame timestamp.
Cycle durations originate from existing bounded
W2/M34 volatile arrays, max 12 records each.

Clicking Reset set resets both W2 counters AND W3
elapsed clock/visibility counts. Switching exercise
starts a fresh session if the camera is running.
Stopping camera, changing camera, hiding the tab
or pagehide resets W3 along with existing M34/M45
state and closes the camera. Selecting Compare/M34/M45
changes only display and does not reset the timer.

The W3 state module contains no video frame,
raw landmark, person ID, browser storage, API
fetch or file upload behavior. Existing separately
enabled RF inference may send 33 landmark points
to the configured trusted Python API by explicit
consent; W3 itself does not enable this.

## Code changes

New:
- web/src/session_dashboard.js
- web/tests/session_dashboard.test.js

Updated:
- web/src/main.js (new UI and on-frame telemetry)
- web/src/style.css (responsive 2-4 KPI layout
  plus keyboard- and screen-reader-accessible
  tempo visualization)
- README.md / docs/W3_SESSION_DASHBOARD.md

NOT changed: web/src/rep_counter.js, original
web/scripts/m45_compare_train_relative.js,
web/src/live_relative_excursion.js, or any
of the 17 M48-frozen research source paths.
W3 does not touch or read RepCount TAR, VALID/TEST
annotations or reproduce M47 F1.

## Local and manual validation

Run:
~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
cd web
npm install
npm test
npm run build
npm run dev
~~~

Open localhost:5173 and perform this manual
camera acceptance checklist:

1. Select Squat, press Start camera, inspect
   the W3 session clock, pose coverage,
   required-joint visibility, analyzed frames
   and status. They start empty then update.
2. Move out of frame; status should become
   NO POSE VISIBLE or LIMITED JOINT VIEW.
   Cover required joints while keeping other
   joints visible to test the coverage rates.
3. Perform a few experimental Squat cycles.
   If either detector observes a complete cycle,
   its own bar chart should gain a bar.
   M34 charts FULL cycles, M45 LOW-TO-RETURN only.
   No prescribed fitness tempo or quality verdict.
4. Switch Compare, M34 and M45 views. W3
   session timer/counters should NOT reset.
5. Click Reset set while camera is running.
   Both research counters and W3 clock/sample
   stats return to zero and rep chart clears.
6. Change to OHP or Row. M45 research chart
   should disappear; M34 chart remains.
7. Change to Push-up. W3 cycle dashboard
   disappears (Push-up preview is not a
   validated rep detector).
8. Change to rear camera, deny camera permission,
   hide the tab, stop and restart the camera.
   Each new session must begin with empty W3
   state and no stale stream/pose data.
9. Test a narrow Android/iOS viewport and
   tab through all controls using a keyboard.

Automatic CI checks timing/coverage logic,
privacy no-storage and static UI integration,
but cannot establish that a particular device
camera or production deployment actually works.

## Research and product limitations

- "Pose seen" and "Required joints visible"
  are frame proportions, NOT model confidence
  probabilities or correctness of repetitions.
- Counters use camera-plane 2D angle
  heuristics affected by side, occlusion,
  tracking gaps and variable frame sampling.
- W3 does not calibrate browser MediaPipe
  Tasks Vision against Python MediaPipe
  Solutions used by the frozen M45 TRAIN study.
- M45 was developed using publisher TRAIN;
  M47 TRAIN F1 improvements cannot be copied
  onto real users or used as live accuracy.
- This dashboard is a research/portfolio
  demonstration, NOT exercise-form scoring,
  injury prevention or validated coaching.
- A green Vite build or GitHub CI does not
  prove the Vercel production deployment is
  live. A real-device browser test is needed.
