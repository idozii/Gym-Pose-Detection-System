# M34 - Experimental local movement cycle counter (Squat / OHP / Row)

## What was implemented

M33 introduced local 2D movement angles. M34 builds on those
observations to display an *experimental count of movement cycles*,
not a claim of correct technique, safe loading or validated
exercise repetitions.

Supported camera-plane cycle signals:

| Exercise | Geometry used to detect cycle | Start/return | Bend endpoint |
| --- | --- | --- | --- |
| Squat | Hip-knee-ankle angle of most-visible leg | At least 155 degrees | At most 120 degrees |
| Overhead Press | Shoulder-elbow-wrist angle of most-visible arm | At least 155 degrees | At most 115 degrees |
| Barbell Row | Shoulder-elbow-wrist angle of most-visible arm | At least 155 degrees | At most 115 degrees |

**Important Row difference:** M33's torso inclination is helpful
for showing changes in torso posture, but a Row can be performed
with a relatively stable torso angle. M34 therefore counts an
approximate **elbow flexion/extension cycle**, not torso tilt.
This signal still does NOT verify a barbell rep, load path,
lumbar safety or exercise correctness.

The high/low numbers are **heuristic event detection thresholds**,
not biomechanical quality standards, goals for range of motion,
diagnostic cutoffs or coaching advice. A person who does not
reach either threshold will simply not receive a counted cycle,
even if they performed a legitimate repetition. Conversely,
camera perspective and poor tracking can create false counts.
These thresholds must be validated separately before any
product claim.

## Counting policy

To reduce false positives due to landmark jitter, M34 requires:

- at least **two consecutive sampled observations** at the
  high-angle starting endpoint;
- at least **two consecutive sampled observations** at the
  low-angle endpoint;
- two consecutive high observations to confirm return;
- one stable visible arm/leg side throughout the candidate
  cycle. Side switch discards pending cycle;
- fixed ≥0.5 landmark visibility for the participating joints,
  finite 2D geometry and nondegenerate joint segments;
- complete return within 0.9 to 12 seconds (candidate
  cycle time, not a recommended exercise tempo);
- pose observation at most every 400ms; a tracking gap
  beyond 1250ms discards an incomplete cycle;
- an angle excursion of at least 35 degrees for Squat,
  40 degrees for OHP/Row.

The number of already completed *observed cycles* is retained
when tracking briefly drops, but an incomplete cycle is rejected.
A camera stop, tab hide, physical camera switch, exercise
switch or explicit **Reset set** discards all cycle history.
The module keeps only at most 12 summary records of cycle
duration and angle excursion, not raw video or landmarks.
No browser persistence or network requests are used for
these observations.

The UI displays:

- Observed cycles since Start camera / Reset set;
- mean duration of recent observed cycles (seconds);
- mean angle excursion of recent observed cycles (degrees);
- clear instructions for camera position and a note that
  counts are not verified reps, form scores or injury advice.

Existing Push-up local geometry and six optional original
Fitness-AQA source-label models are **unchanged**. No new
model, API request or dataset training is involved.

## Verify in local Ubuntu environment

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
cd web
npm install --no-audit --no-fund
npm test
npm run build
npm run dev
```

Open **http://localhost:5173** with the browser camera.

1. Choose Squat, keep a side view of one hip/knee/ankle,
   click Start camera. Begin with the knee relatively
   straight, then perform a slow lowering and return.
   Confirm that only a complete observed cycle increments
   the counter. Standing still must not repeatedly count.
2. Switch to OHP and observe elbow flexion/extension cycles
   with shoulder/elbow/wrist visible. Ensure the previous
   Squat count was cleared.
3. Select Barbell Row. The *movement card* still shows
   torso inclination, while the *cycle card* uses elbow
   flexion and extension. A stable torso by itself
   should not cause repetitions to accumulate.
4. Cover the tracked joints or step out of camera view.
   Counting should stop and pending cycle should expire.
   Restore visibility and check that completed cycles do
   not spontaneously increase.
5. Click **Reset set**. Completed cycles and metrics must
   return to zero. Switching physical camera, stopping
   the camera or switching exercise resets the set.

Node unit tests cover full cycles, jitter/noise, incomplete
movements, missing pose, side changes, time regression,
session reset, limited in-memory history, privacy and
existing model-call isolation.

## Limitations and next experiment

**Do not claim validated gym reps**: camera projection
changes absolute angle. Fixed absolute thresholds have not
been calibrated across body sizes, camera views, tempo,
clothing, occlusion or loading. The current 400ms sample
policy can miss rapid movements and may be sensitive
to devices with low FPS. The Row count is especially
exploratory because elbow motion alone may not equal
a complete barbell rep.

Before any gamification, technique grade, injury-risk claim
or “correct/incorrect” recommendation, independently collect
human-annotated rep timestamps on consenting participants
and measure rep-level precision, recall, missed cycles,
false cycles and camera-specific abstention. Do not
use the original Gym7 M20 final TEST for UI threshold tuning.

**Deployment:** merging GitHub code and passing CI is not
proof of a live public site. Verify Vercel or GitHub Pages
deployment separately. If Vercel Hobby deployment limits
remain, test locally; do not pay just to enable this UI.
