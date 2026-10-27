# M33 - Local explainable exercise feedback (Squat / OHP / Barbell Row)

## Goal and user experience

The previous 3-exercise UI exposed landmark visibility,
6 original Fitness-AQA experimental source-label RF scores
and a significant number of `ABSTAIN` cases. It did not
explain measurable movement in practical terms. M33 adds
a **local, camera-plane movement observation card** below
the pose inspector for the three original exercises. It does
not generate fabricated “good/bad” labels or attach coaching
advice to the RF scores.

| Exercise | Geometry measured at ≥0.5 landmark visibility | Display |
| --- | --- | --- |
| Squat | The most visible side's hip-knee-ankle angle | Current 2D knee flexion angle; 10-second observed range |
| OHP | The most visible side's shoulder-elbow-wrist angle | Current 2D elbow angle; 10-second observed range |
| Barbell Row | Mid-shoulder to mid-hip segment's inclination from image vertical | Current 2D apparent torso inclination; 10-second observed range |

All geometry is calculated in the browser, **not the backend**.
The webcam overlay still uses the existing MediaPipe Tasks
Vision model. No additional training, inference network call
or video upload is needed for M33; viewing local movement data
does **not** require research inference consent.

Only angle values and timestamps are stored ephemerally
in a 10-second in-memory sliding window, sampled at most
once per 500ms. Fewer than 4 valid samples display an
angle but **no fabricated movement range**. If tracking
has been lost for >1.25 seconds, both current angle and
range become unavailable, and the card requests better
framing. The window resets on exercise changes,
camera switch/stop and tab visibility stop; if the
most-visible side changes, the history resets to avoid
mixing different limbs.

Every card states the camera positioning suggestion and
measurement limitation, without asserting a universal
“safe” range or an automatic proper/improper movement
classification. Torso inclination is only the 2D projected
angle of shoulder-to-hip orientation relative to
the *camera image's vertical*, which can change with
camera orientation and cannot measure lumbar spine
curvature. OHP elbow angle does not measure actual barbell
path. Squat knee angle alone cannot determine correct depth,
knee travel safety or whole-exercise quality.

## Source model scores remain separate

The six previously hosted original Fitness-AQA source-label
research models remain unchanged:
- Squat: `error_knees_forward`, `error_knees_inward`.
- OHP: `error_elbows`, `error_knees`.
- Row: `torso_angle`, `lumbar`.

The optional API is still separate, still requires a
user opt-in and 6/6 readiness, and its outputs are
**publisher source annotation associations only**.
Do not map `source_label_flag=false` to “good form”.
The original training source was Python MediaPipe
Solutions, while the browser uses Tasks Vision;
landmark distribution and source-feature parity are
not validated. Squat source error precision/FPR,
OHP source low AP and Row lumbar label geometry
limitations prevent credible technique safety claims.

Push-up remains a separate local geometry-only
preview from M30 and is not modified by M33.

## Local Ubuntu test

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only

cd web
npm install --no-audit --no-fund
npm test
npm run build
npm run dev
```

Then open `http://localhost:5173` and start the camera.
For Squat, try viewing your hips/knees/ankles from
the side; for OHP, show shoulder/elbow/wrist; for
Barbell Row, show both shoulders and hips. The
local card should show `COLLECTING SAMPLES` until
four geometrically valid samples have been observed,
then `MOVEMENT OBSERVED`, together with angle and
observed range. If a joint is not visible or the
camera is stopped, no stale measurement should appear.

M33 is **not** a rep counter, autonomous personal trainer
or validated technique-diagnosis system. A later
milestone would need independently consented human-
reviewed videos, explicit target definitions and
browser-Python feature parity tests before we can
apply validated model-based recommendations.

## Validation and deployment conditions

Synthetic unit tests cover 2D angular geometry,
min landmark visibility, side changes, time windows,
abstention, exercise/camera reset and zero networking
in the local feedback module. GitHub CI performs
web tests/build and Python regression tests.

The current published Vercel production domain may
still be behind `main` because Vercel Hobby already
enforced a 24-hour deployment quota in M30. A merged
PR or successful GitHub CI does not prove Vercel
production deployed. GitHub Pages also requires
the repo owner to enable Settings > Pages > Source:
GitHub Actions before public URLs are confirmed.

No additional paid deployment, model publishing,
private datasets or TEST data are involved.
