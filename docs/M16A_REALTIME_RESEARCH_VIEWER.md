# Milestone 16A - local real-time research viewer (NOT a fitness coach)

## Prerequisite: the owner's M15C model/registry result

The owner ran M15C on actual original publisher Squat TRAIN+VAL pose CSVs:

| Named Squat error | Source validation AP | Prevalence/AP reference | Train-only threshold | Macro-F1 | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Knees forward | .3769 | .2175 | .30 | .6212 | .3676 | .7850 | **.3753** |
| Knees inward | .1600 | .0407 | .20 | .5910 | .1961 | **.2500** | .0434 |

Pose retention was **50.2% TRAIN** and **52.1% VAL** on the source
Squat sampled frames. Squat Knees Forward has a very high false-positive
rate; Squat Knees Inward misses most error-positive frames. Neither
model is ready for reliable coaching alerts.

The owner confirmed **174 passed tests, four preexisting sklearn
single-label warnings**, and a six-slot source-error registry
successfully loading:

- Squat: `error_knees_forward`, `error_knees_inward`.
- OHP: `error_elbows`, `error_knees`.
- Barbell Row: `torso_angle`, `lumbar`.

All bundles are **experimental_source_label_score** models fitted on
original source TRAIN; publisher TEST remains locked and unexamined.
Model compatibility and source-label AP are **not** safety certification.

## Scope: local viewer only

`scripts/run_fitness_aqa_research_viewer.py` reuses the already
integrated `gym_pose.model_registry` for ALL feature schemas and
per-task pose quality gates. One MediaPipe pose pass per webcam frame
is shared across two *manually selected* error tasks, with the original
trained static RF input schemas:

- Squat `squat-103-v1` (103 features).
- OHP two distinct 30-feature task-specific source schemas.
- Barbell Row `barbell-row-torso4-image-20-v1` (20 features).

OpenCV draws only a pose skeleton, **research source-label probability
scores** and train-selected annotation score thresholds. Press 1/2/3
to switch manually between exercises or Q/Escape to exit.

A missing model, missing unreliable joint, missing pose or model
inference exception yields an explicit `ABSTAIN` text on that target,
never a `correct` classification, previous-frame score, green/red
error alert or movement direction.

**Camera and privacy:** frames are processed locally and not saved or
uploaded. No recording, pictures, logging of individual predictions,
microphone use, exercise rep counting, coach dialog, audible alerts,
remote model downloads, or source TEST access. Only aggregate scored/
abstained frame counts print upon closing the viewer.

**Safety / reliability:** The overlays reflect associations with the
original publisher's named error annotations, NOT medical/injury advice
or verified proper exercise. In particular, the four-joint Barbell Row
pose cannot observe lumbar spine curvature. The real-time pipeline
uses MediaPipe video tracking whereas the source models were trained
using static-image detection; viewpoint and camera differences create
an **unmeasured deployment/domain shift**. Threshold crossings are not
validated safety warnings. This is an experimental software demo,
not a released fitness coach.

## Run (on your existing desktop with a webcam and local display)

```bash
git fetch origin
git switch feat/m16a-live-research-viewer 2>/dev/null || \
  git switch --track origin/feat/m16a-live-research-viewer
git pull --ff-only
python -m pytest -q

# Optional: inspect six local models without webcam:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts

# Viewer: 1=Squat, 2=OHP, 3=Barbell Row, Q=quit:
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --trust-local-artifacts
```

Short local display smoke test (also tests cleanup):

```bash
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise ohp --camera 0 --max-frames 60 \
  --trust-local-artifacts
```

The `--trust-local-artifacts` flag explicitly acknowledges that
`joblib.load` executes Python pickle code. **Use only models that
you trained and trust**. In particular never load a model downloaded
from an unknown source. The source models are in ignored local folders
under `artifacts/`, not bundled with the repository.

If your environment lacks a working camera backend, install the
documented optional camera dependencies in a compatible Python
environment (`python -m pip install -e '.[camera,dev]'`); webcam
permission/OS display support is required. Headless environments
can run tests and the registry preflight but **cannot** display
OpenCV's window. This PR's mocked tests do not rely on a camera.

## What M16A does *not* do

- No exercise auto-classifier (you manually select the exercise).
- No rep counting, session analytics, movement feedback or suggested fixes.
- No temporal error RF for OHP/Squat; models are **static** frame-only.
- No calibration of probabilities for the new person's camera.
- No assessment of subject-independent accuracy, live latency or FPS
  yet. These belong to later controlled work after stability checks.
- No interpretation of publisher label `0` as globally correct form.

M16B should measure runtime processing rate and per-task abstention
on locally consented recordings and improve rendering/accessibility,
**without treating high-recall/low-precision source errors as
authoritative warnings**. M17 can consolidate documentation, CI,
dependency reconciliation, frozen evaluation protocol and demo.

This viewer does not edit the owner's notebook, saved RF models,
publisher media/CSV or previous research output.
