# Gym Pose Detection (Real-Time Correct vs Incorrect Form)

## Milestone 13A: Overhead Press pose extraction pilot

OHP now has a **separate versioned 111-feature schema** (103 compatible
MediaPipe pose columns plus 8 OHP-specific upper-body geometry features),
with critical visibility checks for elbows and wrists.

```bash
# First 8 source train clips + 8 source validation clips, source TEST LOCKED:
python -m scripts.extract_fitness_aqa_ohp --split train-val --max-videos-per-split 8

# Once source video/prediction coverage has been inspected:
python -m scripts.extract_fitness_aqa_ohp --split train-val --max-videos-per-split 0
```

Outputs stay under `data/features/fitness_aqa_ohp/` (ignored by Git).
No OHP model performance claims until full train/val extraction and evaluation.
See [M13A OHP guide](docs/FITNESS_AQA_OHP_M13.md).

---

## Milestone 16D: scored-only RF runtime profiling

The owner's M16C webcam test admitted **47/180 Squat frames** and
generated **94 source-label scores**, up from zero in the initial
run. Both ankles are now occasionally visible, but the right ankle
remains below the source pose-quality cutoff in **131/180** frames.
The old average model-stage timing included many abstained frames,
so it was not a measure of real RF execution speed.

M16D now measures **separate** task-level successful inference
and abstention latencies, plus scored/abstained source-label
fractions per exercise/target, without collecting frames,
person data, source-label probabilities or changing model gates:

```bash
python -m pytest -q
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

Inspect `diagnostics.timing_ms.successful_model_task_inference`,
`diagnostics.timing_ms.abstained_model_task_call` and
`diagnostics.source_label_score_admission_by_target`.
Model score admission is **not** model accuracy or form-safety
assessment. This brings the local research viewer closer to M17
release/reproducibility work, not a production fitness coach.

[Milestone 16D scored inference performance audit](docs/M16D_SCORED_INFERENCE_LATENCY.md).

---

## Milestone 16C: webcam framing guide for low-visibility joints

On the owner's first 120-frame Squat webcam diagnostic, MediaPipe detected
a pose in **120/120 frames**, but both ankles and the right knee were
below visibility>=0.50 in **every** frame. The original source-model
quality gate correctly abstained on all 240 Squat detector attempts.
This is a **framing/joint visibility** blocker, not a model load failure.

The viewer now has a separate, display-only **joint visibility side
panel**. It shows which joints are `LOW`, `VISIBLE`, or `NO POSE`
so a user can move back and include their whole body without changing
the original 6 model schemas, gates or thresholds:

```bash
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

Use `--no-framing-guide` to hide the panel. No webcam frames are
recorded or uploaded. MediaPipe visibility is **not** a medical
assessment or confirmation of safe/correct exercise.

[Milestone 16C camera framing guide](docs/M16C_CAMERA_FRAMING_GUIDE.md).

---

## Milestone 16B: live pose abstention and performance diagnostics

The owner's first 60-frame Squat webcam run initialized correctly and
loaded all six source-error RF models but returned **0 source-label
scores / 120 abstentions** (two task scores per camera frame).
That alone did not distinguish *no MediaPipe pose* from unreliable
required joints or an inference error.

M16B reports **aggregate, privacy-safe** reasons and mean/p95
pose/model/frame timing + observed FPS; it keeps all quality gates and
thresholds unchanged:

```bash
python -m pytest -q
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 120 \
  --trust-local-artifacts
```

The viewer now displays no-pose and missing-joint hints and prints a
`diagnostics` summary at exit. No webcam video, pose coordinates,
per-frame predictions or publisher TEST data are stored. This is
diagnostics only, not a technique or safety coach.

[Milestone 16B diagnostics guide](docs/M16B_LIVE_POSE_DIAGNOSTICS.md).

---

## Milestone 16A: Local real-time *research* viewer (experimental)

The owner's M15C local run confirmed six source-label error-model
artifacts are loaded correctly (Squat, Overhead Press, Barbell Row).
However, source-validation precision/recall and pose acceptance remain
too weak to claim reliable technique or injury-prevention feedback.

An opt-in OpenCV webcam **research viewer** displays the manually
selected exercise's pose and two original-source *annotation scores*.
It never turns scores into coaching advice, safe/correct labels,
automatic movement alerts, rep counting, uploads or recordings.

```bash
python -m pytest -q
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
python -m scripts.run_fitness_aqa_research_viewer --exercise squat --camera 0 --trust-local-artifacts
```

Use keys `1` Squat, `2` OHP, `3` Barbell Row, `q` to exit.
Models must be your **own trusted local joblib** files (pickle can run
code). Missing or low-quality poses produce visible **ABSTAIN**, not
'correct form'. OpenCV GUI and MediaPipe camera extras are required.

[Milestone 16A real-time research viewer](docs/M16A_REALTIME_RESEARCH_VIEWER.md).

---

## Milestone 15C: Squat original source-error RF registry

The two genuine source-error targets `error_knees_forward` and
`error_knees_inward` are separate from the notebook's binary
`correct/incorrect` model. M15C packages new **103-feature Squat
source-label RFs** using source TRAIN video-group calibration and
publisher VAL reporting, without touching publisher TEST:

```bash
python -m scripts.train_fitness_aqa_squat_source_errors --seed 42 --trees 300
# Only load your OWN trusted local artifacts:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

Original source TRAIN/VAL pose CSVs are reused; no ZIP/media extraction.
Both Squat source-error slots can become available alongside OHP/Row,
with an explicit experimental-only result and fail-closed abstention.
Six software model slots do **not** mean the exercise coach is ready.

[M15C Squat source-error integration guide](docs/M15C_SQUAT_SOURCE_ERROR_REGISTRY.md).

---

## Milestone 15B: OHP registry feature adapter restored

M15A could already load both local Barbell Row models, but initially
reported `feature_schema_not_integrated` for OHP because its original
30-feature source module lived only on unmerged PR #18. M15B now
integrates those **exact** original OHP elbow/knee schemas on the
current main-based branch (no re-training or ZIP extraction required):

```bash
python -m pytest -q
python -m scripts.check_fitness_aqa_model_registry
# Only for your own trusted locally trained joblib files:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

After trusting the original locally created models, the two OHP and two
Barbell Row tasks can report `loaded_experimental_only`. Squat original knee-error model contracts are supplied by M15C after the user's local TRAIN-only run. This is
experimental source-label inference, **not** a trustworthy realtime
exercise-safety coach.

[Milestone 15B OHP adapter details](docs/M15B_OHP_REGISTRY_INTEGRATION.md).

---

## Milestone 15A: Offline multi-exercise model registry

An experimental **fail-closed** registry can load the owner's locally
trained OHP static and Barbell Row torso4 static RF artifacts, verify
original source target/feature schema/model class/operating threshold,
and score source labels without access to publisher TEST media:

```bash
python -m scripts.check_fitness_aqa_model_registry
# Only for your OWN trusted local joblib artifacts (pickle deserialization):
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

The Squat source knee-error slots are intentionally not automatically
mapped to the legacy generic correct/incorrect classifier. Original OHP
30-feature source schemas are integrated by M15B; the outstanding research
PR #18/#19 histories still require separate reconciliation before final release. Missing model/pose yields
`abstain`, not a "good exercise" classification. This is not a webcam
coach or medical safety assessment.

[Milestone 15A registry contracts](docs/M15_UNIFIED_MODEL_REGISTRY.md).

---

## Milestone 14D: Barbell Row static RF benchmark

Barbell Row full source train/validation now retains around 70% of
original per-task labeled images using the new torso4 20-feature
shoulder/hip geometry. The new benchmark trains a **separate** Static RF
for each publisher error label, selecting threshold from original
**TRAIN candidate-prefix groups only**:

```bash
python -m scripts.benchmark_fitness_aqa_barbell_row_torso4 \
  --seed 42 --trees 300 --bootstraps 500
```

It reads existing local torso4 CSVs (no ZIP/MediaPipe), rejects pilot,
split/prefix leakage and mismatched manifests, reports acceptance and
source-validation AP/precision/recall/FPR, and stores RF models under
ignored `artifacts/fitness_aqa_barbell_row_torso4_rf/`. Source TEST
image contents are never used. Candidate prefix is **NOT verified
video/subject provenance**, and 4-joint geometry does not measure spine
curvature.

[Milestone 14D training/evaluation protocol](docs/FITNESS_AQA_BARBELL_ROW_RF_M14D.md).

---

## Milestone 14C: Barbell Row shoulder–hip 20-feature extraction

The source TRAIN/VAL PR #22 audit established that requiring knees
and ankles was a major cause of the earlier **4–5%** Barbell Row pose
coverage. On equal-size label-stratified audit subsets, a strict
shoulder+hip four-joint gate admitted approximately **68–75%** of
images per task/split. These sample rates are **not** model accuracy.

A new independent `barbell-row-torso4-image-20-v1` schema uses only
the four reliable shoulders/hips as both feature inputs and quality
gate (>=0.50):

```bash
# Source JSON + ZIP already downloaded in M12; small smoke test first
python -m scripts.extract_fitness_aqa_barbell_row_torso4 --max-images-per-task-split 12
# Full original TRAIN+VAL, never publisher TEST
python -m scripts.extract_fitness_aqa_barbell_row_torso4 --max-images-per-task-split 0
```

The new output directory is `data/features/fitness_aqa_barbell_row_torso4/`
and the original 32-feature pipeline remains intact. This is an
**image-label classifier foundation**, not direct lumbar-spine
measurement or reliable real-time coaching.

[Milestone 14C torso4 extraction methodology](docs/FITNESS_AQA_BARBELL_ROW_TORSO4_M14.md).

---

## Milestone 14B prerequisite: Barbell Row Pose Coverage Audit

The original source Barbell Row TRAIN/VAL image extraction succeeded,
but strict 8-joint MediaPipe pose coverage was only **4–5%** across
the two tasks. The source lum-bar validation subset retained only
**8 positive error images**. Training a classifier now would risk
substantial source selection bias.

```bash
python -m scripts.audit_fitness_aqa_barbell_row_coverage --images-per-task-split 180 --seed 42
```

This is a train/validation-only, source-label-stratified diagnostic
comparing 8-joint strict/relaxed gates against hypothetical 6- and
4-joint gates on the **same source photos**. It reports per-joint
failures, positive/negative pose admission, and missing-pose counts;
no test photos decoded, no model retrained, no source archive edits.

[Barbell Row pose quality diagnostic](docs/FITNESS_AQA_BARBELL_ROW_POSE_COVERAGE_M14.md).

---

## Milestone 14A: Barbell Row image pose extraction

The publisher Barbell Row dataset is **images** with two separately
split binary error targets, `torso_angle` and `lumbar` (not video
time intervals). A dedicated 32-feature, quality-gated image pose
extractor processes original TRAIN/VAL images only:

```bash
# Source JSONs / images already verified in milestone 12.
python -m scripts.extract_fitness_aqa_barbell_row --max-images-per-task-split 12

# After pilot success, complete publisher TRAIN+VAL images for each task:
python -m scripts.extract_fitness_aqa_barbell_row --max-images-per-task-split 0
```

Outputs are local and ignored under `data/features/fitness_aqa_barbell_row/`.
**Publisher TEST pixels are never decoded.** The image-ID prefix is an
unverified clip identifier, and 8-joint MediaPipe geometry cannot directly
measure lumbar-spine curvature.

[Milestone 14 Barbell Row extraction guide](docs/FITNESS_AQA_BARBELL_ROW_M14.md).

---

## Milestone 12: Fitness-AQA multi-exercise dataset foundation

The research dataset foundation now supports the original publisher's **three
exercise subsets** without assuming interchangeable labels or globally
correct form:

- **Squat:** named knee error intervals from video ZIP (existing baseline).
- **Overhead Press (OHP):** elbow/knee error intervals from its own video ZIP.
- **Barbell Row:** torso-angle and lumbar **image** labels (0/1), with **separate
  publisher train/val/test splits per error target**.

```bash
# Validate downloaded small publisher JSON metadata, even before ZIP download:
python -m scripts.inspect_fitness_aqa_multi --exercise ohp --skip-archive
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row --skip-archive

# When archives are downloaded, verify exact file-stem IDs WITHOUT unzipping:
python -m scripts.inspect_fitness_aqa_multi --exercise ohp
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row
# or --exercise all after downloading all three datasets
```

The manifest saves locally under `artifacts/fitness_aqa_dataset/`, never
uploads source media, detects missing/duplicate ZIP keys and warns if the
Barbell Row image's two target-specific splits conflict. **No new OHP/Row
accuracy claims or model training** occur in this milestone; the publisher
test IDs are checked for integrity only. Source splits are not verified
independent by athlete.

[Milestone 12 datasets, exact download commands and research protocol](docs/FITNESS_AQA_MULTIBENCHMARK_M12.md).

---

## Milestone 2: Data provenance and subject-independent evaluation

The new commands validate a consented image manifest, extract traceable features, and
benchmark a Random Forest on **held-out subjects** with metrics per camera view.
The code ships without a real dataset or fabricated model accuracy.

```bash
gym-pose manifest-template
# Fill data/manifest.csv using real subject/session/view IDs and real image paths.
gym-pose validate-manifest
gym-pose extract-manifest
gym-pose benchmark
```

See [Milestone 2 protocol, outputs and data policy](docs/MILESTONE_2.md).
A random-frame validation split is **not** evidence of generalization across subjects.

---

## Milestone 1: Modular baseline (new)

The original notebook below remains available as a historical baseline. A new
installable package adds named 103-feature extraction, visibility checks, resettable
probability smoothing, model schema validation, tests and CLI workflows.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera,dev]'
python -m pytest -q

# Capture images for each class, then train an explicitly versioned baseline:
gym-pose capture --label correct --count 100
gym-pose capture --label incorrect --count 100
gym-pose extract
gym-pose train
gym-pose webcam
```

**Evaluation warning:** `gym-pose train` uses a random frame split for an exploratory
baseline only. For independent group evaluation, collect truthful `subject_id`/
`session_id` metadata and use `--group-column subject_id`. No real-world
form accuracy or rep-counting claims are made. Existing old joblib bundles are not
schema-versioned and must be retrained for the new CLI.

See [Milestone 1 setup, scope and limitations](docs/MILESTONE_1.md).

---

This project is structured for building and running a real-time gym form detection pipeline using:

- MediaPipe Pose for landmark extraction
- Scikit-learn for form classification
- OpenCV for webcam capture and live overlay
- YOLO weights available under `models/` for related detection experiments

The current notebook workflow supports one exercise at a time (default: squat), with two labels:

- `correct`
- `incorrect`

## Project Structure

```text
.
├── data/
│   ├── raw/                       # Captured training images by exercise/label
│   └── features/                  # Generated CSV feature files
├── models/                        # Trained classifiers and YOLO weights
├── notebooks/
│   └── gym_pose_detection.ipynb   # Main end-to-end notebook
├── README.md
└── requirements.txt
```

## Setup

Use your existing conda env or create a new one.

### Option A: Conda (recommended)

```bash
conda create -n gym_pose python=3.10 -y
conda activate gym_pose
pip install -r requirements.txt
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Workflow

Open:

- `notebooks/gym_pose_detection.ipynb`

Run cells in order:

1. Imports and path setup
2. Feature utility functions
3. Dataset capture
4. Feature CSV generation
5. Classifier training
6. Real-time webcam inference

### Data Collection

Capture images per label from webcam:

```python
collect_samples("correct", target_count=250)
collect_samples("incorrect", target_count=250)
```

Captured images are saved under:

- `data/raw/squat/correct`
- `data/raw/squat/incorrect`

### Build Features and Train

```python
build_feature_csv()
train_form_classifier()
```

Generated artifacts:

- Feature CSV: `data/features/squat_features.csv`
- Trained model: `models/squat_form_rf.joblib`

### Run Real-Time Form Detection

```python
run_realtime_form_check()
```

Controls:

- Press `q` to stop webcam inference.

## Labeling Guidance

To improve model quality:

- Keep camera angle consistent during one exercise session.
- Record different body positions, clothing, and lighting.
- Ensure each class has similar sample counts.
- Start with one exercise before adding more.

## Troubleshooting

- If webcam does not open, check camera permissions and `camera_index`.
- If MediaPipe import fails, verify package in active environment:

```bash
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
```

- If notebook paths fail, make sure you run from the repository root and use the updated notebook in `notebooks/`.