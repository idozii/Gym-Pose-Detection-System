# Gym Pose Detection - Experimental Fitness-AQA Research Viewer

**Research-only software prototype, not a validated fitness coach or
medical/injury-prevention detector.**

The current code supports a local MediaPipe/OpenCV webcam research viewer
and **six separately versioned source-annotation classifiers** across
Squat, Overhead Press and Barbell Row. It presents annotation scores
or explicit `ABSTAIN`; it never interprets missing labels as proper
technique, and does not issue autonomous exercise safety alerts.

The owner demonstrated **201 passing tests** and a **180-frame Ubuntu
Squat webcam run with 360 successful source-label calls**, observed
**15.78 displayed FPS** and **11.16ms mean successful per-task
feature+RF inference**. This is an engineering runtime demonstration,
not a test of correctness or robust generalization. Evaluation is
exploratory on repeatedly consulted publisher validation samples.

### M30 - Bicep Curl temporal TRAIN research + Push-up local web preview

**Push-up is now selectable on the website** as an on-device,
camera-only research preview showing live 2D elbow angle and
10-second elbow angle range. It **does not** run a Push-up
good/bad model, count reps, send landmark data to the research
API, or claim safe/correct form. The six original Squat,
OHP, Row source-label detector endpoints are unchanged.

[M30 Curl and Push-up technical/Ubuntu guide](docs/M30_BICEP_CURL_AND_PUSHUP_WEB_PREVIEW.md)
also includes a new local-only Bicep Curl temporal feature
experiment using the already downloaded Gym7 archive and
**only frozen M20 TRAIN people**, comparing 264, 268,
276 and 280 feature variants with grouped 5-fold and LOSO
cross-validation. Neither existing M20 VAL/TEST clips nor
private model bundles are accessed. New model metrics
will only be available after the owner runs the commands.

---

### M29 - Freeze independent Push-up video study before collecting new subjects

Real owner M28 TRAIN-only GroupKFold and LOSO both found directional
gains in motion-aware Push-up classifier balanced accuracy, but **every
paired 95% subject-bootstrap difference interval includes zero**.
The camera/pose-quality-only 68-feature model reversed ranking
across validation strategies and all LOSO candidates had the
same bad-label recall as the 64-feature baseline.

[M29 independent cohort preparation](docs/M29_INDEPENDENT_PUSHUP_COHORT_PROTOCOL.md)
predeclares the 64-feature reference, interpretable 76-feature motion
hypothesis and exploratory 80-feature model for future genuinely
new consented adult participants. The new
[`m29_external_pushup_cohort`](scripts/m29_external_pushup_cohort.py)
CLI creates a blank *private* manifest and validates video
deduplication, dual-independent-rater annotations, pseudonymous
consent references and source cohort balance. **No new real videos
have been collected or scored**; no original VAL/TEST video is
accessed and no production inference/model changed.

---

### M28 - TRAIN-only Push-up temporal robustness and feature-family ablation

Owner M27 real-data TRAIN-only GroupKFold on 83 clips and 14 subjects:
64-feature baseline BA **54.18%**, 80-feature temporal BA **58.91%**,
a **+4.73 percentage point** exploratory gain; F1 bad and AP bad
rose slightly, while **recall bad fell from 51.22% to 48.78%**.
No original M20 VAL/TEST subject was evaluated by M27.

[M28 temporal-family robustness](docs/M28_PUSHUP_TEMPORAL_ROBUSTNESS.md)
compares the original 64, motion-kinematics-only 76, pose-quality-proxy
68 and combined 80 feature sets on identical frozen TRAIN clips, both
GroupKFold(5) and Leave-One-Subject-Out, plus descriptive paired
subject-cluster bootstrap intervals. It uses the existing local CSVs,
no new media download, no TEST inspection and no public model release.
The previously consulted M20 VAL is NOT used for adaptive model selection.

---

### M27 - TRAIN-only Push-up motion trajectory hypothesis

Owner M26 original VAL observed: Full paired accepted **23/30**
Push-up clips and achieved **61.36%** BA on 23 paired clips;
Upper expanded accepted **30/30** but achieved **51.89%**
BA on the same clips. On seven upper-only recovered VAL
videos it missed **all three publisher bad labels**.
Full model abstains on those seven videos.
**No current Push-up model is fit for public form-quality advice.**

[M27 frozen TRAIN-only temporal feature study](docs/M27_PUSHUP_TEMPORAL_TRAIN_EXPERIMENT.md)
adds 16 motion trajectory descriptors to the original M23
64-feature upper-body clip schema, and compares identical video
and person-heldout `GroupKFold` subsets on the ORIGINAL M20
TRAIN subjects only. Original VAL/TEST video is untouched by M27.
This is post-VAL exploratory research, not a production model.

---

### M26 - Frozen original M20 validation comparison (no TEST)

Real owner M25 LOSO on original TRAIN: Full paired 54.96%
balanced accuracy, Upper paired 56.39%, Upper expanded 53.57%.
Paired subject-bootstrap interval estimates for both upper
variants' balanced accuracy differences cross zero, so no
form-label accuracy improvement is established. The upper-body
gate remains a **coverage** improvement, not a validated coaching
model.

[M26 predeclared Push-up VAL protocol](docs/M26_FROZEN_PUSHUP_VALIDATION.md)
uses the fixed original M20 participant split, extracts **only
original VAL Push-up video clips**, and compares fixed 200-tree
full-body-trained and upper-expanded RF candidates on identical
paired VAL clips using the same 64-feature schema. Recovered
upper-only clips and gate coverage are reported separately.
No original TEST video, inference endpoint or hosted model is
accessed or updated.

---

### M25 - Push-up TRAIN-only LOSO reliability and subject uncertainty

The owner's M24 grouped TRAIN cross-validation showed increased
Push-up upper-body coverage (83 vs 69 admitted clips) without an
improvement in matched-video Balanced Accuracy: 56.22% full vs
47.69% upper-paired vs 49.24% upper-expanded on 69 matched
held-out TRAIN clips. Upper-expanded improved bad-label recall
(47.06% vs 35.29%) but nearly doubled false-positive bad
predictions (17 vs 8 out of 35 good clips).

[M25 original TRAIN participant LOSO and cluster uncertainty](docs/M25_PUSHUP_LOSO_ROBUSTNESS.md)
repeats an individual-subject-heldout comparison and estimates
approximate paired subject-bootstrap intervals on fixed out-of-fold
predictions. Data provenance checks and privacy-safe aggregate
output are built in. Original M20 validation and final TEST
remain untouched, and no new Push-up model is deployed.

---

### M24 - Compare new Push-up RFs on exactly matched TRAIN-only clips

Real M23 owner extraction recovered **14 previously rejected front-view
Push-up clips** in original TRAIN: **69/84** accepted using full-body
admission, versus **83/84** using upper-body admission, with no file
failures. Both original publisher good/bad labels gained seven
front-view clips.

[M24 participant-heldout paired RF evaluation](docs/M24_PUSHUP_GROUPED_CV.md)
compares original full-admitted video features against upper-admitted
video features on **identical held-out common clips**, using 5-fold
subject-separated cross-validation exclusively within the original
frozen TRAIN subjects. Expanded upper-gate training also uses
newly recovered TRAIN-only clips; their extra held-out predictions
are reported separately instead of changing the common denominator.

This is an offline analysis and **does not touch the original M20
validation/TEST scoring or deployed Fitness-AQA detectors**.

---

### M23 - Frozen Push-up participant split and new upper-body clip schema

Owner M22 TRAIN ablation: baseline original full-body accepted
**24/513 frames, 0/15 rejected clips**; baseline upper-body accepted
**258/513, 14/15**, with heavy/lower-threshold detection providing only
modest additional frame coverage.

[M23 frozen TRAIN-only Push-up features](docs/M23_TRAIN_ONLY_PUSHUP_SCHEMA_PILOT.md)
builds paired full/upper admission data using a NEW
`gym7-pushup-upper32-clip64-v1` schema with masked invisible joints
and the original M20 seed-42 frozen TRAIN participant membership.
Validation and TEST participant media are not decoded.
**No M23 Push-up model is trained or deployed yet**.

---

### M22 - Pairwise Push-up detector and upper-body gate ablation

Real owner M21 TRAIN-only diagnostics recorded **262/513** frames rejected
for complete-side limb visibility, **227/513** with no detectable pose,
and **24/513** accepted. Ankles and knees were most often invisible.

[M22 TRAIN-only ablation](docs/M22_PUSHUP_TRAIN_GATE_DETECTOR_ABLATION.md)
compares three MediaPipe pose-detection configurations against the same
original complete-side and experimental upper-body admission gates on
identical sampled frames. It is diagnostic-only, includes synthetic
holdout isolation tests and does not retrain or deploy models. Independent
validation and TEST camera videos remain untouched.

---

### M21 - Diagnose front-view Push-up coverage loss (local, TRAIN only)

The owner extracted **266/294** Gym7 videos. All **28 rejected**
clips came from **front-view Push-up** (14 original good + 14 original
bad), where coverage was **20/48 (41.7%)**; all Curl and other
Push-up camera views had 100% pose clip extraction coverage.

Use the new [M21 local front camera pose audit](docs/M21_PUSHUP_FRONT_COVERAGE_AUDIT.md)
to measure whether the failures stem from no detected person, the
original same-side limb visibility rule, degenerate torso scale or
too few valid frames. The audit **decodes only original TRAIN-subject
rejected clips**, not VAL/TEST subjects, and does not change the deployed
six source-RF models or the M20 baseline.

---

### M20 - additional gym exercises (local research pipeline, not yet deployed)

The scientific aim is **exercise movement-quality research**, not merely
drawing pose skeletons. The deployed 6 RFs perform real source-label
association for Squat, OHP and Row; their outputs are NOT independently
validated technique-correctness or injury-safety judgements.

A new **opt-in local** research pipeline downloads one 2.9 GB publisher
Mendeley Gym7 ZIP, builds owner-reviewable subject labels, extracts clip
features one video at a time, and trains grouped **Push-up and Bicep Curl**
good/bad *video-level* Random Forest pilots. Original per-video good/bad
labels are kept separate from the original named Fitness-AQA error labels.
TEST is locked by default, training files are ignored by Git, and new
models are NOT silently offered in the live single-frame scoring API.
This repository cannot write directly to the owner's Linux computer.

[Run the M20 local download, extraction and training guide](docs/M20_GYM7_LOCAL_TRAINING.md).

---

### M19 mobile cameras, premium studio UI and new dataset research

The mobile site now offers Front / Selfie and Rear / Observer camera
selection; mirror on/off flips **only the front preview**, not the numeric
MediaPipe landmarks supplied to the experimental source-label models.
Switching physical cameras stops the prior stream and resets research
inference consent. A full responsive movement-studio redesign highlights
camera controls, joint visibility and model status on mobile.

Tracked human-readable docs and source use simple ASCII hyphen instead of
Unicode en or em dashes. Valid CLI `--flags` and CSS `--properties` are
preserved. CI enforces this convention.

[M19 mobile camera and UX acceptance](docs/M19_MOBILE_CAMERA_WEB_REFRESH.md)
and [additional exercise dataset research](docs/M19_EXERCISE_DATASET_EXPANSION.md).
**No fourth exercise or newly trained model has been deployed.**

---

### M18 zero-cost deployment preparation (no paid Render plan)

The optional **free-tier** Render Blueprint (`render.yaml`) creates NO
persistent disk and uses Render's free 0.1 CPU / 512 MB instance. Owner-trained
six private `.joblib` artifacts are **never** included in public Git. The
owner can upload only those six files to a separate **private** Hugging Face
model repository; the backend fetches exactly six SHA256-verified files at
one pinned commit before trusting any model. No secrets are committed.

**No paid resource or cloud inference has been activated.** Render free
services sleep when idle and can run out of RAM; only a successful
real-owner 6/6 `/health` + `/ready` check verifies deployment.
Browser/simulation research accuracy still has not been established.

[Free Render setup, local private upload script, six-model bootstrap and
Vercel acceptance](docs/M18_RENDER_PRIVATE_MODEL_HOST.md).

---

### M18: six-model deployment readiness gate (pending owner cloud artifacts)

The Web v2 frontend and optional Python scoring API are integrated, but the
deployed Vercel site is **not yet connected to a publicly provisioned six-model
backend**. M18 now requires all six original-source research RFs before
enabling the user opt-in; `/ready` returns 503 when any is missing, and a
read-only HTTPS preflight checks exact model availability without transmitting
landmarks. Validation errors do not echo pose coordinates. This does **not**
fix the unvalidated browser/source pose distribution shift.

```bash
python -m scripts.check_web_inference_deployment --api-url http://localhost:8000
```

[Follow the M18 deployment acceptance and model-provisioning guide](docs/M18_SIX_MODEL_DEPLOYMENT_ACCEPTANCE.md).

---

### Web v2: opt-in six source-label RF inference API

The deployed browser camera and pose visibility workbench now has an
optional **real Python inference backend** integration for all six
original-source models (two Squat, two OHP, two Barbell Row). Users
must explicitly opt in before their **33 pose landmarks** leave the
browser; raw camera frames never leave. The backend reuses exact
existing `ModelRegistry` feature generation and thresholds, and
returns genuine experimental source-label scores or `ABSTAIN`.
There are **no invented scores or medical / safe-form judgments**.

```bash
python -m pip install -e '.[webapi,dev]'
python -m pytest -q
# Models stay on the owner's computer or separately provisioned private host.
POSE_TRUST_LOCAL_ARTIFACTS=1 POSE_MODEL_ROOT=. \
POSE_WEB_ORIGINS=http://localhost:5173 \
uvicorn gym_pose.web_api:app --host 127.0.0.1 --port 8000 --no-access-log
# In separate terminal:
cd web
VITE_INFERENCE_API_URL=http://localhost:8000 npm run dev
```

**Vercel static frontend does NOT magically contain the six local joblib
bundles.** Backend hosting with private model provisioning and a
build-time `VITE_INFERENCE_API_URL` is required before deployed
public website users can see real scores. Otherwise it stays a
browser-only visibility demo with model inference unavailable.
CORS, no-pose abstention, request constraints and tests are included;
public backend abuse protection and cross-MediaPipe task/solution
parity are still open research/ops requirements.

[How to provision the model API and connect Vercel](docs/WEB_REAL_INFERENCE_DEPLOYMENT.md).

---

### Web MVP: open camera in your browser (deployable on Vercel)

A new `web/` Vite project brings the privacy-conscious pose
**visibility viewer** to a browser, with no Python install, repo clone
or model pickle required for end users:

```bash
cd web
npm install
npm test
npm run dev
# Open http://localhost:5173
```

It supports webcam pose skeleton, Squat/OHP/Barbell Row exercise
selection, per-source-task required-joint visibility, a local camera
Start/Stop control and responsive desktop/mobile UI. Source videos
or camera frames are **not uploaded**.

**Important:** Six publisher source-error models are still only
available in the owner's trusted **local Python research registry**.
The web demo shows each RF as `NOT DEPLOYED`. No model probabilities
are fabricated; an explicit backend+source-parity validation or
browser ONNX parity is needed before serving those actual classifiers.

Vercel deployment: import GitHub repo and configure **Root Directory
`web`**, Vite `npm run build`, output `dist`. Browser webcam
requires HTTPS/localhost and permission.

[Web MVP setup, deployment and privacy boundaries](docs/WEB_MVP_VERCEL_DEPLOYMENT.md).

---

### Research release source integrity - M17B

The owner confirmed **218 tests passed** and all **six** trusted local
original-source error models loaded on M17A. M17B checks that
the source release has the six correct versioned task contracts,
the original TRAIN/VAL research reproduction scripts, and no newly
tracked private Fitness-AQA media or derived features. Run:

```bash
python -m pytest -q
python -m scripts.audit_research_release
```

The new audit **does not** load model pickle files, access a webcam
or evaluate publisher TEST. The repo still tracks two **historical**
notebook assets (`models/squat_form_rf.joblib` and `models/yolov8s.pt`);
they are disclosed as legacy warnings, not confused with the six
newly trained source-label RF artifacts or silently deleted.

[Release checklist, older OHP PR reconciliation, and handoff](docs/M17B_RELEASE_AUDIT_AND_HANDOFF.md).

### Research release candidate - M17

[Read the M17 experimental results, original source-label tables,
real webcam runtime evidence, privacy scope, historical TEST exposure,
and explicit release readiness/blockers](docs/M17_RESEARCH_REPORT_AND_RELEASE_READINESS.md).

```bash
python -m pip install -e '.[dev]'
python -m pytest -q

# Status-only; no joblib loading or camera:
python -m scripts.check_fitness_aqa_model_registry

# Only run with artifacts that YOU trained and trust (joblib uses pickle):
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts

# Optional local webcam demo, no image recording:
python -m pip install -e '.[camera,dev]'
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 --trust-local-artifacts
```

Source code/tests do not bundle Fitness-AQA original videos/images,
raw features, or the owner's local trained joblib artifacts. A historical
pilot previously inspected some publisher TEST clips, so **do not claim
the full project history kept original TEST pristine**. Independent
athlete-level or clinical claims have NOT been established.

---

# Historical implementation and milestone notes

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

## Milestone 14C: Barbell Row shoulder-hip 20-feature extraction

The source TRAIN/VAL PR #22 audit established that requiring knees
and ankles was a major cause of the earlier **4-5%** Barbell Row pose
coverage. On equal-size label-stratified audit subsets, a strict
shoulder+hip four-joint gate admitted approximately **68-75%** of
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
but strict 8-joint MediaPipe pose coverage was only **4-5%** across
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