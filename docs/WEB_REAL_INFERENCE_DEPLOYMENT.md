# Web v2 — six real Fitness-AQA source-error RFs (opt-in research API)

## What has been implemented

Existing live site: **https://gym-pose-detection-system.vercel.app**
(Vercel project `gym-pose-detection-system`), with browser-local
MediaPipe Tasks Pose Landmarker and manual exercise selection.
Six trusted Python scikit-learn Random Forest bundles exist on the
owner's **local Linux computer only**, not in GitHub or Vercel.

Web v2 adds:
- `gym_pose.web_api:app`: FastAPI `/health` and `/score`.
  It reuses the **exact source-trained** Python ModelRegistry,
  Squat 103-feature, OHP task-specific 30-feature, and Row torso4
  20-feature extractors, pose quality gate >=0.50, and
  train-only thresholds. No model retraining or TEST evaluation.
- Strictly **33 JSON landmarks** (x/y/z/visibility), sent from browser
  **only after separate consent** (unchecked by default), <=2 req/s.
  The original video never goes to Python; no camera photos/frames,
  feature vectors, per-user identities, or session data are persisted.
- UI displays actual `error_probability` *publisher-source annotation
  scores* only when returned by the backend. It renders `ABSTAIN`
  for low-quality poses, missing models, inference errors or invalid
  scores. If the backend is absent, the UI explicitly reads
  `NOT DEPLOYED` / `MODEL SERVER NOT CONFIGURED`, never fake scores.
- Backend CORS restricted to a specific HTTPS/localhost frontend
  origin; finite input bounds, 16 KB body cap, disabled API docs, no
  response caching; Docker entrypoint disables HTTP access logs.
- No public model pickle deserialization; only the server operator
  can set `POSE_TRUST_LOCAL_ARTIFACTS=1` to use **their OWN files**.
- JS and Python tests use synthetic landmarks and six synthetic
  versioned Random Forest bundles; no test archive, webcam or owner's
  real model artifacts are part of CI.

## Why backend cannot instantly activate on Vercel frontend

The six `.joblib` artifacts are **not present on Vercel** or in this
GitHub repository and should not be casually added to public Git.
A static Vite site cannot execute scikit-learn or load Python pickle.
Publishing the website does not magically supply the private weights.

**Provision a separately hosted Python model API with private storage.**
Until that exists, deployed website retains pose inspector functionality
and consent toggle is disabled; the web UI does not pretend the models
have been deployed.

## A. Local end-to-end model test first (owner desktop)

```bash
git fetch origin
git switch feat/web-real-source-inference-api 2>/dev/null || \
  git switch --track origin/feat/web-real-source-inference-api
git pull --ff-only
python -m pip install -e '.[webapi,dev]'
python -m pytest -q

# Reads ONLY your six self-trained source-error bundles, not TEST:
POSE_WEB_ORIGINS=http://localhost:5173 \
POSE_MODEL_ROOT=. \
POSE_TRUST_LOCAL_ARTIFACTS=1 \
uvicorn gym_pose.web_api:app \
  --host 127.0.0.1 --port 8000 --no-access-log
```

Open another terminal, still in the same checkout:

```bash
cd web
npm install
VITE_INFERENCE_API_URL=http://localhost:8000 npm run dev
```

Open **http://localhost:5173**, Start camera, then check the separate
**Enable research inference** consent box. Inspect
**http://localhost:8000/health**: should report all six
`loaded_experimental_only`. When viewable joints meet original gates,
the UI can show real source-label scores, tagged as experimental.

The server accepts **JSON landmarks, NOT webcam pixels**. A request
with an invalid shape/visibility returns HTTP 422; large payload
HTTP 413; untrusted website Origin 403. These restrictions do not
prove full public Internet abuse resistance. No independent accuracy
or injury advice is being added.

Alternative Docker local test from repo root:

```bash
POSE_WEB_ORIGINS=http://localhost:5173 \
docker compose -f compose.inference.yml up --build
```

Only the six trusted original-source files under `artifacts/` are
mounted read-only. The Docker build context omits media and local
model files via root `.dockerignore`.

## B. Provision the backend privately on a real hosting service

Use a Python/Docker backend host that supports **private persistent
model files** (e.g. a VM with HTTPS reverse proxy; Render/Railway
container plus private persistent volume if available on your plan).
This is a **separate service** from the existing Vercel `web/`
static site. The source repo contains a `backend/Dockerfile`; its
build does NOT contain model binaries.

Place only owner-trained artifacts at:

```text
<POSE_MODEL_ROOT>/
  artifacts/fitness_aqa_squat_source_rf/
    rf_error_knees_forward_static_source.joblib
    rf_error_knees_inward_static_source.joblib
  artifacts/fitness_aqa_ohp_task_rf/
    error_elbows_static.joblib
    error_knees_static.joblib
  artifacts/fitness_aqa_barbell_row_torso4_rf/
    barbell_row_torso_angle_torso4_static_rf.joblib
    barbell_row_lumbar_torso4_static_rf.joblib
```

The owner must consciously approve sharing/deploying these trained
weights. **Never upload source TRAIN/VAL/TEST media, raw photos, or
unknown/downloaded pickles.** Only the owner should supply/trust these
particular files. Joblib/pickle executes code on load; the public
website cannot upload or select which pickle to load.

Set backend environment variables:

```dotenv
POSE_MODEL_ROOT=/private-models
POSE_TRUST_LOCAL_ARTIFACTS=1
POSE_WEB_ORIGINS=https://gym-pose-detection-system.vercel.app
```

When hosting behind TLS, check `https://YOUR-API/health` reports
six models loaded and `research_only=true`. **Do not expose a
public Python service without platform-level rate limits, bot/abuse
protection and resource caps;** the Origin allowlist alone does
not authenticate callers. Requests can be sent outside a browser.
If hosting untrusted clients, add an API gateway/edge rate limiter
and monitor CPU/abuse counts without logging landmark payloads.

## C. Connect the deployed Vercel frontend

In existing project **gym-pose-detection-system**, set **Build-time**
environment variable:

```text
VITE_INFERENCE_API_URL=https://YOUR-API-DOMAIN
```

Redeploy the frontend. Your Vite JS bundle embeds this public
non-secret URL, so do NOT place tokens, joblib bytes or any secrets
in VITE_* variables. The consent box becomes available; it
checks backend health first and sends landmarks only when explicitly
enabled. Switching exercises clears prior results; Stop/hidden tab
aborts in-flight requests and removes consent.

If models are absent/host fails or request rejected, the correct
UI behavior is to display an error/ABSTAIN state, NOT a fabricated
probability. The Vercel static frontend stays usable without
backend in pose-only mode.

## Scientific and technical blockers before marketing as AI coaching

The browser uses **MediaPipe Tasks Vision Lite**, while source
models were fitted on original images/videos through **Python
MediaPipe Solutions Pose**. Their normalized XYZ coordinates, Z scale,
camera tracking and occlusion estimates can differ. We have
**not verified feature tensor parity, per-camera probability stability
or athlete-independent accuracy**. The source validation numbers
have been consulted repeatedly and original historic pilot examined
some TEST media, so avoid pristine-holdout claims. Squat Knee Forward
had ~37.5% source-val FPR, Knees Inward ~25% source-val recall;
Row `lumbar` four-joint view cannot measure spinal curvature.

This release is **only experimental source-label association**, with
no automatically issued movement correction, danger warning, fitness
technique certification or clinical safety conclusion.

M18 CI tests can establish *software correctness using synthetic
landmarks and RFs*, but real-owner end-to-end evaluation and
separate backend model provisioning remain required.
