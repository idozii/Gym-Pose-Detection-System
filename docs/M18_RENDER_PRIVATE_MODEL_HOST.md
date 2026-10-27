# M18 — Render private six-model backend handoff

The Vercel website remains the **browser-only pose viewer** until an actual
private Python backend has loaded the six owner-trained RF artifacts.
No original Fitness-AQA images/videos, source TEST media, feature CSVs or
model pickle files are committed or uploaded to a public Git repository.

This repository includes an **opt-in** `render.yaml`. Merely merging it does
not create a Render service or charge the owner. **Importing** the Blueprint
creates a **paid** web service and persistent disk. Review
[Render's plans](https://render.com/pricing) and confirm billing first.

## 1. Review and import the Render Blueprint

From Render Dashboard > **New > Blueprint**, link the existing GitHub
repository `idozii/Gym-Pose-Detection-System` and select
`render.yaml` on `main`. Do this only if the paid plan is acceptable.

The configuration is intentionally conservative:

- Service: `gym-pose-research-api`, Dockerfile `backend/Dockerfile`.
- Region: `singapore` (closest Render region to Sydney).
- Compute: `1c-2g` (paid 1 CPU, 2 GB RAM).
- Disk: 1 GB initially, mounted at `/private-models`. Confirm the six
  actual model files fit; disks can grow but **cannot shrink**.
- Port: platform `PORT` or local Docker fallback 8000.
- `autoDeployTrigger: off`: deployments only when intentionally triggered.
- Render health: `/health` (process healthy even before provisioning).
  `/ready` is the **distinct six-model readiness probe**, returning 503
  until all six correct models load.
- CORS origin: `https://gym-pose-detection-system.vercel.app`.
- `POSE_MODEL_ROOT=/private-models`.
- **Set `POSE_TRUST_LOCAL_ARTIFACTS=0` when the Blueprint prompts.**
  This is a deliberately user-provided setting (`sync: false`).
- One service instance (Render disks cannot support horizontal scaling).

The empty backend is expected to return
`status=model_artifacts_not_provisioned` from `/health` and HTTP 503
from `/ready`. The Vercel site **must not** be connected yet.

## 2. Safely upload ONLY the six locally owned model artifacts

On your **own Linux desktop**, in the project repository, first confirm
these are your self-trained trusted original-source models:

```bash
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

This command deserializes joblib/pickle, so NEVER use it on untrusted model
files. It must report all six `loaded_experimental_only` before proceeding.
Do **not** put the weights, original videos or training datasets into GitHub,
Vercel assets, Render environment variables, public uploads, or this chat.

Create a strictly enumerated bundle locally. It contains only the **six**
trusted `.joblib` models, not a whole directory or dataset:

```bash
tar -czf /tmp/gym_pose_six_source_rf.tar.gz -C . \
  artifacts/fitness_aqa_squat_source_rf/rf_error_knees_forward_static_source.joblib \
  artifacts/fitness_aqa_squat_source_rf/rf_error_knees_inward_static_source.joblib \
  artifacts/fitness_aqa_ohp_task_rf/error_elbows_static.joblib \
  artifacts/fitness_aqa_ohp_task_rf/error_knees_static.joblib \
  artifacts/fitness_aqa_barbell_row_torso4_rf/barbell_row_torso_angle_torso4_static_rf.joblib \
  artifacts/fitness_aqa_barbell_row_torso4_rf/barbell_row_lumbar_torso4_static_rf.joblib

tar -tzf /tmp/gym_pose_six_source_rf.tar.gz
ls -lh /tmp/gym_pose_six_source_rf.tar.gz
```

In the **Render Dashboard**, open your new service and use the **Connect >
SSH** instructions to add your public SSH key to Render and copy the exact
Render SSH target. Never expose the private SSH key. Note that
`backend/Dockerfile` provisions a non-root user with a proper shell and
private `~/.ssh` directory for Render SSH.

Open the service **Shell** and check:

```bash
whoami
ls -ld /private-models
mkdir -p /private-models/artifacts
```

If disk write permissions fail, stop and inspect ownership with
`ls -ld /private-models`. Do **not** broadly `chmod 777` or disable
container security. Render mount ownership must be repaired through an
authorized service configuration before uploading.

Use the **exact SSH target returned by your dashboard**, not a guessed
service username. For example, from your local computer:

```bash
scp -s /tmp/gym_pose_six_source_rf.tar.gz \
  YOUR_SERVICE@ssh.singapore.render.com:/private-models/gym_pose_six_source_rf.tar.gz
```

Run on the Render **Shell**, once the file has been transferred:

```bash
tar -tzf /private-models/gym_pose_six_source_rf.tar.gz
tar -xzf /private-models/gym_pose_six_source_rf.tar.gz \
  -C /private-models
rm /private-models/gym_pose_six_source_rf.tar.gz
find /private-models/artifacts -type f -name '*.joblib' -print
```

Confirm that each directory and artifact has the exact paths specified
above. Ensure the backend process can **read** the models, and avoid
loosening unrelated disk permissions.

After extraction, delete your temporary local tarball if no longer needed:

```bash
rm /tmp/gym_pose_six_source_rf.tar.gz
```

## 3. Turn on trusted inference and verify readiness

In Render Dashboard > service **Environment**, change
`POSE_TRUST_LOCAL_ARTIFACTS` from `0` to `1` and deploy/restart so the
FastAPI lifespan loads models **once at startup**. Do this only after
explicitly reviewing the six owner-generated files. If incompatible or
corrupted, the server may fail to start; inspect logs without exposing
model file contents.

Run the preflight on your local machine with the **real HTTPS endpoint**:

```bash
python -m scripts.check_web_inference_deployment \
  --api-url https://YOUR-REAL-RENDER-SERVICE.onrender.com
```

Exit 0 and `"ready": true` require `/health` and `/ready` to report all
six correctly versioned RF bundles. An untrusted public backend **must not**
be advertised as exercise-form or medical safety advice.

## 4. Protect public inference before wiring Vercel

`/score` serves CPU-consuming RF inference. Browser 2req/s pacing and CORS
are **not adequate public Internet abuse protection** (Origin can be absent
outside browsers). Before enabling for public traffic, put the backend
behind appropriate trusted platform/gateway per-IP rate limits, bot/abuse
protection and concurrency/resource caps. Do not log landmark payloads.
Paid disk single-instance deploys have brief expected downtime.

Once all six models pass AND external abuse protection is in place, configure
the **existing Vercel project** `gym-pose-detection-system` with the public,
non-secret **build-time** variable:

```dotenv
VITE_INFERENCE_API_URL=https://YOUR-REAL-RENDER-SERVICE.onrender.com
```

Redeploy Vercel. In production, open the browser, start the camera and
**separately opt into research inference**. Test Squat, OHP, and Barbell Row
(two named source-label probabilities or visible `ABSTAIN` each). No raw
frames should be uploaded. Check real camera results and quantitative
MediaPipe Tasks Vision / Python MediaPipe Solutions parity. Synthetic CI
tests and a 6/6 availability probe alone **do not** establish scientific
fitness-form validity.

## Current status of handoff

This documentation and Blueprint are repository-side deployment preparation.
They do **not** demonstrate an actual Render service, a paid resource being
created, any transferred private weights, or an inference-connected Vercel
production build. Maintain these distinctions when reporting progress.

Sources: [Render Blueprint reference](https://render.com/docs/blueprint-spec),
[Render persistent disks and SCP](https://render.com/docs/disks),
[Render SSH requirements](https://render.com/docs/ssh).
