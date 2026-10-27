# M18 — zero-cost experimental Render API + private Hugging Face weights

**0 paid compute, 0 paid persistent disk, 0 paid storage by design** within
providers' current free quotas. No resources are created by merging this PR.
Public research API remains **NOT DEPLOYED** until the owner provisions their
six self-trained RF bundles and verifies end-to-end inference.

The publisher-source annotation scores are **not** verified correct exercise
form, medical spine diagnosis, or injury-prevention advice.

## Architecture

```text
Vercel Free (existing browser pose viewer)
      user manually opts into landmarks (not camera images)
      HTTPS POST /score
Render Free (FastAPI + sklearn; 512 MB RAM / 0.1 CPU)
      boot: download exactly six joblib models + SHA256 manifest
      from owner's PRIVATE Hugging Face model repo at pinned commit
      only if POSE_TRUST_LOCAL_ARTIFACTS=1
      local /tmp file cache disappears when container sleeps/restarts
```

No private `.joblib` binary, publisher videos, archives, raw features,
images, local notebook or token is committed to the public GitHub repo.
Do not upload those files to this chat.

**Free-tier limitations**: Render may sleep after 15 minutes idle, wakeup
may take approximately one minute *plus* model retrieval and import. Free
compute is 0.1 CPU with 512 MB RAM, so six RF models might not fit: **actual
model size and RAM usage have not yet been verified**. This is a hobby
research demo, not guaranteed always-on hosting. No free Render SSH,
persistent disk or shell is needed for this route. Hugging Face private
storage has a free quota, subject to their limits. Make sure your Render
workspace has **no payment method** if you require services to stop rather
than bill after free usage limits. Do not enable paid hardware or disk.

## Step 1 — publish only the owner's original six models to a PRIVATE repo

Perform on your own Linux computer in the existing repository (all ignored
`artifacts/`, datasets and notebook remain untouched by Git):

```bash
git pull --ff-only
python -m pip install -e '.[webapi,dev,hfmodels]'
python -m pytest -q
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
hf auth login
python -m scripts.publish_private_model_bundle \
  --repo-id YOUR_HF_USERNAME/gym-pose-private-models \
  --trust-local-artifacts
```

The upload script uses your **locally authenticated Hugging Face account**,
creates/reuses a PRIVATE model repo, checks it is truly private before any
upload, and uploads **only six known files + one SHA256 manifest**. The owner
must explicitly pass `--trust-local-artifacts`: pickle code execution is
possible when loading the owner's models. Never run on untrusted joblib.

Copy the three **non-secret** values printed by the command, such as:
`POSE_HF_REPO_ID`, `POSE_HF_REVISION` (full 40-char commit SHA), and
`POSE_HF_MANIFEST_SHA256` (64-character hexadecimal digest).

Create a **read-only, fine-grained Hugging Face access token** that can read
only the owner's private model repository, if your account permits that scope.
Do not share it with ChatGPT, GitHub issues, public README, Vercel
`VITE_*` variables or other users.

## Step 2 — import the free Render Blueprint

In the [Render Dashboard](https://dashboard.render.com), select
**New → Blueprint**, choose
`idozii/Gym-Pose-Detection-System` on `main`, and use `render.yaml`.
Review the configuration before creating it:

| Property | Value |
|---|---|
| Plan | **free** |
| Region | singapore |
| Runtime | backend/Dockerfile |
| Persistent disk | **NONE** |
| Model path | /tmp/gym-pose-models |
| Explicit model trust | 0 (disabled initially) |
| Allowed browser origin | https://gym-pose-detection-system.vercel.app |
| Auto deploy | disabled |

In the Render environment, configure:

```dotenv
POSE_HF_REPO_ID=YOUR_HF_USERNAME/gym-pose-private-models
POSE_HF_REVISION=FULL_40_CHARACTER_OWNER_COMMIT_SHA
POSE_HF_MANIFEST_SHA256=SHA256_PRINTED_BY_UPLOAD_SCRIPT
HF_TOKEN=THE_READ_ONLY_TOKEN_STORED_AS_A_RENDER_SECRET
POSE_TRUST_LOCAL_ARTIFACTS=1
```

If not ready, keep `POSE_TRUST_LOCAL_ARTIFACTS=0`; the empty FastAPI may
boot but `/ready` must respond 503, and browser inference stays off.
You must **manually deploy/restart** after updating these variables,
since `autoDeployTrigger` is disabled.

The container first downloads and verifies exactly six model digests.
If any manifest is invalid, token is missing or downloaded bytes differ,
startup aborts. No model is ever loaded if verification fails. Model
deserialization happens only after the owner opted into these trusted files.

## Step 3 — verify six-model readiness BEFORE connecting Vercel

On the local computer, run (replace with the real hostname Render gave you):

```bash
python -m scripts.check_web_inference_deployment \
  --api-url https://YOUR-ACTUAL-SERVICE.onrender.com
```

Must exit 0 and report `"ready": true`, `"loaded": 6`. A 503 or
startup error means **do not advertise or connect the AI**. If a 512MB OOM
occurs, there is no honest six-model free cloud deployment on this Render
plan without further quantization, reducing memory/feature runtime, or
choosing an alternative free resource with sufficient RAM.

**Before enabling public traffic** protect CPU and abuse: the existing
browser paces to 2 requests/sec and the API has CORS/body caps, but these
**do not** stop direct nonbrowser spam. Implement/enable appropriate
per-source request throttling and concurrency protections and keep logs
free of landmark payloads. Consider a private/demo-only testing scope.

Only when all six are loaded and reasonable access protections are in place,
set on the **existing Vercel frontend** (build-time public URL only):

```dotenv
VITE_INFERENCE_API_URL=https://YOUR-ACTUAL-SERVICE.onrender.com
```

Redeploy Vercel. Start camera; separately opt into source-label inference.
Check two scores or `ABSTAIN` per exercise (Squat, OHP, Barbell Row).
Avoid safety or exercise correctness claims. Validate real webcam landmark
parity vs Python MediaPipe Solutions training source before using scores
as research findings.

## Fallback without another cloud account

A user-controlled, always-running local Linux machine can serve the same
FastAPI with the existing local six models; a free
[Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel)
can expose that one HTTPS backend to testers **if the user explicitly chooses
to make the local server public**. This avoids uploading weights, but the
computer must remain on and protected against abuse. It is not a hosted
always-on product. Don't open your home router or port-forward it casually.

**Sources:** [Render Free](https://render.com/docs/free),
[Render plans](https://render.com/docs/compute-plans),
[Hugging Face private storage](https://huggingface.co/docs/hub/storage-limits),
[private versioned file downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download).
