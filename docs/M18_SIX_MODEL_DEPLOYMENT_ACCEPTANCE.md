# M18 — Six-detector deployment acceptance (fail-closed)

**Status:** deploy-time validation integrated; **public six-model backend not provisioned**.
This is a research-only Fitness-AQA source-annotation association demo, NOT
automatic correct-form coaching, injury prevention, or clinical assessment.

## Verified boundary

The Vercel project `gym-pose-detection-system` serves the Vite pose viewer.
PR #37 already implemented an optional FastAPI backend, but the six real
owner-trained Random Forest `.joblib` bundles exist **only on the owner's Linux
computer**. A static frontend deployment never implies cloud weights are
available. In the owner's production Vercel project, `VITE_INFERENCE_API_URL`
was absent when M18 work began. **No backend origin may be fabricated.**

M18 adds:
- `/health`: exact 6/6 model count, names, and `ready_for_six_detector_demo`.
  1–5 loaded is explicitly `partial_model_artifacts`, not `research_api_ready`.
- `/ready`: 200 **only** when all six original source-label models are loaded;
  otherwise 503 (suitable for container/liveness routing).
- Frontend consent/inference remains disabled unless all six correct slots load.
- Invalid landmark validation replies are redacted; invalid input is not echoed.
- Read-only deployment preflight checks HTTPS origin, `/health` and `/ready`;
  it transmits **no landmarks** and never loads a pickle.

## A. Complete local verification (owner only)

1. Preserve `notebooks/gym_pose_detection.ipynb` and all ignored local data/models.
   Do not run `git clean -fdx` or commit the six artifacts.
2. Checkout updated `main` after merging M18. Install dependencies:

   ```bash
   python -m pip install -e '.[webapi,dev]'
   python -m pytest -q
   python -m scripts.audit_research_release
   python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
   ```

3. Run the actual owner-trusted research models (terminal one):

   ```bash
   POSE_MODEL_ROOT=. POSE_TRUST_LOCAL_ARTIFACTS=1 \
   POSE_WEB_ORIGINS=http://localhost:5173 \
   uvicorn gym_pose.web_api:app --host 127.0.0.1 --port 8000 --no-access-log
   ```

4. Read-only preflight:

   ```bash
   python -m scripts.check_web_inference_deployment \
     --api-url http://localhost:8000
   ```

   Exit 0 requires **all six exact named models** and both health probes.
   The preflight does NOT establish accuracy, source-domain parity or validation
   against independent athletes.

5. Frontend (terminal two):

   ```bash
   cd web
   npm install
   VITE_INFERENCE_API_URL=http://localhost:8000 npm run dev
   ```

   Open `http://localhost:5173`, manually start camera, then **separately**
   enable research inference. Confirm each exercise displays its TWO actual
   source-label probabilities or explicit `ABSTAIN`. Do not treat a synthetic
   CI test as equivalent to this real webcam acceptance.

## B. Cloud backend (requires owner-provided six model artifacts)

Deploy `backend/Dockerfile` via an owner-controlled HTTPS Python/Docker host
with a **private, persistent, read-only model volume**. Render/Railway or
a secured VM can be used **only if the chosen plan actually supports this**;
the public Vercel frontend is not itself a Python model host.

The volume root must contain `artifacts/fitness_aqa_squat_source_rf/`,
`artifacts/fitness_aqa_ohp_task_rf/`, and
`artifacts/fitness_aqa_barbell_row_torso4_rf/` with exactly the six
owner-trained model paths in `docs/WEB_REAL_INFERENCE_DEPLOYMENT.md`.

Configure on the backend, not in the frontend:

```dotenv
POSE_MODEL_ROOT=/private-models
POSE_TRUST_LOCAL_ARTIFACTS=1
POSE_WEB_ORIGINS=https://gym-pose-detection-system.vercel.app
```

Provision only **owner-approved/trusted** pickle artifacts through the
hosting provider's private file mechanism. Do not put artifacts in GitHub,
a public web root, Vercel `VITE_*` variables, browser localStorage, a public
artifact URL, or a container image committed to this repository. Verify the
server has sufficient RAM and writable temporary storage for the chosen
Python/scikit-learn runtime. Treat artifacts from other machines/unknown
sources as untrusted code, since pickle deserialization executes code.

**Before exposing `/score` on the Internet**, configure provider-side
per-IP/request rate limiting, resource caps, abuse protection and minimal
privacy-safe access diagnostics. CORS/Origin checking is not authentication;
outside-browser clients can send requests directly. A failed `/ready` must
not be routed as an available inference backend.

After HTTPS is live:

```bash
python -m scripts.check_web_inference_deployment \
  --api-url https://YOUR-API-DOMAIN
```

Only after it exits 0, set build-time environment variable on the existing
Vercel frontend project:

```dotenv
VITE_INFERENCE_API_URL=https://YOUR-API-DOMAIN
```

Redeploy Vercel and test the live consent flow on Squat, OHP, and Row. The
Vite variable is **public** at build time, so never put any token/secret
in it. Verify 33 landmarks leave only on explicit consent; raw camera
pixels and videos never leave the browser.

## C. Research acceptance, after deployment

Record per-exercise camera admissibility, ABSTAIN fraction and timing in
several lighting/camera-distance conditions. Validate numeric landmark
feature parity between MediaPipe Tasks Vision Lite (browser) and MediaPipe
Solutions Pose (source Python extraction) on *consented, separately acquired*
reference media. Differences must be measured rather than hand-waved.
Models have not been validated for independently unseen athletes; the
publisher TEST history has limited prior exposure and original validation was
repeatedly consulted. No fitness safety or correctness claims.

A fully ready software backend **is not** a scientifically validated
fitness-form detector. Release only as an experimental six-source-label viewer.
