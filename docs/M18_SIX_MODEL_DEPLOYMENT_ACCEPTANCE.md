# M18 - Six-detector deployment acceptance (fail-closed)

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
  1-5 loaded is explicitly `partial_model_artifacts`, not `research_api_ready`.
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

## B. Zero-cost Render backend (requires owner-owned private models)

The user requested **no paid services**. Use the optional **free** Render
Blueprint, no persistent disk and no SSH. Instead, the owner publishes
**exactly six self-trained trusted files** to their **private Hugging Face
model repository**, with an owner-generated SHA256 manifest and pinned
40-character commit ID. Render Free retrieves and verifies them on startup
only after explicit owner trust.

See the complete [zero-cost Render + private model hosting guide](M18_RENDER_PRIVATE_MODEL_HOST.md).
The real API must pass:

```bash
python -m scripts.check_web_inference_deployment \
  --api-url https://YOUR-ACTUAL-SERVICE.onrender.com
```

No backend URL is assumed or invented; **do not** configure Vercel until
6/6 models are loaded, `/ready` passes, and abuse protection is appropriate.
Render Free has only 512 MB RAM, no stable disk and idle sleeping; real
model-memory feasibility is still unverified. The exact owner model files
cannot be fetched from the public GitHub repository and must never be
published to the browser.

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
