# Real research inference API

Run from repository root with your **own** original source-error artifacts
in the exact `artifacts/` paths recognized by `ModelRegistry`.

```bash
python -m pip install -e '.[webapi,dev]'
POSE_TRUST_LOCAL_ARTIFACTS=1 \
POSE_MODEL_ROOT=. \
POSE_WEB_ORIGINS=http://localhost:5173 \
uvicorn gym_pose.web_api:app --host 127.0.0.1 --port 8000 --no-access-log
```

The browser sends 33 normalized xyz+visibility landmarks only after a
separate, explicit consent checkbox; never send video/images.
Backend returns **two actual RF source annotation scores** or
`abstain`. Models are never stored in the repository, and are not
downloaded from user URLs. No model ever loads unless the SERVER
OPERATOR explicitly sets `POSE_TRUST_LOCAL_ARTIFACTS=1`, trusting
locally provisioned joblib files. This is not a public authentication
mechanism. Configure `POSE_WEB_ORIGINS` to your exact HTTPS frontend
origin (comma-separated), never use `*`.

On production hosting, mount the model bundles (private persistent disk
or operator-managed secure artifact deployment) at
`POSE_MODEL_ROOT/artifacts/fitness_aqa_...`. Use a service with
private artifact storage and Python scikit-learn support (for example
Render/Railway, not a frontend-only Vercel static build).

Before public exposure add platform rate limits and appropriate
request throttling / abuse protection; origin allowlisting alone
is not authentication. No original publisher TEST files are accessed.

**Important:** Web Tasks Vision and original Python MediaPipe pose may
differ. API returns only exploratory source-label correlations; model
portability, camera domain shift, consented real-world accuracy and
possible unsafe false positives remain unvalidated.
