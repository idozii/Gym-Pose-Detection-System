# Browser Research Viewer — Web MVP (Vercel-ready static frontend)

## What actually works in this version

The repository's Python/OpenCV viewer works locally, but the six
`joblib` research RF bundles exist **only on the owner's machine**,
are intentionally ignored by Git and are not on Vercel. A generic
static site cannot load or run those Python pickles safely.

`web/` is a **standalone deployable browser pose visibility demo**:
- User opens a public HTTPS site and deliberately clicks **Start camera**.
- The browser asks for camera permission and downloads the MediaPipe
  Tasks Vision runtime and an official pose landmarker asset.
- The app runs **MediaPipe Pose Landmarker in the browser**; it draws
  a skeleton overlay on the local camera view. Camera video/frames
  are **not uploaded, stored, or sent to a Python backend**.
- User selects Squat, OHP, or Barbell Row manually; per-target
  source-required joint visibility is visualized at cutoff 0.50.
- The inspector provides basic pose-seen fraction, observed UI FPS,
  required joint visibility status and **explicit** indications that
  all six Python source-label RF models are **NOT DEPLOYED**.
- The app **never fabricates probability scores, error warnings,
  injury advice, body correction, rep counts, or clinical predictions**.
  Basic visibility readiness is not equal to full source feature
  admission and has not been shown to correlate with exercise form.
- Includes camera OFF/ON, denial/failure handling and camera teardown
  on Stop, hidden tab and navigation away. No accounts/logins.

**Privacy caveat:** camera frames remain in the local browser. The
MediaPipe model/runtime are downloaded from Google/JSDelivr at startup;
Google's MediaPipe Tasks documentation notes the package may send
usage/performance metrics. Google Fonts used for UI typography can
also make third-party asset requests. Do not describe this as fully
air-gapped or as making zero connections.

## Develop locally

```bash
cd web
npm install
npm test
npm run dev
# http://localhost:5173
```

Requires Node 20.19+ / 22+. Running `npm run build` generates
`web/dist/`, which is ignored. There are no API keys or database
credentials in the frontend. The first camera start requires network
access to download the model/WASM.

## Deploy with Vercel + GitHub

1. Merge the approved Web MVP PR to GitHub main after CI passes.
2. Open [Vercel's new project page](https://vercel.com/new).
3. Import `idozii/Gym-Pose-Detection-System` as the repository.
4. Set **Root Directory = `web`** in project build settings.
5. Configure:
   - Framework preset: **Vite**
   - Install command: `npm install`
   - Build command: `npm run build`
   - Output directory: `dist`
6. Deploy to Vercel's HTTPS preview URL. Allow browser camera access
   and verify Start/Stop, pose skeleton, each exercise selection, and
   that all RF detector badges explicitly read `NOT DEPLOYED`.
7. Add custom domain only after the camera/permission/mobile smoke
   tests pass.

Vercel's Git root directory must be `web`; deploying from repository
root incorrectly would look for Python files, not the Vite app.
`getUserMedia` requires HTTPS or localhost in modern browsers.

No production URL has been claimed in this PR; until a deployment
succeeds, this is **deployable source code**, not a live website.

## What it takes for real six-detector browser use

The 6/6 Python research registry and local Squat webcam evidence
remain intact (236 Python tests in M17C). Browser MediaPipe
`tasks-vision` may produce different normalized pose landmarks and
tracking behavior than the Python `mp.solutions.pose` extractor
used to train those RFs. **Matching the joint IDs alone is not a
proof of feature/distribution parity.** A separate evaluation must
compare paired video/image feature tensors and coverage.

Two implementation options:
- **Backend research route:** retain exact Python feature and model
  pipeline in a FastAPI HTTPS backend. Browser would send only
  opt-in normalized landmark vectors, never raw camera frames.
  Backend must have user-provided, safely deployed versions of all
  six trusted joblib artifacts. Add strict CORS/origin allowlist,
  auth/rate limits as needed and clear privacy notice; don't expose
  owner-specific models until provenance/licensing is reviewed.
- **On-device inference route:** convert each source RF to ONNX and
  verify probability/threshold parity + input schema against
  trustworthy Python reference on synthetic and consenting real
  samples. Port **all** source-specific feature extraction functions,
  including the Squat 103-feature contract. Only then claim the
  browser is locally running those six classifiers.

Neither option is implemented yet. Do not add a backend URL or show
model probabilities until model artifact provisioning, test parity,
privacy and error abstention behavior are demonstrated.

## Acceptance checklist

| Gate | Web MVP |
| --- | --- |
| CI web unit tests | Required on PR |
| CI Vite production build | Required on PR |
| No publisher TEST use | YES, no dataset import |
| Camera video uploads | NONE implemented |
| Pose visibility | Browser-only 3-exercise UI |
| Six RF inference | **NO — not deployed** |
| Original Python research viewer | Unmodified |
| Source-label safety claims | Explicitly disallowed |
| Live Vercel URL | Pending actual deployment |
