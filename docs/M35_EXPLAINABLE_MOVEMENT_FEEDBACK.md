# M35 - Explainable Movement Feedback (local research preview)

## Motivation and evidence

The owner has run M34 on a real Ubuntu webcam and reported
that the observed Squat/OHP/Barbell Row movement-cycle counts
work reasonably well. This is **informal usability feedback,
not externally annotated rep-level precision/recall**.

The previous original Fitness-AQA six models output
*publisher-source annotation associations* with unvalidated
browser/Python landmark source parity and modest validation
quality. The ML results are **not** reliable enough to
display "GOOD FORM", "BAD FORM", injury warnings or
prescriptive lifting instructions. Neither a low RF score
nor a completed M34 movement cycle implies good technique.

M35 therefore prioritizes interpretable and truthful local
feedback rather than inventing a form-grade threshold.

## What changes in the browser

For the original three exercises, an
**M35 / EXPLAINABLE OBSERVATIONS** panel appears below the
M34 angle-cycle summary.

- **Camera offline**: no evaluation and ask user to start camera.
- **Insufficient tracking**: request improved camera framing;
  preserve previously observed cycle count but do not infer
  an additional rep.
- **0 completed cycles**: angle visible; no complete
  *heuristic* angle cycle yet. Do not say an incomplete
  motion is incorrect technique.
- **1-2 observed cycles**: show the current cycle count and
  mean observed duration, without inventing comparisons.
- **At least 3 observed cycles with valid timing history**:
  compare the *last up to five* cycle durations, report
  observed min-max time in seconds and mean camera-plane
  angular excursion. The local text may call durations
  "vary" if the max-min time span is >=30% of the mean.
  This is a **descriptive display threshold only**, not
  a physiotherapy rule, a recommended tempo or a quality
  classification.

The M34 counter still owns the movement counts, 400ms
sampling and fixed 2D heuristics. M35 reads only its
existing bounded ephemeral summary records and M33
camera-plane landmark measurements. It adds **no
network request, video upload, model training or new
personally identifying data**.

The camera instructions and limitations are specific
to each exercise: Squat knee angle, OHP elbow angle,
and Row torso presentation plus elbow-flexion cycle
count. In particular, shoulders/hips do not permit a
direct lumbar curvature diagnosis.

## Original source-label RF context remains separate

The M35 panel includes a distinct **SOURCE MODEL CONTEXT /
OPTIONAL RESEARCH ONLY** section:

- When remote research inference is OFF, say so plainly.
- If consent is ON but no trusted, task-matched result
  has arrived, display unavailable/awaiting.
- When the backend returns a source-label report for the
  correct exercise, explain that the scores relate to
  **original publisher labels**, not whether *this*
  repetition was correctly executed. A low source
  score **does not prove good form**.

The original opt-in flow is unchanged:
strict six-model trusted health readiness, no backend
score request for Push-up, only 33 landmarks transmitted
for the original three exercises after consent, and
source estimator parity still unvalidated. No RF
model thresholds are reinterpreted as safety cutoffs.

## Local testing (Ubuntu)

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
cd web
npm install --no-audit --no-fund
npm test
npm run build
npm run dev
```

Then open `http://localhost:5173` and try:

1. Squat: start camera; before tracking is ready,
   verify "Need a clearer camera view". Show visible
   knee geometry, then complete 1-2 slow cycles.
   Confirm the feedback only describes the detected cycles.
2. After >=3 recorded cycles, verify the panel reports
   your observed tempo range and approximate angle excursion
   without asserting "correct/wrong".
3. OHP and Row: switch exercises; all counters and
   previous feedback must reset. Row should still use
   elbow flexion for M34 cycles, not torso tilt.
4. Cover a required joint or stop camera: the panel
   must explain why a new measurement is unavailable.
5. Test "Reset set": the old cycle comparison must
   disappear. Do not use webcam video or pose data
   for research inference unless you explicitly consent.

Synthetic Node tests assert each status, duration
comparison, malformed history, source-report provenance,
separation from Push-up and lack of added network/storage.
Existing Python and web CI continue to protect backend
behavior and prior milestones.

## Honest next step to true form feedback

M35 is **not** "AI correct/wrong training" and does
not turn the original source-label RF probability
into an injury or form diagnosis. A credible
correct/wrong recommendation needs a new,
well-defined ground-truth rubric; independently
annotated complete repetitions from *new people*
across camera views; proper subject-disjoint
evaluation; rep-level false-alert and missed-error
estimates; and browser vs training feature parity.

Avoid opening the original Gym7 M20 TEST or tuning
previously inspected VAL to make the UI look accurate.
A public deployment is separate from merging
into GitHub. Vercel Hobby quota or GitHub Pages
settings may still prevent the new UI from going live.
