# OHP M13: Task-aware pose visibility feasibility (follow-up to real pilot)

## Ground truth from owner's actual annotation-stratified 60-clip pilot

Full output of `audit_fitness_aqa_ohp_quality`, 30 publisher train
+ 30 publisher validation clips, 2 FPS and 0.1-second annotation margin:

| Metric | Train | Validation | Total |
|---|---:|---:|---:|
| Eligible sampled frames | 276 | 302 | 578 |
| Accepted with production OHP 111-feature gate 0.50 | 44 | 42 | 86 (14.9%) |
| Accepted with all 12 critical joints at 0.35 | 61 | 55 | 116 (20.1%) |
| No detected pose | 16 | 46 | 62 |
| Legacy Squat shoulder/hip/knee/ankle gate is first rejection | 149 | 168 | 317 |
| OHP elbow/wrist gate is first rejection | 67 | 46 | 113 |

A first rejection reason is exclusive, but multiple lower/upper joints
may be simultaneously occluded. Among **492** rejected eligible frames,
317/492=64.4% have the legacy base-joint gate as first failure. The
0.35 relaxed gate increases total accepted frames only by 30 in this pilot.

Critical **error-bearing** accepted frames:
- TRAIN elbows: 0/19 at 0.50, **still 0/19 at 0.35**.
- VAL elbows: 2/28 at 0.50, 4/28 at 0.35.
- TRAIN knees: 4/15 at both 0.50 and 0.35.
- VAL knees: 6/18 at 0.50, 7/18 at 0.35.

The video sample deliberately oversampled annotated clips. Results
are **not a population acceptance estimate**. The 0.35 experiment
was only a quality-filter counterfactual, not classifier performance.

## Key hypothesis: different errors require different reliable joints

A global one-size-fits-all OHP 111-feature gate excludes a frame if
**any one** of 12 joints has visibility below 0.50. Some OHP videos
may not show ankles even though the elbow/wrist chain is observable.
Conversely, a knee-related error should not be scored unless the
relevant hip/knee/ankle chain is reliable.

This PR adds **diagnostic admission-only rules**:

| Rule | Joints required at 0.50 | Additional restriction |
|---|---|---|
| `all_050` | Both shoulders, hips, knees, ankles, elbows, wrists | Production gate |
| `all_035` | Same 12 critical joints at 0.35 | Relaxation only, not approved |
| `elbows_arm_chain_050` | Both shoulders, elbows, wrists | Ignores legs (hypothesis for elbow task) |
| `elbows_arm_050_other_035` | Both elbows/wrists 0.50 | Shoulders, hips, knees, ankles at 0.35 |
| `knees_lower_chain_050` | Both shoulders, hips, knees, ankles | Ignores elbows/wrists (hypothesis for knee task) |
| `knees_lower_050_upper_035` | Both shoulders, hips, knees, ankles 0.50 | Elbows/wrists 0.35 |
| `all_except_ankles_050` | All production critical joints except ankles | Gate contribution check ONLY |

All rules require finite coordinates and a valid shoulder span.
They **do not create trustworthy model input features** where
excluded joints are unreliable. No existing RF can be evaluated
or deployed on those masks without defining a task-specific feature
schema and retraining on source train clips.

## Running (same deterministic 30+30 publisher train+val video IDs)

```bash
git fetch origin
git switch --track origin/feat/m13-ohp-task-aware-gate-feasibility
python -m pytest -q

python -m scripts.audit_fitness_aqa_ohp_quality \
  --videos-per-split 30 \
  --sample-fps 2 \
  --seed 42 \
  --min-visibility 0.5 \
  --alternative-visibility 0.35
```

Revisits the **same source videos** as earlier 60-video diagnostic;
not a statistically independent replication. Uses MediaPipe landmark
outputs directly on those clips, with no test video decode, no model
fitting, no new feature CSVs and no original ZIP modification.

Report at:
`artifacts/fitness_aqa_ohp_quality/diagnostic.json` (ignored by Git).

New sections under each split:
- `task_visibility_admission_policies`: number of eligible frames and
  error-positive frames admitted by every gate hypothesis;
- `task_specific_positive_recovery.error_elbows`: original gate
  versus strict arm-chain eligibility;
- `task_specific_positive_recovery.error_knees`: original gate
  versus strict lower-chain eligibility.

The script asserts `all_050` / `all_035` counts exactly match
the production OHP extractor's acceptance rules on **each** eligible
pose frame, preventing accidental gate changes.

## Decision boundary for moving toward M13B

If `elbows_arm_chain_050` recovers a meaningful fraction of truly
elbow-error-annotated frames, prototype a separate elbow-only
feature schema using solely reliable shoulder/elbow/wrist coordinates
(and explicitly represent pose abstentions), then train on publisher
TRAIN and evaluate on original VAL. Knee task should independently
use a reliable hip/knee/ankle feature chain.

If arm-chain admissions are still tiny for elbow positives,
investigate MediaPipe arm visibility / camera viewpoint or a different
detector. Do not simply lower every joint threshold to claim a better
model.

Keep publisher TEST untouched until all preprocessing, threshold
calibration and metrics are frozen. Publisher video IDs are not verified
athlete IDs. Nothing in this analysis establishes injury prevention
or clinically correct lifting technique.
