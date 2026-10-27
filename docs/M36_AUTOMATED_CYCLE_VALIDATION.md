# M36 Auto - No manual rep CSV or participant recording required

## Change of direction

The owner explicitly does **not** want to complete M36 with
manual rep counting or enter repeated session-count rows.
The original M36 manual protocol remains available for a
future genuine human-annotated validation but **is now
optional and is not needed to continue the project**.

This alternative runs a **deterministic engineering stress
test** of the unchanged M34 2D angle-cycle counter:

- 3 exercises: Squat, Overhead Press, Barbell Row;
- 3 generated 2D projection transformations: original,
  in-plane image rotation, and horizontal scaling;
- 2 body-side inputs (left/right);
- 0, 1, 3 and 5 known artificial angle cycles;
- clean motion and small numerical jitter, plus non-cycle
  stationary, partial bending, temporary pose loss, and
  mid-cycle visible-body-side switching.

In total **216 deterministic synthetic cases** are
generated, using the existing production `RepCycleCounter`
and `measureCycleAngle` functions. Expected case results
are frozen independent of the counter's predictions.
If any case disagrees with its known artificial count,
the command exits nonzero so GitHub Actions fails.

The automated evaluator does **not** train a model,
download source videos, open camera, use the original
M20 TRAIN/VAL/TEST, change numeric M34 thresholds,
or make remote API calls. It creates its own synthetic
33-point 2D landmark arrays in memory only.
No user recordings, personal identities, CSVs,
paid hosting or manually entered results are needed.

## One-command validation on Ubuntu

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
cd web
npm install --no-audit --no-fund
npm run validate:m36
```

Optional full frontend test:

```bash
npm test
npm run build
```

The local command writes an ignored, non-sensitive report:

```text
web/reports/m36_synthetic_cycle_audit.json
```

It shows number of artificial scenarios, known expected cycles,
cycles actually observed by the M34 algorithm and synthetic
count error per scenario/exercise. All **216 expected matches**
should be verified before proceeding.

GitHub Actions also runs the same command **automatically**
on every PR/push to main. The user does not need to
execute `init` or `audit` from the old manual M36
Python workflow.

## What 216/216 synthetic results CAN and CANNOT mean

**Can show:** no regression in M34 state transitions when
given these particular generated landmark trajectories
and scripted visibility/timing perturbations. This is
a deterministic unit/integration contract, useful for
preventing broken counting logic and checking privacy.

**Cannot show:** that M34 is actually 100% accurate at
counting real-human repetitions, that real front/side/
diagonal camera views work equally well, or that
any completed cycle represents good form. No human
repetition ground truth, pose-estimation domain shift,
real-world clothing/occlusion or body-size distributions
are included. In-plane camera rotations and x-axis
compression are *not* actual 3D camera-view variations.

Do **not** market this automated synthetic suite as
rep-level precision/recall/F1, user-generalization,
validation on independent adults or safe technique advice.

## Route to real-world validation without user labeling

A feasible future step is to evaluate offline on
an independently collected **public dataset with existing
rep timestamps**, such as the RepCount dataset from
Gao's Lab:

https://svip-lab.github.io/dataset/RepCount_dataset.html

It includes Squat and annotated cycle locations. Before
any automated external video evaluation, confirm
official access/terms, availability of paired videos
and timestamps, action-label compatibility, and
pose-extraction/camera-plane parity. RepCount's Squat
videos may be relevant; there is **not yet evidence
of matching original Barbell Row and OHP rep annotations**,
and it would be misleading to report those classes as
externally validated on RepCount.

An external dataset pipeline should:
1. Fetch or read *only officially available, permitted
   source files* without publishing copyrighted media.
2. Parse existing ground-truth start/end rep timestamps.
3. Extract camera-plane landmarks with a documented
   version of the pose estimator.
4. Replay the unchanged M34 counter on chronological
   landmarks at equivalent sampling rate and compare
   event times with a predeclared matching tolerance.
5. Report precision/recall/F1, counting error, view
   coverage and dropout conditional on source quality;
   keep all video-level artifacts private and do not
   use any previously reserved M20 TEST.

Until this real-data evaluation is implemented and
executed, keep the project labeled **research preview**,
not "validated gym coach".

## Next implementation priority

After this fully automatic M36 gate, prioritize either
**M37 public-annotated-data evaluation** (no owner
manual annotation) or improving the website's actual
user experience with evidence-qualified exercise feedback.
If validating rep events, ensure timestamp annotations
exist and they refer to exactly the same rendered source
frames/clip windows; never invent timestamps from video labels.

The owner's earlier report that M34 counting "works fairly
well" remains an informal qualitative observation only.
