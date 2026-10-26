# M17B — final source-integrity gate and research-demo handoff

## Confirmed software milestone

The owner's local M17A run from updated GitHub source returned
**218 passed tests, four existing sklearn single-label fixture warnings**.
All **six** owner-created local Squat/OHP/Barbell Row source-label
Random Forest artifacts were successfully loaded with explicit
`--trust-local-artifacts`, no publisher TEST interaction in that
registry check. PR #33 is merged into `main`.

Historic OHP PR #18 (**4 changed files**) and PR #19 (**6 changed
files**) were compared with `main` **file by file**: all **10 files
were byte-for-byte identical** to the integrated content. Both
outdated PRs were closed as *superseded* with explanatory comments.
No code was lost and no duplicate OHP feature module was merged.

Owner M16D camera evidence: 180-frame Ubuntu Squat demo, six
locally trusted RF models available, 180/180 detected poses, 360
successful Squat task scores (two per frame), 0 abstentions,
**11.16 ms** mean per successfully scored source-error task,
**15.78 observed GUI FPS**. This measures local runtime operation,
**NOT** validated recognition of real technique quality.

## New read-only audit

```bash
python -m scripts.audit_research_release
```

It returns JSON with `status=research_software_release_audit_PASS`
when the following **source-code** conditions hold:

1. Exactly six named original publisher source-error model contracts:
   two Squat with 103 features, two OHP with 30 each, and two
   Barbell Row with 20 each. Version, feature ordering, and output
   paths are checked from source definitions **without unpickling**.
2. Original TRAIN+VAL reproduction sources exist for Squat, OHP and
   Row, including the previously unmerged OHP extraction, benchmark
   and seed-stability analyses; the research report and LICENSE exist.
3. README clearly labels the prototype as **research-only** and
   the research report discloses the historical publisher TEST pilot
   exposure and subject-independence limitations.
4. No NEW private media, Fitness-AQA archives, train/validation feature
   CSVs, or new joblib/PT binaries are tracked by Git. Audit uses
   `git ls-files` and **does not traverse or read ignored local
   data/features, publisher TEST clips, or private camera images**.

The two pre-existing tracked **historical** binary files
`models/squat_form_rf.joblib` and `models/yolov8s.pt` are reported
as warnings, *not automatically removed*, to preserve notebook
compatibility. They are NOT the six versioned and provenance-attested
experimental source-error models. Their license/provenance and the
pickle security implications should be reviewed before a separately
distributed binary release. A passing audit must not be described as
“no model binaries were ever committed.”

The new audit becomes part of each Python 3.10 and 3.11 CI run
immediately after `pytest`. Synthetic tests verify that new
source media/derived CSVs trigger a source release BLOCK, while
historical tracked legacy models are recorded honestly and ignored
local data/pickles are never read.

## Commands for local owner acceptance

```bash
git fetch origin
git switch feat/m17b-release-audit-protocol 2>/dev/null || \
  git switch --track origin/feat/m17b-release-audit-protocol
git pull --ff-only

python -m pytest -q

# Read-only and SAFE: does NOT load .joblib, open webcam or read TEST:
python -m scripts.audit_research_release

# Separate optional step: ONLY your own trusted local model pickles:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts

git status --short
```

Do not discard the user's currently modified notebook
`notebooks/gym_pose_detection.ipynb`; switching Git branches
normally keeps those changes. Avoid `git clean -fdx` and do not
commit original datasets/models while preparing a public demo.

## Release decisions and limitations

This audit is a **release code hygiene gate only**. It does not
establish fresh holdout accuracy, generalization to unseen athletes,
medical/injury prevention validity, independently reproducible
publisher dataset rights, or review of legacy YOLO weight licensing.
The old M1 documented pilot did process some publisher TEST videos;
hence no whole-history test-pristine claim.

The original source validation sets were repeatedly examined, so
their reported AP/F1 scores are exploratory, and some tasks have
high false-positive or false-negative rates. No model thresholds,
pose visibility gates or research data changed in M17B.

The only real webcam performance supplied is **Squat** on the owner's
computer; **OHP and Barbell Row have not yet been live-smoke-tested by
the owner**, and all camera viewpoint changes are unmeasured domain
shift. They must not be described as production-ready.

**Recommended publishable scope:** project source with a versioned
*experimental local research viewer*, six optional owner-trained
source-label models, methods documentation, source validation
limitations, frame-level performance profiling and explicit
ABSTAIN. Do NOT release a production “AI Correct Form/Safe Lifting
Coach,” confidently tell people how to fix movement, or advertise
real-world error accuracy without new independent data.

A GitHub version tag, prebuilt executable or public GitHub Release
has **NOT** been created by this milestone. First check this audit,
then decide whether to publish a research-only release candidate.
