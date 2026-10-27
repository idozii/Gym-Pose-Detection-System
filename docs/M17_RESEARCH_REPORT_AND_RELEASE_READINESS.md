# Milestone 17A - Research Report and Release Candidate Readiness

**Status:** *experimental research prototype*, **NOT** a validated
fitness-safety product, clinical device, or reproducible academic
publication. This report separates confirmed local runs from
unverified model accuracy and release work still pending.

## Research question and workflow

Can a single privacy-conscious local webcam viewer run **six
exercise-specific publisher error-label detectors** from three
Fitness-AQA subsets while respecting source-specific task labels,
pose admission rules and strict training/validation separation?

Pipeline: **local webcam → MediaPipe pose → versioned per-task
features → trusted local Random Forest → source-label research scores
or ABSTAIN → optional framing/aggregate FPS diagnostics**.

| Exercise | Original source labels | Original source unit | Current live STATIC feature schema |
| --- | --- | --- | --- |
| Squat | `error_knees_forward`, `error_knees_inward` | Publisher video-time error intervals | `squat-103-v1` (103 features) |
| Overhead Press | `error_elbows`, `error_knees` | Publisher video-time error intervals | `ohp-elbows-upper8-30-v1`, `ohp-knees-lower8-30-v1` (30 each) |
| Barbell Row | `torso_angle`, `lumbar` | Independently publisher-split original image binary labels | `barbell-row-torso4-image-20-v1` (20 each) |

Original error-label 0 means **absence of that particular named
annotation**, not globally correct technique. Barbell Row's
shoulder/hip landmarks **cannot measure lumbar spinal curvature**.

## Evidence - original publisher validation, exploratory only

The owner ran these training/validation scripts locally with full
source features. Numbers are **not independent-person performance**
and not prospective live-webcam error-detection accuracy.

| Original source target | Pose coverage (train/val) | Validation AP | AP prevalence reference | Macro-F1 | Error precision | Error recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Squat - knees forward | 50.21% / 52.12% | .3769 | .2175 | .6212 | .3676 | .7850 | **.3753** |
| Squat - knees inward | 50.21% / 52.12% | .1600 | .0407 | .5910 | .1961 | **.2500** | .0434 |
| Barbell Row - torso angle | 70.14% / 69.77% | .4460 | .1118 | .6908 | .4408 | .4650 | .0743 |
| Barbell Row - lumbar | 70.20% / 70.86% | .5056 | .1591 | .6900 | .6133 | **.3665** | .0437 |

OHP source task-specific static vs causal temporal RF investigation
(separate train/validation populations per task):
- Elbow static AP **.2858**, causal temporal **.3411**.
- Knee static AP **.1439**, causal temporal **.1957**.
- Five-seed *TRAIN group-calibrated* sensitivity found temporal knee
  AP better in all five runs, but these are **not independent
  independent-trial replications**, and original validation was reused.
- The 6-slot **live model registry uses STATIC variants** for all
  three exercises, NOT the OHP causal temporal experiment. Do not
  report the temporal scores as live model accuracy.
- Source elbow pose acceptance about **31.51%/30.11%** and knee
  about **39.27%/36.97%** train/validation. These are selected-frame
  subsets and may be systematically biased.

No automated warnings are activated because of low precision,
low recall and significant possible false alerts in these
exploratory source-label tasks.

## Owner-verified real Ubuntu webcam pipeline

The owner ran M16D with six locally trained trusted model files:

| Measurement | Observed M16D run |
| --- | ---: |
| Test suite | **201 passed; 4 existing sklearn fixture warnings** |
| Research model contracts | **6/6 loaded** |
| Webcam frames | **180** |
| MediaPipe frames with pose | **180/180** |
| Squat accepted camera frames | **180/180** in this session |
| Squat score calls | **360/360** (180 each named target) |
| Abstentions | **0** in this session |
| MediaPipe mean / p95 | **20.140 / 27.772 ms** |
| RF-forward task mean / p95 | **11.160 / 11.889 ms** |
| Two-target model stage mean / p95 | **22.596 / 23.932 ms** |
| Full-frame displayed mean / p95 | **62.972 / 70.588 ms** |
| Observed camera+GUI FPS | **15.78** |
| Frames recorded / alerts / rep counting | **0 / 0 / disabled** |

For the two Squat tasks, mean successful calls were approximately
**11.409ms** knees-forward and **10.911ms** knees-inward, including
source feature extraction and fitted RF scoring. The observed FPS
includes camera capture, OpenCV GUI and waits; it is NOT a standalone
model-throughput figure. A different camera distance/angle earlier
gave **0/240** task scores because ankles/knees were invisible, and
the intermediate guide run gave **94/360** scores. Thus admission
depends strongly on live framing: a single ideal 180-frame run does
not guarantee robust operation across subjects, positions or lighting.

## Privacy, safety and evaluation protocol

- Viewer runs locally: no recording, webcam uploads, automatic
  technique/medical advice, audio alerts or rep counting. Printed
  diagnostics report aggregate counters and timings rather than
  identifying webcam coordinates or source probabilities.
- Local joblib files are *pickle*: loading them can execute arbitrary
  code. Only explicitly trusted, locally trained model artifacts
  should be deserialized.
- Original publisher TRAIN is used for fitting and classification
  operating-threshold selection. Original publisher VALIDATION was
  consulted repeatedly for exploratory assessment; it cannot now
  be advertised as a pristine untouched model-selection holdout.
- Squat/OHP original source splits are video-ID-disjoint but **not
  verified athlete-independent**. Barbell Row candidate image prefix
  has **not** been verified as independent clip/athlete identity.
- **Historical caveat:** an earlier pilot described in
  `docs/FITNESS_AQA_BASELINE.md` reports processing *eight source
  TEST videos*. Later M13-M16 extraction/modeling code was designed
  to **not** access original publisher TEST, and M16D reported
  `publisher_test_accessed=false`; nevertheless the entire
  repository history must **not** claim that source TEST was
  never examined. This prior exposure is a meaningful reason to
  require a separate frozen, independently collected evaluation set
  for publishable claims.
- All cross-dataset/real-camera distribution shifts are unquantified.
  Raw error probability is a publisher-source classifier output, NOT
  a statistically calibrated probability of injury.
- MediaPipe refusal/ABSTAIN is an explicit operating requirement,
  not an error-free or "safe lifting" judgment.

## M17A research reproducibility / historical PR cleanup

The PR #18 extraction scripts and PR #19 benchmark/stability scripts
had remained unmerged although their OHP feature adapter was later
integrated separately by PR #26. M17A ports **only the remaining
missing** original research scripts, synthetic tests, and methods
documentation from their historic head commits to updated `main`.
It does not overwrite the already integrated adapter or trained
artifacts, does not run a publisher TEST, and should be validated
on CI Python 3.10 + 3.11. Following merged PR #33, every changed file in original PR #18 and
PR #19 was verified **byte-for-byte identical** on `main` (4/4
and 6/6 respectively). Both obsolete PRs were subsequently closed
as superseded with an explanatory comment, preserving provenance.

## Reproduction (locally downloaded Fitness-AQA files only)

```bash
python -m pip install -e '.[dev]'
python -m pytest -q

# Does not load any joblib by default:
python -m scripts.check_fitness_aqa_model_registry

# ONLY your own trusted locally trained research artifacts:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts

# Existing original OHP task-specific pose CSVs and source manifest:
python -m scripts.benchmark_fitness_aqa_ohp_tasks --help
python -m scripts.assess_fitness_aqa_ohp_seed_stability --help

# Local optional camera environment:
python -m pip install -e '.[camera,dev]'
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 --trust-local-artifacts
```

Original Fitness-AQA download rights/attribution and locally
reproducible source files are the user's responsibility. **Never
commit original source media, private webcam images, source CSVs
or additional trained pickle artifacts** into the public repository.
The **six original-source research RF artifacts are not tracked**,
but two older notebook-era binary files ARE tracked in the current
history: `models/squat_form_rf.joblib` and `models/yolov8s.pt`.
They are not the six trusted research models; M17B reports them as
historical release caveats without deleting notebook dependencies.
Never load downloaded joblib/pickle files from untrusted sources.

## Release acceptance and known blockers

| Gate | Status / requirement |
| --- | --- |
| Six correct task/model contracts | **Observed 6/6 on owner's machine** |
| Headless Python matrix tests | **218 passing** in owner's real M17A run; rerun after M17B release-audit additions |
| Live capture with 6 local model slots | Squat-only actual 180-frame proof; OHP/Row **not yet checked live by owner** |
| Per-task source validation transparency | Preliminary research table above; no pristine held-out evaluation |
| Standalone subject-independent accuracy | **NOT established**, cannot claim research validation |
| TEST split clean across entire history | **NOT established**; historical pilot explicitly includes some TEST videos |
| OHP reproduction on `main` | **Source scripts/tests integrated** via merged PR #33; code parity with original PR #18/#19 verified |
| Release source/data hygiene | M17B read-only tracked-data audit available; older notebook binaries are disclosed rather than deleted |
| Privacy and safety scope | Local-only research viewer; no automatic form or injury warnings |
| PyPI/public production fitness-coach release | **NOT ready**, beyond current experimental prototype |

**Proposed release label:** `0.1.0-research-rc` (*candidate;
not a published GitHub Release and not a production fitness product*).
The project can be presented as a **versioned, reproducible
research software prototype** once source integration and CI pass,
with fully disclosed evaluation boundaries. Independently validated
accuracy/injury claims require new external evaluation.
