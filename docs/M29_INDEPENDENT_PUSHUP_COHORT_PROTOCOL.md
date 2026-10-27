# M29 - Prepare an INDEPENDENT Push-up validation cohort

## What the owner's M28 results actually support

M28 re-ran M27 with perfect 5-fold metric reproducibility on 83 Push-up
TRAIN videos from 14 original TRAIN participants, without viewing
original M20 VAL/TEST.

| Model | GROUP5 BA | LOSO BA | LOSO bad recall |
| --- | ---: | ---: | ---: |
| baseline_64 | 54.18% | 56.56% | 51.22% |
| kinematic_76 | 58.97% | 60.13% | 51.22% |
| quality_proxy_68 | 60.22% | 54.18% | 51.22% |
| temporal_80 | 58.91% | 62.51% | 51.22% |

Paired original TRAIN person-bootstrap BA deltas relative to baseline:

| Model | GROUP5 interval, percentage points | LOSO interval, percentage points |
| --- | --- | --- |
| kinematic_76 | [-1.19, +10.80] | [-6.07, +13.47] |
| quality_proxy_68 | [-1.22, +13.10] | [-11.90, +7.40] |
| temporal_80 | [-2.61, +10.80] | [-2.62, +14.29] |

**All intervals include zero.** With just 14 TRAIN participants,
neither superiority nor real-world generalization has been established.

`quality_proxy_68` notably reversed ranking between GROUP5
and LOSO, which may reflect sensitivity to participant
composition and/or reliance on camera/pose-quality metadata.
It is a hypothesis, not a proven confounding mechanism.
Kinematic features show more consistent *direction* of BA gains
across the two strategies, while Temporal 80 gives the highest
LOSO BA and AP. **No model improved LOSO bad recall over baseline
at the fixed 0.5 threshold.**

Most importantly, the owner already inspected original M20 VAL
during earlier iterations. More tuning on that VAL would not
produce credible fresh confirmation. The original TEST remains
sealed, and there is no research obligation to open it now.

## Milestone decision

M29 is a **preparation/ethics/data-quality milestone**,
not another hyperparameter search or deployment.

We freeze the [prospective comparison contract](../configs/m29_pushup_external_validation_frozen.json)
before collecting or looking at independent labels:

- `baseline_64` - fixed 64-feature M23 reference
- `kinematic_76` - primary interpretable motion hypothesis
- `temporal_80` - secondary complete temporal variant

The feature extractor remains the existing upper-body
MediaPipe Solutions Pose pipeline (complexity 1, confidence 0.5),
2 FPS, maximum 240 sampled frames, four accepted frames per clip.
RF hyperparameters stay 200 trees, min leaf 2, class weight balanced,
seed 42 and threshold 0.5. Report the same-video-cohort
balanced accuracy as primary, plus bad recall, F1, AP, coverage
and viewpoint breakdowns; no tuning after external outcomes.

The dataset's labels are **video-level "good/bad"** according to a
written external annotation rubric, not specific elbow/knee/lumbar
error labels. Human disagreement should be reported.

## New source participants and privacy

Recruit **genuinely new consenting adults** who are known not to be
the people in Gym7 or any previously used training/validation dataset.
Pseudonymous ID differences alone cannot establish this: keep
a separate *offline*, access-controlled recruitment record and
document how subject independence was established. The script
deliberately reports this confirmation as **FALSE** until
manually substantiated.

Record only voluntarily performed exercises. Do not instruct
someone to adopt intentionally unsafe Push-up technique just to
obtain `bad` labels. Use independent trained reviewers assessing
naturally recorded movement, or a separately licensed and
independently sourced set of pre-annotated videos. Obtain consent
for research use, processing, retention and any sharing.
Ensure participants can withdraw within the study's stated
data-retention/withdrawal window. Keep contact information,
signed consent forms and non-pseudonymized original video
**out of GitHub and the public website**.

An illustrative **pilot** can start with >=5 distinct adult
participants, at least one `good` and one `bad` label in
each of front/side/diagonal camera views. A more ambitious
planning target is >=20 separate adults, >=120 distinct
video clips and >=10 per view/label combination. These are
logistical targets, not claims of statistical power. If an
ethical collection cannot meet them, report only a small
descriptive pilot, not a validated AI coach.

Two independent pseudonymous reviewers label each whole video.
When they disagree, a third distinct independent reviewer
must adjudicate. Keep the written labeling rubric stable.
Avoid using the model's own predictions to decide ground-truth
labels. `reviewer_a_id`, `reviewer_b_id`, `adjudicator_id`
and `consent_ref` must be non-identifying codes only.

## Commands on Ubuntu

Use your already isolated M20 virtual environment. No dataset
is downloaded, created or collected automatically.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

# Create a BLANK LOCAL template and edit it only when you have consented data.
python -m scripts.m29_external_pushup_cohort init
```

The CSV header is stored at:

```text
data/external/m29_pushup/manifest.csv
```

Store privately consented clips beneath:

```text
data/external/m29_pushup/videos/
```

For each clip fill these exact manifest columns:

```text
video_id,subject_id,view,source_file,reviewer_a_id,reviewer_a_label,reviewer_b_id,reviewer_b_label,final_label,adjudicator_id,consent_ref,age_18plus_verified,annotation_version
```

Use `video_id` starting with `extclip_` and
`subject_id` starting with `extperson_`, with at least
eight additional alphanumeric/underscore/hyphen characters.
Reuse a person's **same subject ID** across views/sessions.
Allowed labels are `good` and `bad`, views are
`front`, `side` and `diagonal`, adults must use
`age_18plus_verified=yes`, and `annotation_version`
must equal `gym7-external-pushup-v1`.

The source file is a **relative path** beneath the local
private videos directory, not a URL or absolute path. Do not
commit manifests or videos. Annotators should label without
seeing inference scores.

After independently collecting and reviewing the data:

```bash
python -m scripts.m29_external_pushup_cohort audit
```

The validator checks consent-reference presence, adult verification,
two independent human labelers, adjudication for disagreements,
safe paths, byte-distinct source files and aggregate view/label
balance. It writes only anonymous aggregate counts to
`data/processed/m29_external_pushup_source_audit.json`.
It **does not open original M20 TEST**, decode pose, train RFs,
measure accuracy, or upload a source video.

A successful audit does **not prove scientific ground-truth
correctness** or real-person disjointness: these require human
oversight and protocol documentation. A new model evaluator
should be added only after the independent cohort exists
and is qualified, with source/feature parity and consent
reviewed first.

## What not to do

Do not silently add new videos to the M20 TRAIN split, refit
existing models using original M20 VAL, peek at original TEST,
resample until the model wins or publish any Push-up form-safety
score on the live Vercel site. Current production remains
the six original Fitness-AQA source-label research models.
