# Milestone 14B prerequisite — Barbell Row pose coverage failure audit

## User's real source dataset results (M14A / PR #21)

Barbell Row images have been decoded successfully from the publisher
Fitness-AQA **image** archive. The original source TRAIN and VALIDATION
task-specific splits were respected; no test pixels were accessed.
117 unit tests passed, image decoding failed **0 times**. However,
the strict 32-feature MediaPipe lower-body gate accepted an extremely
small fraction of images.

| Target | Source split | Selected source images | Pose accepted | Coverage | Positive accepted / Positive selected |
| --- | --- | ---: | ---: | ---: | ---: |
| Torso-angle error | Train | 11,881 | 483 | **4.07%** | 59 / 1,055 |
| Torso-angle error | Validation | 2,564 | 109 | **4.25%** | 20 / 278 |
| Lumbar error | Train | 10,360 | 437 | **4.22%** | 28 / 1,580 |
| Lumbar error | Validation | 2,227 | 108 | **4.85%** | **8 / 347** |

Across all source TRAIN/VAL unique images, MediaPipe returned
**no pose** for 3,824 / 12,729 train and 813 / 2,736 validation.
Many images received a pose but then failed the current 8-joint
visibility gate or failed geometry. These numbers alone do NOT
establish which particular joint causes most losses.

Only **8** retained lumbar-positive validation images are available,
and accepted positives are a much lower fraction of the original
lumbar-positive class. Do NOT train or publish Barbell Row RF
performance from these accepted-only CSVs; doing so without
accounting for abstention would be substantially selection-biased.

The M12 manifest also listed **15,558** extra ZIP image stems outside
the publisher's labeled task-split union. These are intentionally
excluded, not source decoding failures.

## New diagnostic: source train/validation annotation-stratified audit

The new script samples up to **180 original image IDs per task/split**,
label-stratified between positive and negative error annotations.
It uses original publisher train and val lists for EACH task,
reads a shared source image only once if selected for both tasks
on the same split, and never opens publisher TEST image pixels.

```bash
git fetch origin
git switch feat/m14-barbell-row-pose-coverage-audit 2>/dev/null || \
  git switch --track origin/feat/m14-barbell-row-pose-coverage-audit
git pull --ff-only
python -m pytest -q

python -m scripts.audit_fitness_aqa_barbell_row_coverage \
  --images-per-task-split 180 \
  --seed 42
```

**No dataset download or new full extraction required**. The script
reads the already-available `images.zip` and annotation JSONs, and
writes a new Git-ignored JSON (no photos, extracted frames, altered CSVs
or model files):

```text
artifacts/fitness_aqa_barbell_row/pose_coverage_audit.json
```

This is a deliberately label-stratified diagnostic, so its raw
acceptance rates are **not population pose coverage estimates**.
The full source extraction above remains the population reference.

## Policies tested on *identical* sampled images

| Candidate gate | Required joints | Minimum visibility | Is its feature schema already implemented? |
| --- | --- | --- | --- |
| `all8_050_production` | shoulders, hips, knees, ankles | 0.50 | YES — original 32 features |
| `all8_035` | Same eight | 0.35 | Could reuse 32 features, but reliability unverified |
| `all8_025` | Same eight | 0.25 | Higher pose-error risk |
| `six_no_ankles_050` | shoulders, hips, knees | 0.50 | NO — would require separate new features |
| `torso4_050` | shoulders, hips only | 0.50 | NO — needs new feature schema |
| `torso4_035` | shoulders, hips only | 0.35 | NO — needs new feature schema |

Each policy reports source-label positive and negative acceptance
separately, and how many new positives/negatives it hypothetically
admits compared with the strict 8-joint production gate.
The report also separates **no pose**, **invalid pose/geometry**,
and **low joint visibility**, and tallies failing joints. It does
not train, validate or compare models, nor does it change the
production feature schema or extracted source CSVs.

The four-joint candidate is particularly useful for checking whether
the current Row project rejects frames because **ankles/knees are
missing in cropped upper-body images**. The lumbar label is
especially problematic: MediaPipe's shoulder/hip/knee/ankle skeleton
does not directly recover **lumbar-spine curvature**. Even a
four-joint model would be only a source-label classification
surrogate, not clinical assessment of the spine.

## Decision after the audit

1. If **pose absent** dominates, examine source image scale,
   crop/aspect, and alternate pose detectors before RF.
2. If **knees/ankles low visibility** dominate and torso4 policy
   preserves source error-positive samples, implement a separate
   shoulder+hip-only feature schema and re-extract source train/val.
   Do **not** simply reduce quality thresholds while keeping
   unreliable knee/ankle coordinates in the model.
3. If all8 at 0.35 helps on *source positive images*, assess
   feature reliability and viewpoint issues before modifying
   the quality policy globally.
4. Only then create a separate `torso_angle` and `lumbar` static
   classifier with source-specific train/val splitting, uncertainty
   and honest pose coverage.
5. Do not claim clip- or subject-independent evaluation from
   publisher image IDs or the unverified `image_id.rsplit("_",1)[0]`
   prefix heuristic. Never access TEST image contents for
   model design, threshold selection or pre-publication evaluation.

This audit is not evidence that the assistant can inspect
actual camera viewpoints or individual source photos in the
user's local ZIP: those pixels remain on the owner's machine.
