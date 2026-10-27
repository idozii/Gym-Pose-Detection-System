# M20 - Train NEW Gym7 exercise-quality baselines locally

## Scope - current product vs research goal

The existing public site first performs **MediaPipe pose tracking** in
the browser. Once the user explicitly opts in to upload 33 numeric landmarks,
its separate Render Free API runs **six actual owner-trained Random Forest
models** for Squat, Overhead Press and Barbell Row. Those scores reflect
association with their original Fitness-AQA publisher named-error labels.
They are **not validated exercise-form correctness**, injury safety
assessments, nor general-quality diagnoses.

The scientific product goal is a *multi-exercise form quality research
workbench*. New exercises require separate participant-independent
training, target semantics, temporal windows and cross-MediaPipe
validation. Do not describe source scores as proof of correct gym form.

## First new dataset

**A Multi-View Raw Video Dataset of Seven Fitness Exercises with
Good/Bad Form Labels** (v3, published 10 August 2026),
[DOI 10.17632/kgbb3yn47p.3](https://data.mendeley.com/datasets/kgbb3yn47p/3).
Seven exercises, 26 participants, front / side / diagonal mobile videos,
CC BY 4.0. It offers three ZIP files of approximately 2.9 GB each,
but **all three contain the same videos arranged differently**.
Choose exactly **Dataset Exercise-wise.zip** (ONE archive). This does
not contain named knee / elbow / lumbar error labels.

**M20 initial experiments:** Push-up and alternating Bicep Curl.
Later: dumbbell overhead triceps extension, standing side bend,
and independent experiments on existing three movements.
M20 labels are `good` and `bad` **per original video** only,
not per-frame named anatomical errors.

## Prerequisites on the user's OWN Linux computer

No tool inside this chat can remotely write to `~/Desktop` or operate
its webcam or GPU. Code below runs **locally** when the owner executes it.
Only generated source code, docs and synthetic tests are pushed to GitHub.
Videos and all binary models remain local and Git-ignored.

Use a *dedicated virtual environment*, not the existing shared
`/home/idozii/envs/ds`, since it previously reported Hugging Face
and Streamlit dependency conflicts.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
python3 -m venv .venv-m20
source .venv-m20/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,camera]'
python -m pip check
```

If venv is unavailable: `sudo apt install python3-venv`.
MediaPipe 0.10.14 includes the original Solutions Pose API used for
the source extractor; do not silently replace it with Tasks Vision
without comparing extraction distributions.

## Step 1 - download ONE ZIP (owner-opt-in, no automatic cloud copy)

Publisher data is approximately 2.9 GB compressed. Allow more space for
temporary clip extraction. This downloader uses the official public
Mendeley metadata and file download endpoint, requires acceptance of
the CC BY 4.0 conditions and refuses to overwrite any existing file.

```bash
python -m scripts.download_m20_gym7 --accept-cc-by-4-0
```

Expected saved path:

```text
data/external/mendeley_gym7/Dataset Exercise-wise.zip
```

If Mendeley API blocks scripted downloads or times out, **download only**
the file named `Dataset Exercise-wise.zip` from the
[publisher webpage](https://data.mendeley.com/datasets/kgbb3yn47p/3)
using your browser and move it to the exact local path:

```bash
mkdir -p data/external/mendeley_gym7
mv ~/Downloads/'Dataset Exercise-wise.zip' data/external/mendeley_gym7/
```

Do not download `Dataset Subject-wise.zip` or the
`Dataset Exercise Quality-wise.zip` simultaneously - they are
reorganizations of identical videos. Do not commit ZIPs/videos to Git.

## Step 2 - scan video names into a strict subject manifest

```bash
python -m scripts.m20_gym7_pipeline scan \
  --zip 'data/external/mendeley_gym7/Dataset Exercise-wise.zip'
```

The scanner writes an editable CSV:
`data/external/mendeley_gym7/m20_manifest.csv`, containing only
Push-up and Bicep Curl video paths and their inferred original label,
view and subject. Because the exact filename/dir conventions have NOT
been inspected on the owner's archive, **you MUST verify the whole CSV**
against the publisher README before continuing. Unknown metadata
fields are intentionally left empty and extraction will refuse to guess.
A subject ID must remain **identical across all views and both exercises**;
the same person must never receive two IDs.

Use any CSV editor. Required columns:

```text
video_id,archive_member,subject_id,exercise,label,view
```

Allowed exercises `pushup`, `bicep_curl`; labels `good`, `bad`;
views `front`, `side`, `diagonal`. Source `video_id` and
`archive_member` must not be altered or duplicated.

## Step 3 - extract pose vectors, ONE ZIP member at a time

Start with a small **pilot** to estimate pose coverage. Its output is
not a full model release.

```bash
python -m scripts.m20_gym7_pipeline extract \
  --zip 'data/external/mendeley_gym7/Dataset Exercise-wise.zip' \
  --output data/features/m20_gym7/pilot_clip_features.csv \
  --limit 20
```

Then run full extraction in a different output path:

```bash
python -m scripts.m20_gym7_pipeline extract \
  --zip 'data/external/mendeley_gym7/Dataset Exercise-wise.zip' \
  --output data/features/m20_gym7/m20_clip_features.csv \
  --sample-fps 2 --max-frames 240
```

The ZIP is not extracted wholesale: each video is temporarily decoded
using MediaPipe Solutions Pose, sampled up to 240 frames at 2 FPS and
discarded immediately. Each accepted *video* creates ONE 264-feature
example (132 normalized landmark+visibility values by median and std).
Both label classes remain exactly the original `good/bad`, not synthetic
pose-correctness or named error labels. Low visibility poses are dropped,
and you must inspect the rejection rate by class and camera angle.

## Step 4 - TRAIN + validation only (TEST remains locked)

```bash
python -m scripts.m20_gym7_pipeline train \
  --features data/features/m20_gym7/m20_clip_features.csv \
  --output-dir artifacts/m20_gym7/pilot_run_01 \
  --seed 42 --trees 300
```

The trainer uses **one global SUBJECT-disjoint split** shared by both
exercises: roughly 60% subjects for TRAIN, 20% for validation, 20%
reserved TEST. It rejects splits lacking either class in any fold.
Random Forest is a first baseline, not an optimized neural system.

The report includes validation balanced accuracy, bad-class precision,
recall, F1 and average precision, view-wise diagnostics and confusion
matrix. **It deliberately does not read or evaluate TEST by default.**
Clip aggregation avoids treating video frames as independent samples,
although weak labels, repeated attempts, person diversity and camera
domain shifts remain limitations.

Outputs inside ignored `artifacts/m20_gym7/pilot_run_01`:

```text
pushup_clip_quality_rf.joblib
bicep_curl_clip_quality_rf.joblib
m20_research_report.json
```

The owner can later run a ONE-TIME final test in a NEW output directory
after methodology and thresholds are frozen:

```bash
python -m scripts.m20_gym7_pipeline train \
  --output-dir artifacts/m20_gym7/final_once \
  --evaluate-test
```

Treat this as final publication analysis, NOT iterative validation.
Do not upload or deserialize untrusted joblib from the Internet.

## Why these new files are NOT live models yet

A clip-level good/bad model consumes **a window of frames** and emits
one quality label for a whole video. The current Vercel API instead
accepts a **single frame's 33 landmarks** for specific original
Fitness-AQA source annotation targets. These contracts do not match.
Do not add Push-up or Curl selectors to the production inference UI
until new **streaming temporal windows**, participant holdout quality
and browser-vs-Python MediaPipe feature parity are implemented and tested.

To move toward a research paper, report across-person metrics,
performance broken down by angle, rejection-rate bias and multiple
seeds, then acquire consented, independent phone recordings for
out-of-distribution evaluation. A high validation score alone is not a
scientifically validated form-correction product.

## Next research tier

- **EgoExo-Fitness (ECCV 2024)** has temporal segments, technical
  keypoint checks and quality scores. Requires signing the publisher
  license and requesting gated access; the current HF listing notes
  that **raw video is not part of its downloadable release**.
  [Official data terms](https://github.com/iSEE-Laboratory/EgoExo-Fitness).
- **MM-Fit** may support exercise recognition and rep-counting,
  but its labels are not named gym form errors.
- **UI-PRMD** supports correct/incorrect rehabilitation exercise
  movements with different skeletal sensor structure.

Data sources should not be merged unless label semantics, legal terms
and subject splits are compatible.
