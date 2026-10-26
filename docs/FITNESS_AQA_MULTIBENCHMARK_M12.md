# Milestone 12 — Fitness-AQA Multi-Exercise Dataset Foundation

Scope: **dataset integrity and reusable metadata adapters**, not model training.
The same research repo will cover Squat, Overhead Press (OHP), and Barbell Row
without confusing their annotation formats or claiming subject independence.

Source: https://drive.google.com/drive/folders/16JWrJB-N7noUYNdZsZ_76suU_qArzzRn
Parmar, Gharat, Rhodin (2022), "Domain Knowledge-Informed Self-Supervised
Representations for Workout Form Assessment", arXiv:2202.14019.
Respect the original dataset terms and retain credit.

## Actual publisher data formats confirmed

| Exercise | Source medium | Label semantics | Original split |
| --- | --- | --- | --- |
| Squat | Video ZIP | `error_knees_forward`, `error_knees_inward` = time intervals (seconds) | train/val/test **video IDs** |
| OHP | Video ZIP | `error_elbows`, `error_knees` = time intervals (seconds) | train/val/test **video IDs** |
| Barbell Row | `barbellrow_images_raw.zip` | `labels_torso_angle_error`, `labels_lumbar_error` = **per-image 0/1** | separate train/val/test **image IDs for each target** |

OHP's two label dictionaries map 2,260 video IDs to interval lists;
`error_elbows` has 576 videos containing >=1 annotated error interval
and `error_knees` has 777. These are dictionary statistics, not
accuracy scores or training split counts.

Barbell Row's torso label dictionary has 17,440 image keys and lumbar
has 14,978 image keys. Keys resemble `56067_3_51`. There are two
distinct task split trees; **do not merge them** into a single train/val/test
without checking cross-task conflicts and source-clip provenance. The final
number of images matching the archive and publisher splits must be
verified locally; labels count alone does not prove ZIP completeness.

## 1. Download OHP annotations, splits and labeled video archive

From repository root, install `gdown` separately if needed:

```bash
python -m pip install gdown

OHP=data/external/fitness_aqa/ohp/labeled
mkdir -p "$OHP/annotations"

# Source split keys: train/val/test (test IDs for audit only)
gdown 1t3x3gPBVfpRBXa5-IUotb04thoOn2MoA -O "$OHP/annotations/train_keys.json"
gdown 1NgOP2rhSaTbeDtbPHA8JjN-Ln3m0g1s0 -O "$OHP/annotations/val_keys.json"
gdown 1nQVzdxMHTI3yDsmeS0oiRwVfzvIguMip -O "$OHP/annotations/test_keys.json"

# Two separately annotated named errors
gdown 1Q4uMJF0Tk2Dbv9A5o_CgXQHF2fBh-6Go -O "$OHP/annotations/error_elbows.json"
gdown 1thvXSXfp9YTKjIyUbjbyj4oSOgfcb9zu -O "$OHP/annotations/error_knees.json"

# Labeled video ZIP (~907 MB), only if not already downloaded
gdown 1bcIZZPq6O5JdGtwba1Bs7w1EJPqTERp3 -O "$OHP/videos.zip"
```

Before downloading the large archive, you can validate the five JSONs:

```bash
python -m scripts.inspect_fitness_aqa_multi --exercise ohp --skip-archive
```

## 2. Download Barbell Row annotations, **both** split schemes and images

```bash
ROW=data/external/fitness_aqa/barbell_row/labeled
mkdir -p "$ROW/annotations/splits/torso_angle"
mkdir -p "$ROW/annotations/splits/lumbar"

# Two independent binary image label dictionaries
gdown 1-RRxfr-h1hMql5QADf30dVWe5RM_QmDB -O "$ROW/annotations/labels_torso_angle_error.json"
gdown 1tn0gSRnfD24DZnCrDy5d0tf5rIG9VaBg -O "$ROW/annotations/labels_lumbar_error.json"

# Publisher torso-angle error image splits
gdown 1RdEBjwiZ8-TcidLqNT5LHWCF0eHzC78d -O "$ROW/annotations/splits/torso_angle/train_ids.json"
gdown 110W1r_f21yNILQ1ZfdUpMVJybmOY0Qhb -O "$ROW/annotations/splits/torso_angle/val_ids.json"
gdown 1YuVfLYPtmjyv6XQ-LFTZfbKTZwSJa25r -O "$ROW/annotations/splits/torso_angle/test_ids.json"

# Publisher lumbar error image splits (may differ; MUST NOT be overwritten)
gdown 1qvZPuMtgAblOb0BYrs04MCIA1dRiyTcK -O "$ROW/annotations/splits/lumbar/train_ids.json"
gdown 1sYFv1WU1NtL7j4OlAuvtsNCFnasjxFJA -O "$ROW/annotations/splits/lumbar/val_ids.json"
gdown 190BTbZvedsjRK3_fwn6qC4WY7cOE1lzd -O "$ROW/annotations/splits/lumbar/test_ids.json"

# Labeled images ZIP (~550 MB), not the OHP/Squat video ZIP
gdown 1hhKohLQ85H_zNTLHuWlCuIW19XK7oY1E -O "$ROW/images.zip"
```

As with OHP, check metadata before fetching images:

```bash
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row --skip-archive
```

If `gdown` fails due to Google Drive rate limits or permissions, use the
publisher's original folder links to download manually, preserving the
folder names shown here. Do not invent or rename source JSON formats.

## 3. Validate archives via ZIP central-directory metadata (no unzip)

```bash
# Inspect OHP by itself; does not open Squat or Row.
python -m scripts.inspect_fitness_aqa_multi --exercise ohp

# Inspect Barbell Row independently, preserving its two label-specific splits.
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row

# Run all three source inspections when *all three* datasets exist locally:
python -m scripts.inspect_fitness_aqa_multi --exercise all
```

Generated local, Git-ignored report:

```text
artifacts/fitness_aqa_dataset/dataset_manifest.json
```

The manifest reports:

- Source split counts, source-ID uniqueness and disjointness **within each task**.
- OHP and Squat named-error interval validity and positive annotation counts.
- Barbell Row positive/negative *image* labels, with independent split counts.
- Exact media filename-stem coverage in ZIP: missing/extra/duplicate IDs and
  summary file paths, with archive uncompressed byte totals.
- **Cross-task Barbell Row ID split conflicts** and a conservative warning
  about candidate clip-prefix overlap (prefix is unverified, NOT subject ID).
- Per-task readiness for further extraction; a separate flag marks whether
  naive unified multi-task Barbell Row training appears safe at ID/prefix level.
  Even a true flag **does not** prove independent athletes, near-duplicate
  isolation, or model reliability.

For incomplete ZIP coverage or duplicate source stems, command returns exit
code `2`. Bad schema/missing JSON returns `1`; valid metadata-only audit
returns `0` but **does not** mark the archive as checked.

Test **IDs** are inspected for split leakage and ZIP completeness, but NO
test media is opened, and no test labels are fed to a model. In particular,
this milestone MUST NOT report AP, accuracy, or F1 for OHP/Barbell Row.

## 4. Next milestones (not bundled here)

- M13: OHP video adapter + pose extraction; adapt feature schema to overhead
  pressing, train task-specific elbow/knee temporal models on original
  publisher train/val split. Avoid reusing Squat's trained head.
- M14: Barbell Row *image* adapter; per-task image labels and original
  splits; start static pose/image RF benchmarks. Derive temporal sequences
  only **after** original clip ordering is proven.
- M15: Cross-exercise registry and standardized evaluation with validity
  caveats. Any unified model training requires a split-provenance plan.

**All six target zero labels** mean only absence of a named annotated
error in that task, never medically certified correct form.

## Research hygiene

- Do not commit ZIPs, extracted frames, model outputs or private footage.
  Both `data/external/` and `artifacts/` are ignored in `.gitignore`.
- Keep publisher test set locked after this metadata/ZIP-integrity audit.
- Do not invent `subject_id`, `session_id`, video IDs for Barbell images,
  or treat image filename prefixes as independently verified athlete IDs.
