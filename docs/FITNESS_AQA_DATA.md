# Fitness-AQA: external dataset setup (no video commits)

**Original source:** https://drive.google.com/drive/folders/16JWrJB-N7noUYNdZsZ_76suU_qArzzRn
**Dataset citation:** Parmar, Gharat, Rhodin (2022), *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment*, arXiv:2202.14019. Verify usage rights and retain attribution.

## Actual Drive directory structure inspected
- Squat/Labeled_Dataset/videos.zip (~970,007,911 bytes), file ID `1LvxLhX-Ctljr0xP8WNtHsKvObfISkZ70`
- Squat/Unlabeled_Dataset/videos.zip (~1,675,157,861 bytes), file ID `12SArZjK9i-MAeqnfUD5PREPxiplJAYmk`
- Squat/Labeled_Dataset/Labels/error_knees_forward.json
- Squat/Labeled_Dataset/Labels/error_knees_inward.json
- Squat/Labeled_Dataset/Splits/{train_keys,val_keys,test_keys}.json
- Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images.zip (~104,721,913 bytes), file ID `1kPIQT_D6IwkZI18QJoWCVkjJm9YARaRn`
- Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json and splits/
- OHP and BarbellRow also contain labeled archives but are **not** yet supported by the squat-only classifier.

## Download only the subset you need
On your computer, from the repository root:

```bash
python -m pip install gdown
mkdir -p data/external/fitness_aqa/squat/labeled
gdown 1LvxLhX-Ctljr0xP8WNtHsKvObfISkZ70 \
  -O data/external/fitness_aqa/squat/labeled/videos.zip
```

If quota or Drive permissions block `gdown`, download the archive through the supplied
Drive browser link and place it in the same local directory. Do not commit ZIPs.
Archive names and internal layouts can change: inspect before extraction:
`unzip -l data/external/fitness_aqa/squat/labeled/videos.zip | head -50`.
Extract into the ignored `data/external/` tree only after inspecting.

## Crucial label incompatibility
The supplied JSON annotation README states:
`[Video number]: [error_start_time, error_end_time]` in **seconds**,
with an empty list meaning no annotated interval *for that particular error*.
Knees-forward and knees-inward use distinct annotations. Shallow-depth dataset uses
0 = no error, 1 = erroneous.

These are **not** ready-made per-frame `correct` and `incorrect` labels.
An absence of an annotation for one error does not imply flawless technique.
Preserve each annotation type, source split, and video/interval boundaries.

The existing Milestone 2 manifest requires verified `subject_id` and `session_id`.
The supplied Drive directory and annotation descriptions do not establish unique
identity for each athlete. **Never invent subject IDs** from random frames.
Until actual identities are verified, use the dataset's *provided train/val/test
keys* for a separate source-split experiment, with duplicate/near-duplicate checks,
and report it as a source-split benchmark, not subject-disjoint.

## Git handling
All large media and generated datasets remain ignored via `.gitignore`.
Run `git status --short` and `git ls-files | grep -E '(data/external|\\.zip$|\\.mp4$)'`
before pushing. If a file was already added to the index, untrack it without deleting
the local file: `git rm --cached -- path/to/file`. Ignoring files **does not remove
them from earlier commits**; if giant files are already in Git history, inspect the
history and use a history-rewrite tool only with collaborator coordination.

Keep code, manifests *without private identifying data*, instructions, split definitions,
dataset provenance and lightweight metric reports in Git. Keep original raw videos
and frames in external storage.


## After downloading the labeled Squat video ZIP - next steps

Only 5 tiny annotation/split JSON files are needed to validate the video archive;
these files are supplied by the original dataset publisher.

```bash
mkdir -p data/external/fitness_aqa/squat/labeled/annotations
AQA_JSON=data/external/fitness_aqa/squat/labeled/annotations

gdown 11ew1zELHx_d4vWlQ993IMeSW9-iDUASv -O "$AQA_JSON/train_keys.json"
gdown 1v450hOnfjcUjRORmA8Ac_aMoTZ_iFV2K -O "$AQA_JSON/val_keys.json"
gdown 1OtE4QheK-sd1bCZr9_A10-p0aY4s25D1 -O "$AQA_JSON/test_keys.json"
gdown 1v5n0gZDnN0WjZ5TNhT4U5i9wxANtv_qY -O "$AQA_JSON/error_knees_forward.json"
gdown 1DtJ9w8oX1wv7vtUEixbnBmCYWhgv6Cki -O "$AQA_JSON/error_knees_inward.json"

# Inspect ZIP integrity and contents WITHOUT extracting videos:
unzip -t data/external/fitness_aqa/squat/labeled/videos.zip
unzip -Z -1 data/external/fitness_aqa/squat/labeled/videos.zip | head -30

# Compare every ZIP video ID with the publisher-provided split and error labels:
python scripts/inspect_fitness_aqa.py
```

Dataset JSON files examined on October 26, 2026:
- train: 1,136 video IDs
- validation: 243 video IDs
- test: 244 video IDs
- combined: 1,623 unique video IDs, no overlap between splits
- annotation dictionaries contain matching keys for all 1,623 IDs

These counts describe split/annotation metadata only; the ZIP video filename
coverage cannot be confirmed until you run the local inspector on your downloaded ZIP.

The independent research pipeline must **not** equate knees-forward or knees-inward
annotations with globally "correct/incorrect" exercise form. Keep error types
and video/time intervals separate. Likewise, split membership is video-ID-disjoint,
not confirmed athlete/subject-disjoint.

If the inspector finds unknown extensions, archive structures, or missing IDs,
review its `first_video_paths` / missing examples and correct path matching
before extracting. Do not invent files to satisfy validation.


## Timestamped pose extraction from the verified ZIP (new)

The archive inspector may legitimately report 116 additional video keys not named
by the publisher's train/validation/test JSON. This is **not a corrupt ZIP**:
all 1,623 officially assigned videos are present, so extra keys are ignored.

To start with a small safe pilot using only 8 train videos, without
extracting the whole 970 MB archive:

```bash
python -m pip install -e '.[camera,dev]'
python -m scripts.extract_fitness_aqa_pose --split train --max-videos-per-split 8 --sample-fps 2
```

For a controlled smoke pilot of all three publisher splits:

```bash
python -m scripts.extract_fitness_aqa_pose --split all --max-videos-per-split 8 --sample-fps 2
```

This creates `data/features/fitness_aqa/fitness_aqa_train_pose.csv` (also
`_val_pose.csv` and `_test_pose.csv` with `--split all`) and
`fitness_aqa_extraction_summary.json`. All are ignored by Git.

Each CSV row preserves `video_id`, publisher `split`, `timestamp_s`,
`frame_index`, and **separate** `error_knees_forward`,
`error_knees_inward` targets, plus the 103 pose features.

The script uses only one video at a time: it decompresses an archive member
into a temporary file, decodes/samples frames, then deletes the temp file.
It never mass-extracts the ZIP and never commits video or image data.
Frames very close to an annotation start/end are excluded (by default +/-0.1 s).
Frames without reliable MediaPipe pose are counted and excluded.

**When the smoke pilot succeeds**, process all official videos only after
checking available space and runtime:

```bash
python -m scripts.extract_fitness_aqa_pose --split all --max-videos-per-split 0 --sample-fps 2
```

`--max-videos-per-split 0` intentionally means *all* clips. This will take
substantially more work than the 8-video smoke check; model accuracy is not
measured by extraction. The output summary reports skipped and accepted frames,
along with labels present among accepted frames.

**Research and evaluation warning**: keep publisher video-id-disjoint splits
unchanged; they are not verified subject-disjoint. Do not feed this dataset to
the existing `gym-pose benchmark` command because it requires real subject IDs
and a *global* correct/incorrect label that this dataset does not provide.
This dataset supports **two error-specific interval labels**; a 0 denotes
absence of that particular named annotation, not a fully correct squat.
Train/validation should be used for initial model selection; lock away test
until evaluation protocol is fixed.
