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
