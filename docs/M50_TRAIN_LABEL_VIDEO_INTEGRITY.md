# M50: TRAIN-only publisher label and video-timing integrity audit

## Evidence inherited from owner-confirmed M49

The owner's M49 provenance audit succeeded against the M48
frozen SHA256 lock: all 17 model, pose, GT matching and
evaluation sources agree with original Git commit
fee514c3b34213ad34e3472e88477bfc7fd74ca9
and updated M49 commit
d44694e42d5ad2f9a5838052e4609165631e8038.
The original M46 (20 new TRAIN files, 209 GT) and M47
bootstrap reports are unmodified. No official TEST was read.

M50 does NOT change any frozen source file or produce
additional model accuracy. The hypothesis under test is
the basic **integrity of time denominators and source labels**.

The existing M37/M45 parser enforces paired chronological
publisher L1..L2n frame intervals. Pose extraction uses
OpenCV's source fps to map frame number into milliseconds.
Consequently a misunderstanding of the original CSV frame
coordinate system, variable-FPS video, truncated decoding
or mislabeled participant could invalidate later claims.

## Exactly what M50 audits

M50 only reads the publisher **TRAIN CSV** inside the
owner-provided RepCountA.tar.gz. It ranks the source filenames
using the ORIGINAL frozen SHA256 seed 2026 algorithm,
selecting first20 (M45 development) and next20 (M46
replication). The TRAIN annotation CSV SHA256 MUST match
the existing M48 lock. Source GT reps MUST match the
previously observed first cohort 312 and second cohort
209, totaling 521 publisher TRAIN reps.

Rather than re-extracting or rerunning MediaPipe, M50 reads
the forty **already extracted** local Squat TRAIN clips:

- First20 existing video root:
  data/external/m45_train_relative_motion/
- Next20 existing video root:
  data/external/m46_train_rank21_40/

The helper uses each root's
official_tar_selected/train subdirectory. All 40 exact
source filenames must be present. It refuses silent
replacement or a mismatched collection. It DOES NOT
read the local validation/test directories.

For each TRAIN video, OpenCV decodes all source frames
once without running pose inference. M50 checks:

- Header FPS is finite within the frozen M37 5-120 limit.
- Video metadata frame count versus actually decoded frames.
- Publisher last GT frame endpoint is within both source
  claimed and actually decoded frames.
- Every positive count corresponds to chronological non-
  overlapping publisher GT frame intervals.
- Original GT window durations, min/max in seconds,
  short windows under 400ms, long windows over 12s, and
  total frame-index-based labelled time coverage.
- Video duration ranges and FPS range across the two
  pre-existing TRAIN cohorts.

The whole publisher TRAIN CSV header is inspected for
recognizable explicit columns such as subject_id,
participant_id, person_id, session_id or camera_id.
If exactly one compatible field exists and values are
complete on all selected TRAIN records, M50 reports
only the number of distinct column values and the
number of raw identifiers **shared** across the two
cohorts. IDs and video filenames are never written.
No id is inferred from video filename patterns.

An explicit CSV identifier is NOT independently verified
as a true participant identity and absence of the column
is recorded as **unknown**, not zero leakage. No claim of
participant independence across official dataset splits
is possible without inspecting those splits under a
separate authorized protocol.

All 40 file-level measurements exist ONLY in memory;
the output retains two cohort aggregates and metadata
availability flags. Frames are immediately discarded by
the video decoder. There is no model threshold/parameter
change, no performance comparison, no TEST or VALIDATION
frame, CSV, or media read, and no standalone negative
control is invented.

M50 first runs the equivalent of M49 --check-only.
If original M46/M47/M48 or the 17 frozen research source
files fail Git provenance, M50 stops before reading
the publisher archive. It also refuses to overwrite its
own prior report.

## Ubuntu command

After fetching the merged main:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m50_train_label_video_integrity.py -q

python -m scripts.m50_train_label_video_integrity \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --development-root data/external/m45_train_relative_motion \
  --replication-root data/external/m46_train_rank21_40 \
  --report data/processed/m50_train_label_video_integrity.json
~~~

M50 scans the TAR stream **once** to locate/read only
TRAIN CSV data and never extracts compressed video
members from it. It uses previously extracted TRAIN
videos from your original M45 and M46 runs; if these
files have been removed, the strict audit stops.

M50 does not establish that the local MP4 copies are
byte-for-byte identical to their TAR members. This
could be added only via a separate authorized source
checksum pass, not assumed from filename agreement.

The complete decode pass for 40 local videos may read
more frame data than a metadata-only probe, but is
necessary to compare claimed frames with actual frames.

## Inspect aggregated findings

~~~bash
python - <<'PY'
import json
from pathlib import Path

r = json.loads(Path(
    "data/processed/m50_train_label_video_integrity.json"
).read_text())
print("Protocol:", r["protocol"])
print("Split:", r["source_split"])
print("Videos:", r["videos_total"])
print("GT reps:", r["annotated_gt_reps_total"])
print("Timing/decode flags:", r["critical_timing_or_decode_integrity_flags"])

for c in r["cohort_video_timing"]:
    print("\n", c["cohort"])
    for k in (
        "videos", "annotated_reps", "fps_min", "fps_max",
        "fps_mean", "video_count_decode_metadata_difference",
        "videos_decoding_below_90pct_metadata",
        "videos_with_gt_bounds_exceeding_decoded_frames",
        "short_gt_intervals_lt_0_4_sec",
        "long_gt_intervals_gt_12_sec",
        "GT_window_duration_min_seconds",
        "GT_window_duration_max_seconds"
    ):
        print(k, ":", c[k])

print("\nExplicit TRAIN metadata availability:")
for name, check in r["explicit_csv_identifier_availability"].items():
    print(name, check)
PY
~~~

## Limits and researcher interpretation

A zero observed frame-count/annotation-bound failure is
**useful evidence of gross label-video integrity** but
not proof that publisher GT and decoded video timelines
are synchronized precisely. Header FPS might differ from
true packet presentation timing, and manual rep boundaries
may not be defined using identical frame-index conventions.

M50 can identify possible repeated **raw metadata labels
within TRAIN** if author columns exist, but cannot verify
authoritative subject IDs or overlap with any TEST person.
A participant-disjoint evaluation would need explicit
trusted identity metadata, data source permissions, and
a study design separate from this TRAIN-only audit.

No new F1/MAE, no TEST score, no injury-risk or squat
quality classifier is claimed. M48's software freeze,
M45 detector and publisher TEST embargo remain intact.
