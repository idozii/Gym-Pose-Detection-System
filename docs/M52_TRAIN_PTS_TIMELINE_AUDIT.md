# M52 - TRAIN video frame PTS vs original FPS-derived GT time

## Evidence inherited from original owner runs

M51 established SHA256 **byte parity for 40/40** original author
Squat TRAIN MP4s: first20 312 ground truth reps, next20 209
GT reps, 521 total. M50 found zero header/decoded frame
count mismatches and no publisher annotation bounds extending
past actual decoded frames. Source video FPS ranges across these
cohorts from approximately 23.976 to 30. The original TRAIN
CSV has no recognized explicit subject/session/camera IDs.
Neither M50 nor M51 confirmed presentation timestamp parity.

The original M37-M47 event matching converts publisher frame
boundary indices using:
    nominal_time_ms = frame_index / OpenCV_fps * 1000
This implicitly assumes a constant frame rate and correctly
mapped labels. M52 checks only the FIRST assumption, and
only descriptively on existing TRAIN media.

## Frozen M52 diagnostic protocol

M52 runs the original M49 source/Git integrity check and
verifies original M50/M51 owner JSON SHA and counts. It
streams the original publisher TAR to read ONLY the
original TRAIN CSV and checks that CSV SHA against the
M48 lock. Exactly 40 original SHA256 filename-hash-ranked
TRAIN MP4s are selected from previously extracted M45/M46
roots. M52 never accesses publisher TEST or validation.

For each TRAIN video:
- Use OpenCV only to read nominal FPS and frame count.
- Use FFprobe on v:0 (video stream only) with -show_frames
  and frame=best_effort_timestamp_time to observe displayed
  video frame PTS. The number of FFprobe PTS observations
  must equal the original OpenCV video frame count.
- Reject missing, nonfinite, duplicated, non-monotonic,
  excessive or otherwise inconsistent PTS values.
- Compare relative PTS[i]-PTS[0] with i/OpenCV_fps.
- Compute inter-frame duration and report steps deviating
  more than 20% from the median frame step.
- Compare GT boundary frame-index time at ORIGINAL
  publisher [start,end) indices with observed relative PTS.
  Count boundaries with >50ms absolute clock drift.
- For an end index equal to total frame count, do NOT
  extrapolate an extra nonexistent observed frame timestamp;
  report it as unavailable for direct PTS comparison.

50ms and 20% are predeclared DESCRIPTIVE flag thresholds,
not detector hyperparameters, statistical tests, or changes
to the existing M37 one-to-one matching. The original
counter and all 17 M48-frozen source files remain untouched.

No video names, per-frame timestamp arrays, individual
GT intervals, landmarks or private identifiers are saved.
The new report has only two-cohort aggregate measurements.

## Run on Ubuntu

First ensure FFprobe is available:

~~~bash
ffprobe -version
~~~

If missing, install Ubuntu's ffmpeg package with your
system package manager. Do not change Python pose
model dependencies for this audit.

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m52_train_pts_timeline_audit.py -q

python -m scripts.m52_train_pts_timeline_audit \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --development-root data/external/m45_train_relative_motion \
  --replication-root data/external/m46_train_rank21_40 \
  --m50-report data/processed/m50_train_label_video_integrity.json \
  --m51-report data/processed/m51_train_archive_byte_parity.json \
  --report data/processed/m52_train_pts_timeline_audit.json
~~~

The compressed TAR is streamed to read TRAIN CSV, with
no MP4 extraction. All PTS probes use pre-extracted
TRAIN MP4 files. Report creation refuses overwrite.

~~~bash
python - <<'PY'
import json
r=json.load(open("data/processed/m52_train_pts_timeline_audit.json"))
print("Split:",r["split"])
print("Videos:",r["videos_evaluated"])
print("GT reps:",r["annotated_gt_events"])
print("Descriptive timing flags:",r["PTS_timeline_anomalies_descriptive"])
for c in r["cohort_video_timeline"]:
    print("\n",c["cohort"])
    for key in (
        "videos","GT_reps","ffprobe_frame_count_total",
        "videos_with_irregular_interframe_PTS",
        "irregular_interframe_steps_total",
        "videos_with_any_gt_boundary_drift_gt50ms",
        "GT_boundary_points_with_abs_drift_gt50ms",
        "exclusive_gt_ends_not_directly_timestamped",
        "max_video_abs_timeline_drift_ms",
        "max_abs_gt_boundary_timeline_drift_ms"
    ):
        print(key,c[key])
PY
~~~

## Interpretation boundaries

FFprobe's best-effort timestamps reflect decoder/container
metadata, not an independent physical camera clock.
A zero timestamp-drift flag would support a constant
frame-index/FPS timeline in the 40 original TRAIN videos.
It does NOT prove the human rep labels align accurately
to action endpoints, participant independence, original
publisher archive authenticity or source licensing.

Nonzero GT boundary clock drift may explain part of the
event-end tolerance mismatch, but it is not causal proof.
Never tune matching tolerance or M45 on these TRAIN
results and then call a subsequent test confirmatory.

Official publisher TEST remains untouched, and missing
zero-rep Squat controls and subject/session identity
metadata remain important scientific limitations.
