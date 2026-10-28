# M51: Check the 40 existing TRAIN videos against original TAR.GZ bytes

## Why this matters after M50

The owner's M50 audit read 40 original publisher Squat TRAIN
video copies, exactly the frozen SHA256 ranks 1-20 and 21-40,
total 521 GT repetitions. M50 reported:

- 20 original development videos / 312 GT reps:
  decoded FPS range 25-30, no header-vs-decoded frame mismatch,
  no annotation endpoints past the decoded video, and no GT
  intervals <0.4 sec or >12 sec.
- 20 filename-disjoint TRAIN replication videos / 209 GT reps:
  FPS range 23.976-30, no frame-count mismatch or annotation
  endpoint overflow, no <0.4 sec or >12 sec windows.
- No recognized explicit subject_id, session_id or camera_id
  fields in the original TRAIN CSV. Participant/session/view
  independence is UNKNOWN.

M50 was a gross annotation/frame consistency check. It did NOT
prove byte-for-byte source-video identity. Earlier M45/M46 tools
had already extracted these 40 video files; M51 checks their
SHA256 digests against the same owner-local author TAR members.

Note: the owner's latest attempt to *run* M50 returned
'original report exists, refusing overwrite'. The printed
M50 results came from that previously existing report, not
that second invocation. M51 requires this original M50 report
and checks that its publisher TRAIN CSV hash and M48 lock
hash match its own inputs.

## M51 exact and limited method

- Run M49's original evidence/Git provenance verification before
  opening the owner-provided TAR.
- Stream RepCountA.tar.gz to read ONLY original embedded TRAIN
  annotation CSV, then require its SHA256 == original M48 locked
  TRAIN annotation. Reconstruct SHA256 filename rank cohort
  first20 and second20, with GT reps 312 and 209.
- Locate exactly those 40 original previously extracted TRAIN
  MP4 files under the owner's M45/M46 roots.
- In the next TAR streaming pass, match only explicit publisher
  TRAIN paths and the 40 frozen basenames. Compute SHA256 of
  each selected TRAIN TAR member payload and SHA256 of the
  corresponding owner-local file. Require exact payload
  length and digest equality. No mismatched/missing file
  is silently skipped or replaced.
- Record only cohort-level total source MP4 bytes and a
  SHA256 digest-root over the 20 video SHA256 digests in
  frozen hash-rank order. No filenames or individual video
  hashes are persisted.
- Refuse to overwrite a previous M51 report.

The second compressed streaming pass necessarily passes through
other archive member **headers**, including other split names,
and decompresses the TAR stream to reach the selected TRAIN
members. **Only selected TRAIN member payloads are opened for
SHA256; validation/TEST MP4 payloads are not extracted, hashed
or decoded.** No validation/TEST annotation CSV is parsed.

M51 performs no pose inference, model selection, threshold
change, event evaluation, new accuracy report or TEST run.
The 17 model/evaluator source files frozen at M48 remain
untouched. M50 and M48 reports are also untouched.

## Ubuntu instructions

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m51_verify_train_archive_byte_parity.py -q

python -m scripts.m51_verify_train_archive_byte_parity \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --development-root data/external/m45_train_relative_motion \
  --replication-root data/external/m46_train_rank21_40 \
  --m50-report data/processed/m50_train_label_video_integrity.json \
  --report data/processed/m51_train_archive_byte_parity.json
~~~

Reading a large compressed TAR.GZ requires sequential
decompression. M51 needs TWO streaming passes because
original publisher TRAIN CSV may appear after video members
(the existing safe M37 parser has the same behavior).
It does not create additional video copies or replay M45.

~~~bash
python - <<'PY'
import json
from pathlib import Path
p=Path("data/processed/m51_train_archive_byte_parity.json")
r=json.loads(p.read_text())
print("Protocol:",r["protocol"])
print("Source split payload hashed:",r["source_split_video_contents_hashed"])
print("Matched TRAIN videos:",r["videos_byte_identical_total"])
print("Publisher TRAIN GT:",r["annotated_gt_reps_total"])
print("Status:",r["verification_status"])
print("Archive traversal includes other split headers:",
      r["tar_archive_headers_across_splits_traversed"])
print("TEST/validation video payload hashed:",
      r["validation_or_test_video_payload_hashed_or_decoded"])
for c in r["cohorts"]:
    print(c["cohort"],"matched:",c["videos_byte_identical_to_source_tar"],
          "source bytes:",c["source_video_bytes_total"])
PY
~~~

The output digest roots summarize source-byte identity under the
frozen cohort order. They are not a publisher signature or
external provenance verification.

## Limitations and next scientific gate

If M51 succeeds, we may say that the 40 local TRAIN files
match the corresponding 40 files in the user-owned TAR.GZ
**byte-for-byte**. We still CANNOT conclude:

- the TAR came directly from the publisher or has a valid
  usage license;
- the author CSV rep interval timestamps are synchronized
  with actual video presentation timestamps, as opposed to
  being within the frame-count range;
- individual participants or sessions are disjoint between
  TRAIN cohorts or between TRAIN and official TEST;
- the Python offline MediaPipe pose tracker matches the
  deployed browser pose/runtime;
- the model performs well on publisher TEST or on correctly
  labeled zero-rep Squat negative videos.

Do not replace frozen M45 thresholds or silently select
more TEST videos to address any failures. An ideal M52
would examine TRAIN video presentation timestamp consistency
(or obtain original publisher annotation specification)
before proposing any single predeclared TEST evaluation.
