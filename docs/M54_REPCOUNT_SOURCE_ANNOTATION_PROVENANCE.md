# M54 - RepCount-A Source Provenance and Annotation Semantics Gate

## M54 objective

Assess two unresolved assumptions before future paper claims or one-shot
publisher TEST: where the owner's 8-9 GB RepCountA.tar.gz originated,
and what its annotated cycle [start,end] frame positions actually mean.

The ShanghaiTech authors' official source states:
- RepCount-A has 1,041 YouTube-collected videos.
- RepCount-B is a separate student/teacher examination subset.
- Part A has TRAIN/VALID/TEST; Part B is TEST-only.
- Label count is repetition number; label location is time-axis
  placement of each action cycle, with beginning and ending period
  indications. Dataset website describes this but does not settle
  a particular knee-angle phase such as bottom/high or a precise
  inclusive/exclusive endpoint implementation.

Official primary references:
- https://svip-lab.github.io/dataset/RepCount_dataset.html
- https://github.com/SvipRepetitionCounting/TransRAC

A separate Hugging Face/community archive and its license/credit
text is NOT independent proof of byte authenticity of the owner's
specific archive. Original Part-A video creators retained rights
over YouTube-uploaded source videos, so dataset web availability
does not imply blanket commercial-use or redistribution permission.
The official web page links downloads but does NOT, by itself,
provide a fully verified redistribution license for each video.
Do not package publisher video or GT labels in the Vercel demo.

## Exact label contract currently used in the benchmark

Original publisher CSV columns are name, type, count, L1, L2, etc.
The already frozen project parser treats one pair of integer
boundaries (L_{2j-1}, L_{2j}) as an ordered nonempty frame span.
The project evaluates end locations on a half-open [start,end)
timeline. That is an IMPLEMENTATION CONVENTION of the benchmark,
not an independently verified description of the original human
annotation policy or an anatomical movement phase label.

M54 checks the **original TRAIN CSV only**:
- Parse with unchanged original M37 annotation parser.
- Match CSV SHA256 against the original M48 TRAIN freeze.
- Read-only original M49 check of 17 frozen research source files,
  original M46/M47 report checks and Git source revisions.
- Re-select the exact two 20-file hash-ranked TRAIN cohorts
  used by M45/M46: 312 + 209 annotated Squat cycles.
- Count positive- and zero-repetition Squat source rows in
  the entire TRAIN CSV and each original selected cohort.
  A zero-rep row is possible in the original parser. Do NOT
  assume the entire publisher Squat TRAIN dataset has none
  merely because the selected 40 videos have positive GT.
- Verify ordered non-overlapping interval pairs and report
  only aggregate lengths, gaps and boundary counts.
- Reuse M50 schema-only person/session/camera field inspection.
  No participant ID is inferred from filename patterns.
- Report licensing, original download identity and human
  boundary semantics as NOT INDEPENDENTLY VERIFIED.

M54 does **NOT** fetch/download anything over the network,
access publisher TEST/VALID CSV/video, decode/source videos,
run any detector, change thresholds, read image frames, or
save individual filenames, source URLs, person IDs, GT locations
or per-video counts. The output is a Git-ignored aggregate JSON.

## Ubuntu one-time run

Use the original owner-local TAR and M46/M47/M48 reports.
The original TAR reader may stream through the large compressed
file to locate TRAIN CSV and therefore performs sequential
decompression, but extracts no videos. Run from repository root.

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m54_train_annotation_provenance.py -q
python -m scripts.m54_train_annotation_provenance \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --m48-lock data/processed/m48_train_evidence_freeze.json \
  --report data/processed/m54_train_annotation_provenance.json
~~~

Report creation refuses overwrite. No public TRAIN label data are
embedded in website build. The owner-local path is ignored by Git.

Summarize without publishing filenames:

~~~bash
python - <<'PY'
import json
r=json.load(open("data/processed/m54_train_annotation_provenance.json"))
x=r["train_label_aggregate"]
print("split:",r["source_split"])
print("Squat TRAIN rows:",x["train_squat_annotation_rows"])
print("Squat TRAIN zero-rep clips:",x["train_squat_zero_gt_videos"])
print("recognized identity columns:",
      {k:v["recognized_explicit_column"] for k,v in
       x["recognized_explicit_person_session_camera_schema"].items()})
print("cohorts:",x["original_frozen_cohorts"])
print("blocked claims:",r["license_and_provenance_blockers"])
print("TEST/VALID opened:",r["test_or_validation_csv_or_media_opened"])
PY
~~~

## Additional human verification gate (M54 does NOT do it)

1. Find the ORIGINAL download email/link, date, release version,
   or checksum provided by the authors. Record a stable official
   artifact digest if actually published; do not invent one.
2. Check the original official download package or author response
   for a legally applicable license and whether reproduction of
   screenshots/YouTube source videos in a public web portfolio,
   paper supplement or commercial product is allowed. Ask
   the authors when unclear; official citation alone is not a license.
3. Seek official annotation instructions explaining whether each
   boundary denotes the start/end of the full visual action,
   exact frame conventions, and whether pauses or truncated
   partial reps are labeled. Avoid relabeling ground truth using
   the knee-angle detector's own predictions.
4. Confirm whether the same person/scene/view appears across
   TRAIN source filenames. Explicit metadata or independently
   documented grouping is necessary; guessing IDs from filenames
   does not establish subject independence.
5. Do not run the publisher TEST split until protocol, permission,
   and original source identity are settled and preregistered.

## Claims that remain blocked

Until evidence is obtained:
- "The downloaded archive is the official original publisher byte set".
- "The project uses a permissive/commercial redistribution license".
- "TRAIN replication is participant-independent".
- "Publisher GT boundaries are exactly high/low knee-angle phases".
- "M45 is validated across subjects and on official TEST".
- "The model prevents injuries or certifies fitness technique".

The author source confirms dataset DESIGN and general cycle
annotation meaning, not those stronger project-specific statements.
