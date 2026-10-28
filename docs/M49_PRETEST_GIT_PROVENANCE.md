# M49: Pre-TEST Git source parity and study preregistration

## Starting point: owner-confirmed M48 freeze

The owner successfully created and verified their M48 report:

- Frozen Git HEAD at M48: fee514c3b34213ad34e3472e88477bfc7fd74ca9.
- Original M46 and M47 JSON bytes match the stored SHA256 hashes.
- All 17 source files used for TRAIN selection, pose extraction, event
  matching and frozen M45/M39 comparisons match the local SHA256 lock.
- The M48 verification command accessed no TEST or VALIDATION media.

This is a useful research-chain-of-custody result, but M48 by itself
**does not compare local locked source bytes to the Git-tracked blobs**
at its recorded Git commit. Someone could change a research file
locally before freezing it while retaining an unrelated Git HEAD.

M49 addresses ONLY that provenance question. No detector is changed.

## Pre-TEST gate checks

M49 first executes exactly the read-only M48 verification: original
M46/M47 JSON bytes, source-code fingerprints and semantic owner evidence
must all still match the M48 report.

M49 also compares **each of the 17 source files** byte-for-byte against:

1. The exact tracked source blob at the original M48 Git commit.
2. The exact tracked source blob at the current Git HEAD.
3. The current working-tree file bytes, which M48 also validates.

All three must hash to the same M48 SHA256. Since M49 itself must be
added as a new commit, current HEAD will normally differ from the
M48 commit; this is ALLOWED as long as none of the 17 frozen research
files changed. Changes to unrelated docs, this new M49 tool or other
non-fingerprinted files do not invalidate the model source lock.

M49 therefore checks something M48 did not establish: the original
M48 frozen model and evaluator bytes really match the recorded Git
commit, and no subsequent commit or local edit modified them.

If one research file was changed locally before M48's first freeze,
M49 must STOP. Do not solve this by silently rewriting the original
M46/M47 JSONs, overwriting M48's lock or running TEST. Inspect the
Git difference and record whether the original freeze was actually
commit-aligned.

## Pre-registered outline for a future TEST

M49 records a *proposed* one-time protocol; it does not authorize or
execute a test. Future authorized analysis must freeze, in advance:

- Candidate: M45 relative-excursion detector, exactly as at M48.
  Control: unchanged M39 low-to-high detector.
- Primary: micro event F1 with original M37 one-to-one GT-end
  tolerance matching, on the 200ms offline MediaPipe pose track.
- Mandatory accompanying metrics: per-video absolute count MAE,
  precision, recall, TP/FP/FN, exclusions and FP events/video.
- Secondary: every-other observation of the SAME 200ms dense
  MediaPipe tracking context, nominal 400ms cadence. Do not call
  this an independent sparse MediaPipe execution.
- Subject/participant resampling if authoritative participant IDs
  are available; otherwise a qualified paired source-video
  bootstrap, 5000 draws, seed 2026, reporting the dependence risk.
- Complete eligible publisher TEST Squat cohort, without selecting
  clips after observing results, with strict original publisher
  GT and split checks, and a single precommitted run after permission.
- No threshold, model, matching, pose pipeline or exclusion changes
  after examining future TEST outputs.

This is not yet a license check or final executed TEST protocol.
Any future TEST execution requires a separately authorized, fully
specified evaluation action. M49 has NO TEST reading function.

The M49 JSON explicitly reports
external_test_readiness_approved=false, because the independent
subject/session relationship, authorized archive provenance,
publisher labeling/time synchronization, offline/browser pose
parity, TEST authorization, and negative-only Squat controls
remain unresolved. The original TRAIN Squat source lacks
publisher-labeled zero-rep negative videos.

## Ubuntu commands

From the project root, keeping the owner's original M46, M47
and M48 reports unchanged:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m49_pretest_git_provenance.py -q

# Read-only preflight; creates NO report:
python -m scripts.m49_pretest_git_provenance --check-only

# Save a local, Git-ignored provenance and preregistration report:
python -m scripts.m49_pretest_git_provenance \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --m48-lock data/processed/m48_train_evidence_freeze.json \
  --report data/processed/m49_pretest_provenance.json
~~~

The --check-only flag reruns all checks without writing any file.
Report generation refuses to overwrite an existing M49 JSON.

The report only contains commit hashes, source counts, the SHA256
of M48 lock and preregistration statements. It never saves video
names, landmarks, per-event timestamps or any TEST data.

## How to interpret results

If research_source_provenance_check_passed=true,
source_bytes_match_original_m48_git_commit=true and
source_bytes_match_current_git_commit=true, then the original
M48 source freeze is consistent with BOTH tracked Git revisions.

That proves the *software snapshot* is traceable; it does not
prove the source data are licensed/unaltered, participants are
independent, the algorithm generalizes, the publisher TEST score
or the physical correctness/safety of squats.

The preceding M47 95% bootstrap confidence-style intervals are
conditional n=20 TRAIN video percentile intervals. They do not
measure a subject-disjoint population, and the FP confidence
intervals still allow increased false alarms.

Recommended next goal after M49: resolve dataset provenance,
official annotation alignment, participant/split leakage and
negative controls BEFORE separately approving any TEST run.
