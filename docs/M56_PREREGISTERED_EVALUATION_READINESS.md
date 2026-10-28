# M56 - Locked scientific evaluation design (TEST remains BLOCKED)

## Purpose
M56 is a reproducible **preregistration and evidence-readiness gate** for a
possible **future, separately authorized, one-shot publisher TEST** study.
It is NOT a TEST evaluator, not permission to read TEST/VALID and not
a claim of conference-ready generalization. The original M48-frozen
research pipeline and M34/M39/M45 thresholds are not modified.

The [official ShanghaiTech RepCount website](https://svip-lab.github.io/dataset/RepCount_dataset.html)
and [Hu et al., TransRAC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html)
describe YouTube-sourced Part-A videos and temporally annotated
action-cycle start/end intervals. They do **not**, on their own,
authenticate the owner's TAR bytes, grant blanket video redistribution
or commercial rights, determine an anatomical knee phase or settle
the annotation's inclusive/half-open frame indexing.

## What M56 checks
`scripts/m56_preregister_evaluation_readiness.py` accepts ONLY the
existing owner-local aggregate `m55_author_evidence_review.json`,
original `m48_train_evidence_freeze.json`, and M46/M47 TRAIN
reports needed by the **original M49 check-only** verification.

It verifies:
- Existing M55 protocol, TRAIN scope, original 101 Squat TRAIN rows,
  40 selected TRAIN clips and 521 annotated GT cycles; no zero-GT
  Squat TRAIN clips and no unsupported author/TEST/rights claims.
- Original M48 source freeze, the 17 exact research source checksums,
  selected M39/M45 rules and original 200ms/400ms study design.
- **Read-only M49** frozen-source Git byte and owner-report checks.
  It never reruns the pose model or opens the publisher TAR/CSV/video.
- Future TEST remains blocked; output is Git-ignored, small and only
  contains known protocol statements, scalar aggregate assertions and
  M48/M55 JSON SHA256 fingerprints. No source filenames, landmarks,
  GT timestamps, identities, video or TEST manifest is saved.

A failing M49 source gate, changed M55 claim, wrong split, unexpected
TEST authorization, missing frozen path or changed method setting
rejects M56. It refuses to overwrite an existing report. There is
NO command-line flag for enabling TEST.

## Preregistered study proposal, NOT authorized execution

| Dimension | Proposed fixed design |
| --- | --- |
| Target | All eligible **official** Squat TEST entries, if independently authorized |
| Candidate | Frozen M45 relative excursion |
| Control | Frozen M39 low-to-high |
| Primary cadence | Original dense 200ms Python MediaPipe single tracking history |
| Secondary cadence | Every-other 400ms observation from the same dense tracking history |
| Primary metric | Micro event F1, using original M37 one-to-one GT-end matching |
| Matching tolerance | Original M37 documented `max(750ms, 35% GT duration)`, capped at 2500ms |
| Secondary metrics | Event TP/FP/FN, precision/recall, mean per-video absolute count MAE, exact count, OBO, FP events/video |
| Positive-GT normalized MAE | Average of per-video `abs(pred-GT)/GT`, only for GT>0 |
| Zero-GT policy | Include all authorized negative examples for FP, count MAE, exact/OBO; normalized MAE undefined for GT=0 and separate denominators |
| Uncertainty | Paired 5,000-draw percentile bootstrap, seed 2026; source-video unit only when subject identity grouping unavailable, with explicit dependency warning |
| Study stopping | One complete authorized run, no TEST-based threshold tuning or reruns |
| Missing media and labels | Record all eligible exclusions/reasons; resolve integrity questions before execution; never exclude based on performance |

Important: these are **future study design commitments**, not known
properties of official TEST. The exact TEST source inventory and
inclusion/exclusion decision table **cannot be preregistered from the
current owner evidence without opening/confirming an authorized
source**. This is a remaining blocker, not a silent default.

The original three M38 validation clips were already inspected and
are not untouched independent validation. Existing 20+20 TRAIN
filename-disjoint cohorts do not establish participant-disjoint
generalization. Existing browser same-input W6 logic parity does NOT
establish Python Solutions vs browser Tasks Vision pose-model parity.

## Blockers before any independent TEST step

**All remain UNVERIFIED** in M56:
1. Authoritative original archive provenance/release digest
2. Written source and video usage/redistribution rights
3. Annotator boundary, endpoint inclusivity and time-sync semantics
4. Precommitted official TEST manifest/eligible-set definition
5. Authoritative subject/session/camera grouping and overlap policy
6. GT frame-index to decoded-video PTS/FPS protocol
7. Independently authorized annotated zero-GT negative control plan

Even after documentation is obtained, an **independent, explicit
authorization and release review** and a separate preregistered
execution implementation would be necessary. M56 does not approve
or perform that step.

## Ubuntu owner-run steps

Run from the repo root after updating to the M56 merged revision:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m56_preregister_evaluation_readiness.py -q

python -m scripts.m56_preregister_evaluation_readiness \
  --m55-report data/processed/m55_author_evidence_review.json \
  --m48-lock data/processed/m48_train_evidence_freeze.json \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --report data/processed/m56_test_readiness_blocked.json
~~~

Expected result: `status=BLOCKED_REQUIRES_EXTERNAL_AUTHORIZATION_AND_PREREGISTRATION`,
`frozen_source_provenance_check_passed=true`,
`publisher_test_accessed=false` and
`test_execution_authorized=false`. A successful M56 run means the
**protocol and original TRAIN provenance gate are consistent**, NOT
that scientific TEST readiness or permission was granted.

The resulting JSON is Git-ignored and uses no source media. Do not
post the private M46/M47/M48/M55 reports publicly without reviewing
their contents; sharing the small M56 terminal summary is sufficient
for an engineering handoff.

## Conference-paper evidence claims

Safe: original frozen TRAIN-only M39 vs M45 comparisons, exploratory
video-paired bootstrap intervals, exact per-cohort methodological
limitations, author-documented temporal annotation design and
reproducible versioned implementation.

Blocked: official TEST/subject-independent accuracy, per-clip
annotator agreement, anatomical phase alignment, zero-rep specificity,
publication acceptance, video redistribution/commercial license or
clinical safety. Independently collected and authorized negative and
cross-subject datasets would materially improve future paper validity.
