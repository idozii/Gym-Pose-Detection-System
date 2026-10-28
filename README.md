# Gym Pose Detection - Experimental Fitness-AQA Research Viewer

**Research-only software prototype, not a validated fitness coach or
medical/injury-prevention detector.**

The current code supports a local MediaPipe/OpenCV webcam research viewer
and **six separately versioned source-annotation classifiers** across
Squat, Overhead Press and Barbell Row. It presents annotation scores
or explicit `ABSTAIN`; it never interprets missing labels as proper
technique, and does not issue autonomous exercise safety alerts.

The owner demonstrated **201 passing tests** and a **180-frame Ubuntu
Squat webcam run with 360 successful source-label calls**, observed
**15.78 displayed FPS** and **11.16ms mean successful per-task
feature+RF inference**. This is an engineering runtime demonstration,
not a test of correctness or robust generalization. Evaluation is
exploratory on repeatedly consulted publisher validation samples.

### Free GitHub Pages mirror for local Push-up pose preview

Vercel Hobby returned HTTP 402 because the production deployment
quota was reached for the 24-hour period. The latest Push-up
preview build exists but is protected by Vercel SSO. A separate
[GitHub Pages static frontend workflow](docs/M30_FREE_GITHUB_PAGES_FALLBACK.md)
can publish the local Push-up camera geometry explorer for free,
after the repository owner enables GitHub Pages - GitHub Actions
as the source in repo Settings. It does **not** expose Python
model scoring or private API/model binaries, and it does not
change the Vercel production site.

---

**M37 source availability / 404 note:** A publicly indexed third-party
RepCount-A Hugging Face page did not guarantee live download access;
the owner encountered HTTP 404 at the annotation-file URL.
The revised [M37 source diagnostic](docs/M37_REPCOUNT_REAL_VIDEO_EVALUATION.md)
adds a metadata-only `--check-source` command before any video fetch,
clear repo-vs-missing-file errors, and optional `--local-source` to
evaluate an already authorized, publisher-compatible RepCount-A
mirror layout without new annotations or network access.
If neither source is available, real-video accuracy remains **unknown**
rather than synthesized.

**M37 owner's actual author archive is TAR.GZ:** The downloaded
`~/Downloads/RepCountA.tar.gz` is about 8.9 GB, **not**
a ZIP file. The [M37 TAR.GZ runbook](docs/M37_REPCOUNT_REAL_VIDEO_EVALUATION.md)
now accepts `--inspect-source-archive ~/Downloads/RepCountA.tar.gz`
and `--source-archive ~/Downloads/RepCountA.tar.gz --max-videos 3`.
It streams compressed members and extracts only selected
already-annotated Squat clips, with external publisher CSV
option when the TAR has no embedded labels. Full archive
extraction and new human annotation are unnecessary;
the 8.9GB compressed stream may still require sequential
decompression to find source members. Real accuracy
remains unmeasured until actual videos are evaluated.

**M37 author's OneDrive ZIP support:** The official RepCount
site links to an original OneDrive folder (extraction code
`repcount`); the user provided that source after the
third-party Hugging Face mirror returned HTTP 404.
[M37 official archive guide](docs/M37_REPCOUNT_REAL_VIDEO_EVALUATION.md)
adds `--inspect-source-zip /path/to/RepCountA.zip` (metadata
only) and `--source-zip /path/to/RepCountA.zip --max-videos 3`.
It parses publisher CSV `name/type/count/L*` cycle bounds
from ZIP, or accepts `--annotations-csv` for separately
obtained official labels; extracts only matching Squat
MP4s, never the full archive. Actual archive layout is
unverified until inspected on the owner's Ubuntu machine.
No fabricated labels, rep scores or production claims.

### W7b + M55 - Privacy-safe real-device QA snapshots and author evidence review

[W7b local browser QA capture](docs/W7_BROWSER_PERFORMANCE_MOBILE_AUDIT.md)
adds an explicitly clicked, on-page **aggregate-only** diagnostics
snapshot after >=30 processed frames, without uploading or
storing any video, landmark, device fingerprint or file name.
The existing rolling latency, throughput and gap metrics now
report gaps in the recent window rather than permanent startup
stalls. No browser/mobile device speed benchmark is claimed
until the owner tests and shares real device results.

[M55 RepCount author evidence and annotation semantics gate](docs/M55_AUTHOR_ANNOTATION_EVIDENCE_GATE.md)
adds a read-only local summary tool that checks the owner
M54 TRAIN report against the original M48 TRAIN annotation
SHA and original 20+20 cohort invariants. It documents what
the **original authors** state (YouTube origin, start/end
action-cycle interval annotations, two-annotator review) and
separately what the owner observed (101 Squat TRAIN clips,
0 zero-rep examples, 521 selected GT events). It does NOT
certify official TAR byte provenance, redistribution/
commercial rights, an anatomical phase interpretation of
GT boundaries, subject independence or official TEST.
No original research detector thresholds or 17 frozen M48
paths are changed.

---

### W8 - Multi-window browser robustness and mobile acceptance (pending hardware)

[W8 private session stability protocol](docs/W8_BROWSER_ROBUSTNESS_MOBILE_VALIDATION.md)
adds opt-in, browser-local rolling session envelopes over non-overlapping
120-frame windows (maximum 16 retained). Capture requires 240 processed
frames; it reports per-window median/P95 and analyzed-throughput ranges
plus session processing gaps without raw video, device fingerprints,
landmarks, files, server telemetry or persistence. Synthetic CI does
NOT establish real-device speed or multi-browser compatibility.
Ubuntu owner W7 Squat/OHP snapshots remain separate from W8 and from
any official publisher benchmark.

---

### W7 + M54 - Local browser performance and TRAIN dataset provenance gate

[W7 real-device performance and mobile acceptance](docs/W7_BROWSER_PERFORMANCE_MOBILE_AUDIT.md)
adds bounded, nonpersistent on-device pose processing median/P95
latency, real processed-frame wall throughput and observed
processing gaps to the existing camera/local-video studio.
These statistics are **not** camera sensor FPS, dropped-frame
counts, GT performance or measured device benchmarks from CI.
Users still need to test Android/iPhone camera over trusted HTTPS
and confirm local model/codec behavior. W7 sends no telemetry.

[M54 RepCount-A provenance and annotation semantics](docs/M54_REPCOUNT_SOURCE_ANNOTATION_PROVENANCE.md)
documents the official ShanghaiTech / TransRAC author's
RepCount-A description (Part-A sourced from YouTube,
1,041 videos, cycle count/location annotations) while
explicitly leaving video redistribution/commercial licensing,
human frame-boundary phase semantics, owner archive authenticity
and subject/session identity unverified. A local audit only reads
the original TAR's TRAIN CSV and verifies its M48 SHA and the
original read-only M49 gate. It summarizes the total number
of zero-rep Squat TRAIN source clips separately from the 40
previously selected positive-GT Squat videos, then reports
schema/interval consistency without saving filenames,
identities or GT intervals. **No TEST/VALID access and no video
extraction.** M54 owner-run outputs are unavailable until the
owner executes the command.

Original M34/M39/M45 rules, 17 M48-frozen files and W5
research numbers are unchanged. CI verifies functions and
frontend build only, not physical phone performance or source
authorization.

---

### W5 M53/W6 results addendum (owner-run TRAIN replication)

The W5 research page now includes **owner-reported M53**
normalized count MAE (M39 0.4940 to M45 0.4100 at 200ms;
0.5398 to 0.4838 at 400ms), OBO and exact-count results
from the same 20 frozen replication TRAIN files (GT209).
The user-reported W6 same-input JS event-parity report
found 0 direct and 0 gated event timestamp disagreements,
while the W2 200ms gate discarded 2 source frames in total
(400ms condition: 0). These statements are **not**
browser MediaPipe/Python pose parity, official TEST,
or published paper benchmarking. Owner-reported M53 paired-video
bootstrap 95% intervals are now shown for each TRAIN cadence:
normalized MAE and OBO intervals contain zero at 200ms and
400ms; exact-count touches/includes zero; absolute count MAE
intervals exclude zero in the direction favoring M45.
These are not multiple-comparison-adjusted or subject-disjoint
population estimates; original private JSON bytes were not
independently examined.
The development cohort never inherits replication M53
numbers. [W5 evidence scope and updates](docs/W5_RESEARCH_EVIDENCE_PAGE.md).

---

### M53 + W6 - TRAIN normalized count metrics and same-input M45 browser logic parity

[M53 normalized count MAE / OBO / exact-count protocol](docs/M53_TRAIN_NORMALIZED_COUNT_METRICS.md)
and [W6 M45 incremental/browser logic parity](docs/W6_BROWSER_SAME_INPUT_PARITY.md)
add a single owner-run, original publisher TRAIN-only replication
audit on the **same frozen 20 M46 source-file hashes** (GT209).
M53 reruns original M47 replay and cross-verifies original
M46 aggregate count MAE, exact count fraction and within-one
fraction, then calculates video-mean normalized absolute count
error and paired 5,000-draw video-bootstrap percentile intervals
without saving private per-clip predictions or raw landmarks.
W6 compares **identical offline Python landmark inputs**
and timestamps between original M45 and the experimental
incremental website M45 counter, distinguishing direct input
from the browser's nominal 200ms sample gate. Both original
M49 Git/source gate and M46 reference are mandatory; the
separate owner-local output reports remain Git-ignored.
**No M53 or W6 owner-run numerical result is available before
running the command.** W6 same-input algorithm parity is NOT
browser MediaPipe Tasks Vision vs Python Solutions end-to-end
pose-estimator parity or live-camera accuracy. No publisher
validation/TEST data, source archive source identifiers or
17 M48-frozen modules are modified/accessed outside original
TRAIN scope.

---

### W5 - Public research evidence page (TRAIN-only, frozen M46-M52)

[W5 research evidence page and verification](docs/W5_RESEARCH_EVIDENCE_PAGE.md)
adds a separate, responsive `web/research.html` page with
an accessible dropdown for original development TRAIN
versus filename-disjoint TRAIN replication and nominal
200ms versus paired shared-tracker 400ms observation.
It displays original M39/M45 TP/FP/FN, dynamically
computed micro Event F1 and absolute per-video count
MAE from frozen M46/M47/M48 owner-confirmed aggregates,
with video-level 5,000-draw paired bootstrap intervals
ONLY for the 20-video TRAIN replication cohort.
The page also presents the original motion-rule
methodology, explicit M50-M52 TRAIN video integrity
results, source documentation links and open research
limitations. This is never a publisher TEST score,
participant-disjoint estimate, calibrated safety
metric, or MediaPipe browser/Python parity result.
No video or landmark data, annotation files or
person-level records are embedded. A Vite multipage
build retains the W1-W4 index studio and separately
builds research.html for Vercel and GitHub Pages.
Original 17 M48-frozen source paths and all W1-W4
detectors remain unchanged.

---

### W4 - Local-only MP4/WebM video analysis (no upload)

[W4 local file analysis and manual acceptance](docs/W4_LOCAL_VIDEO_ANALYSIS.md)
adds a user-selected MP4/WebM input to the W1-W3 browser
workbench (<=250 MiB and <=5 minutes). It decodes frames
locally with the same on-device MediaPipe Tasks Vision,
reuses unmodified M34 and the experimental W2 M45
Squat counter, and displays W3 tempo/visibility
observations. The video model clock uses HTMLVideoElement
currentTime, not RAF wall time. Playback is manual
(Play/Pause), and Replay creates a **fresh** tracker and
resets cycle/timing data. No seeking controls are exposed.
Files use temporary blob: URLs, revoked when changing or
clearing sources; no raw video, file name, landmark or GT
is uploaded. The optional remote Python RF research
inference is forcibly disabled in local video mode,
regardless of prior camera consent. This is an
experimental browser demonstration, not evidence of
rep-count correctness, source GT, publisher TEST
performance, technique safety or browser/Python parity.
Original M48-frozen research source remains unchanged.

---

### W3 - Private local session dashboard for rep-cycle visualization

[W3 local session dashboard](docs/W3_SESSION_DASHBOARD.md)
adds an elapsed set clock, analyzed-frame count, pose-seen
coverage, required-joint visibility coverage, readable
tracking status and **bounded 12-observation tempo charts**
to the live Squat/OHP/Row browser workbench. The clock
continues to update when camera video stalls, and both
M34/M45 cycle history and session telemetry reset together
on Reset set/camera stop/exercise change. M45's Squat-only
chart explicitly measures low-to-return segment duration,
different from M34's full-cycle time, so heights are not
directly comparable. Coverage is NOT classifier confidence,
form correctness or rep accuracy. Original M34/M45 research
source and M48 17-path SHA256 freeze remain unchanged.
No TEST/VAL media, backend requests, video recording or
persistent pose data are introduced. Tests and Vite build
verify logic but not physical phone cameras or live Vercel.

---

### W1 + W2 - Browser readiness and M34/M45 Squat live research comparison

[W1/W2 browser validation and live comparison protocol](docs/W1_W2_LIVE_RESEARCH_DEMO.md)
enhances the existing local, user-consented browser pose workbench:
explicit camera permission/device error guidance; immediate camera
revocation when a tab is hidden even DURING startup; ended camera
track recovery; and a responsive experimental Squat-only
**M34 / M45 / Compare** readout using the same Tasks Vision landmarks.
The original M34 high-low-high counter stays unchanged at 400ms;
a *separate* new live M45 module uses its frozen relative excursion
rule at a nominal >=200ms sample cadence. Switching view shows/hides
cards but does not change model state; Reset set, camera stop and
exercise change clear both local counters. Synthetic tests validate
event parity against the exact untouched original M45 TRAIN JS
comparator. No GT labels, webcam accuracy, user-video recording,
automatic uploads, exercise-form advice or browser/Python parity
are inferred. All 17 M48-frozen research source files are untouched.
CI validates code and Vite build, but real mobile/desktop camera
testing and public Vercel deployment must be checked separately.

---

### M52 - Audit TRAIN presentation timestamps versus original FPS clock

[M52 original TRAIN PTS timeline audit](docs/M52_TRAIN_PTS_TIMELINE_AUDIT.md)
builds on owner-verified M51 source-byte parity of
40 frozen TRAIN Squat videos (521 GT). It uses FFprobe
per-frame video presentation timestamps to compare
the container-derived clock with the frozen original
M37-M47 GT time formula (frame_index / OpenCV fps).
It reports anonymous two-cohort PTS timing drift at
observed GT boundaries and variable frame-step signals
without changing detector source, event matching or
ever running publisher TEST. Original M48/M49 source
freeze, M50 and M51 owner evidence are required.
Video PTS cannot validate human GT label timing,
archive provenance or subject independence.

---

### M51 - Verify frozen TRAIN video bytes against the original source TAR

[M51 publisher TRAIN source-copy parity](docs/M51_TRAIN_ARCHIVE_SOURCE_PARITY.md)
checks that the 40 already-extracted frozen Squat TRAIN
video files (312 + 209 publisher GT reps) are **byte-for-byte
identical** to their selected original TRAIN MP4 members
inside the owner-provided RepCountA.tar.gz. The owner-run
M50 audit found zero frame decode/header count mismatches
and zero GT bounds extending beyond decoded video frames,
but M50 could not independently establish video-copy source
identity or subject independence. M51 rechecks frozen
M46-M50 report/source provenance, reconstructs exact original
SHA256 filename-rank cohorts and verifies selected original
TRAIN video payload SHA256 plus file length. The large TAR
stream traverses other split headers but DOES NOT open,
hash, extract or decode any non-TRAIN video member payload
or non-TRAIN annotation CSV. Cohort aggregates only;
no original M34/M39/M45 code, thresholds or TEST evaluation
are modified. Archive publisher origin, participant overlap
and precise GT timestamp alignment remain unverified.

---

### M50 - Audit publisher TRAIN GT/frame timing before any TEST

[M50 TRAIN label/video integrity audit](docs/M50_TRAIN_LABEL_VIDEO_INTEGRITY.md)
uses the 40 original publisher TRAIN Squat clips already
extracted in M45/M46 (312 + 209 annotated reps).
After rechecking original M49/M48 Git evidence, it reads
ONLY the embedded publisher TRAIN CSV and its frozen
SHA256, then probes local TRAIN video FPS and *actual*
decoded frame counts against GT frame bounds and reported
header frames. It aggregates GT interval durations, FPS
ranges and timing anomalies, and reports whether original
publisher TRAIN CSV has explicit subject/session/view
fields. It does NOT infer IDs from filenames, evaluate
TEST, generate additional model scores or modify the
17 research files frozen at M48. Clean frame bounds
cannot establish precise GT timing or person-level
independence; original archive provenance and real
negative controls are still unresolved.

---

### M49 - Confirm original M48 source bytes match tracked Git

[M49 pre-TEST Git provenance check](docs/M49_PRETEST_GIT_PROVENANCE.md)
extends the owner-verified M48 SHA256 lock. It checks the
original M46/M47/M48 bytes without alteration, then verifies
that **all 17 locked source files** are byte-identical in the
working tree, at original frozen Git commit
`fee514c3b34213ad34e3472e88477bfc7fd74ca9`
and at the current Git HEAD. This distinguishes a genuine
committed research snapshot from uncommitted changes
that merely happened to be hashed locally. New commits
for M49/docs are allowed if frozen research files remain
identical. M49 also records a prospective, **not yet
authorized**, one-shot future TEST protocol and explicit
blockers: source provenance, independent participant
splits, negative-only controls, GT/video synchronization
and offline/browser pose differences. No TEST, validation
or original video content is accessed by M49.

---

### M48 - Freeze M46/M47 TRAIN evidence and detector source fingerprints

[M48 frozen evidence protocol](docs/M48_FROZEN_TRAIN_EVIDENCE_PROTOCOL.md)
locks original owner M46 and M47 report SHA256 bytes and
17 source file fingerprints BEFORE any possible TEST use.
On 20 new source TRAIN clips (209 GT), M47 found paired
candidate-minus-M39 F1 +4.30 percentage points at 200ms
(95% clip bootstrap percentile +0.18 to +8.49 pp)
and video MAE -0.65 [-1.25,-0.10]. FP/video +0.05
[-0.20,+0.30] is **not** evidence of non-inferiority.
At shared-tracker every-other 400ms, F1 improved +11.18 pp
[+3.13,+19.07 pp], MAE -1.25 [-2.45,-0.20].
M48 adds only report/source SHA256 verification and
strict owner result reconciliation; no algorithms,
thresholds, tests on TEST, independent subject claims
or clinical/exercise quality certification.

---

### M47 - Paired video-bootstrap uncertainty on frozen TRAIN replication

[M47 frozen M46 clip-bootstrap audit](docs/M47_FROZEN_TRAIN_CLIP_BOOTSTRAP.md)
assesses uncertainty WITHOUT modifying M45 or M39. The
owner-run, previously unseen publisher TRAIN rank-21..40
cohort had 20/20 evaluated and 209 GT reps. At 200ms,
the frozen relative-excursion rule attained F1 74.57% and
video count MAE 3.80 vs frozen M39 F1 70.27% and MAE 4.45
(TP +12, FP +1). At paired ~400ms, F1 72.02% vs
60.84% and MAE 4.20 vs 5.45 (FP tied at 6).
Source-video filenames are disjoint, but participant/
session independence is unverified. M47 replays EXACTLY
that cohort and requires the existing M46 report to
reproduce before a deterministic 5000-draw paired
**video-level** percentile bootstrap of delta micro
F1, video MAE and FP events/video. Results are TRAIN-only
conditional uncertainty and never TEST accuracy,
confirmed subject-level generalization or medical/form advice.

---

### M46 - Frozen M45 replication on unused TRAIN source videos

[M46 frozen TRAIN source-file replication](docs/M46_FROZEN_UNUSED_TRAIN_REPLICATION.md)
reuses the **unchanged M45 relative-excursion and M39 baseline**
rules on fixed SHA256 TRAIN filename ranks 21-40 rather than
the already-investigated first 20. M45's owner-run development
cohort at 200ms yielded event F1 72.76% and count MAE 5.8,
versus M39 F1 63.14% and MAE 7.9 (312 source GT reps).
Those initial TRAIN results motivated freezing, **not**
further parameter search. New cohort selection uses
`--cohort-offset 20` and never borrows validation/TEST.
Source-file disjointness is enforced; participant/session
disjointness remains unknown. Negative-control specificity
and external TEST performance remain unmeasured.

---

### M45 - Experimental relative-excursion rep comparator (TRAIN only)

[M45 relative excursion experiment](docs/M45_TRAIN_RELATIVE_EXCURSION_STUDY.md)
tests one standalone, non-deployed hypothesis inspired by owner-run
M44 diagnostics: the original <=120-degree knee-angle rule missed
132 GT repetitions with valid in-window pose samples on the
20-video TRAIN cohort, including 80 minimum angles in
(120,150] degrees. Instead of arbitrarily raising the absolute
threshold, the M45 candidate counts a >=35-degree same-side
descent from a >=155-degree peak, followed by >=35-degree
recovery to >=155 degrees within the original timing
constraints. Compare unchanged M39/M34/zero on identical
publisher TRAIN 200ms poses and every-other 400ms subseries.
M45 aggregates FP tradeoffs and paired GT/video error changes;
its outputs are exploratory **in-sample** development results,
not generalization, safety, or a deployed model.

---

### M44 - TRAIN angular observability and publisher annotation boundary audit

[M44 descriptive boundary-angle experiment](docs/M44_TRAIN_BOUNDARY_ANGLE_AUDIT.md)
uses the unchanged M41/M43 original TRAIN cohort and M39
low-to-high detector, without changing any counting thresholds.
Owner-run M43 found 132/163 dense 200ms false negatives had
usable knee-angle samples but no angle <=120 degrees **inside**
their publisher [start,end) annotation window; 11 unmatched
detections were outside all event-end match tolerances.
M44 distinguishes whether minimum within-GT knee angles
lie just above the fixed threshold or far above it and
whether a low angle occurs within a fixed 750ms on either
side of the publisher interval. False positives are also
classified as inside or outside any source GT interval.
This is post-hoc TRAIN diagnostic evidence, not detector
optimization, held-out results, biomechanical quality
judgment, or proof of label misalignment.

---

### M43 - Frozen TRAIN temporal error audit (descriptive, not a new model)

[M43 temporal error analysis](docs/M43_TRAIN_TEMPORAL_ERROR_AUDIT.md)
inspects publisher half-open GT windows, matching tolerance and
temporal angle evidence to partition **unmatched** GT and detected
events for original frozen M39 low-to-high at shared-tracker
200ms versus every-other-sample 400ms. It records paired
annotated-event recovery (both/dense-only/sparse-only/neither),
reproduces original M41 metrics, and aggregates counts only.

On the owner's M41 20-clip TRAIN cohort, dense low-to-high
obtained TP=149, FP=11, FN=163 (F1 63.14%), whereas sparse
had TP=118, FP=3, FN=194 (F1 54.50%). The original TRAIN CSV
has no publisher-annotated Squat zero-rep clips; the M42
negative-control study correctly reported unavailable evidence.
M43 is post-hoc descriptive, not causal, held-out accuracy
or a physically safe/unsafe squat classifier.

---

### M42 - Publisher-annotated zero-rep TRAIN negative controls

[M42 negative-control evaluation protocol](docs/M42_TRAIN_ZERO_REP_NEGATIVE_CONTROLS.md)
addresses a key blind spot: the owner's 20-video M41 TRAIN cohort
had **zero** publisher-annotated zero-rep clips, so its observed
false-positive events do not measure false alarms on negative
clips. An opt-in `--cohort-filter zero` on the M41 runner
uses publisher TRAIN labels with zero count and no rep bounds,
before fixed-seed clip selection. Original M34, M39 low-to-high
and 200ms/400ms comparison stay unchanged; event F1 is undefined
on an entirely zero-GT cohort, while false-event counts and
false-alarm-video rates are meaningful. Never substitute
validation or TEST clips if no such TRAIN labels exist.

---

### M41 - Paired TRAIN pose observation cadence sensitivity

[M41 paired 200ms and 400ms observation study](docs/M41_TRAIN_PAIRED_CADENCE_STUDY.md)
compares original M34, frozen M39 low-to-high and zero baseline
using 200ms-target MediaPipe TRAIN pose traces and every-other
sample 400ms subseries. M34's internal 400ms gate stays unchanged.
The 400ms subsample shares the dense pose tracker history, so this
is not an independent 400ms MediaPipe run. In owner-run M40,
20 TRAIN clips yielded M39 F1 48.57% and count MAE 10.4,
versus M34 F1 8.59% and count MAE 14.9. No TEST score or
independent generalization is claimed.

---

### M40 - Larger fixed-sampling TRAIN-only development cohort

[M40 TRAIN hash cohort protocol](docs/M40_TRAIN_HASH_COHORT.md)
adds deterministic, fixed-seed filename-hash selection of up to 20
publisher TRAIN Squat clips, reducing dependence on archive member
order. In the owner-run **five-video development pilot**, M39's
low-to-high rule achieved event F1 65.26%, count MAE 3.8,
versus original M34 F1 6.90%, MAE 10.8 (56 source reps).
These numbers are **TRAIN-only exploratory observations**, not
held-out performance or a trained model. M40 retains the original
M34, temporal detector, event matching and TEST embargo.

---

### M39 - TRAIN-only temporal counting baseline (development, NOT held-out test)

[M39 fixed temporal baseline protocol](docs/M39_TRAIN_ONLY_TEMPORAL_BASELINE.md)
compares an always-zero count control, the **unchanged M34** browser
counter, and a simple single-observation low-to-high knee-angle
return detector on **identical publisher TRAIN pose traces**.
The frozen exploratory M39 detector is separate from the
deployed website; this is not a trained model and not a
validated improvement. No validation/TEST clips, private webcam,
manual rep labels or external dataset download are used.
All outputs retain aggregate statistics only; any future TEST
evaluation requires a pre-registered, frozen research design.

---

### M38 - Reproducible 3-video real-pilot failure audit (no data reload)

[M38 research protocol and aggregate baseline](docs/M38_PILOT_FAILURE_AUDIT.md)
checks the already-generated M37/M37.1/M37.2 JSON reports are
consistent, and compares M34 to an always-zero counting control.
The owner-run 3-video pilot observed **1 of 45** Squat events,
F1 ~4.35%, 14.67 video-level count MAE versus **15.0** MAE for
the always-zero baseline. These are limited exploratory findings,
not successful generalization, browser parity or evidence of
safe exercise technique.

Run with `python -m scripts.m38_pilot_aggregate_audit` and the
three local reports (full command in the M38 document). No
video/pose decode, manual labels, external downloads, M34 tuning,
or touching TEST is required. A separate development cohort and
genuinely held-out TEST are required before any new method can
claim independent improvements.

---

### M37 - Automatic RepCount real-video Squat evaluation (no user labeling)

[M37 automated RepCount benchmark](docs/M37_REPCOUNT_REAL_VIDEO_EVALUATION.md)
selectively reads an already-annotated RepCount-A Squat subset
(default three validation clips) from the owner's author-hosted
`RepCountA.tar.gz` or an eligible third-party mirror, extracts
chronological local MediaPipe Python poses with OpenCV, and replays
them through the **exact M34 JavaScript `RepCycleCounter`** to
compare event timestamps against published cycle-frame bounds.

It outputs actual-video count MAE, event TP/FP/FN,
precision/recall/F1, tracking coverage and excluded clip counts
only **if one or more real annotated video clips were decoded
and evaluated successfully**. GitHub CI uses synthetic mocks:
no dataset download, actual RepCount accuracy result,
browser/Python estimator parity, source annotation/clip
byte-level provenance or OHP/Row event ground truth is
claimed by CI. No user-created CSV, manual webcam rep count,
private data, model training or original M20 TEST access.

On Ubuntu, for the inspected 8.9 GB archive with the built-in
`LLSP/annotation/valid.csv`, run from the repo root with
the existing research venv and Node 22:

```bash
python -m scripts.m37_repcount_video_benchmark \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --split validation --max-videos 3
```

No Hugging Face download or external CSV is required for this
archive layout. The author's archive remains local; only
selected clips are extracted. The old documented fallback
`TransRAC/open_set/new_valid.csv` contains front-raise labels,
not Squat, and should **not** be used for this pilot.
Actual accuracy metrics remain pending an owner-machine
run of the genuine annotated video pipeline.

---

### M36 Auto - deterministic cycle stress suite (no manual validation)

The owner does **not** want to manually count repetitions,
label videos or populate session CSV files. The updated
[M36 automated validation guide](docs/M36_AUTOMATED_CYCLE_VALIDATION.md)
provides a **one-command regression audit** running the
unmodified M34 `RepCycleCounter` on **216 generated
angle-cycle scenarios** across Squat/OHP/Barbell Row,
two observed limbs, in-plane image perturbations, jitter,
tracking loss, partial movement and stationary periods.
`npm run validate:m36 --prefix web` produces a local
Git-ignored synthetic metrics JSON, and GitHub Actions
runs it on each change automatically. It requires **no
camera, videos, manual CSV, annotation, ML retraining or
additional paid hosting**.

Passing these artificial tests is *engineering validation
only*, not independently measured human repetition
counting accuracy. The earlier manual M36 pilot remains
**optional** for a future annotated-human study, not
a prerequisite for the next milestone.

---

### M36 - Measure real rep-count error before any form grading

M34 observed cycle counts worked reasonably well in the
owner's informal webcam test, and M35 now provides
non-diagnostic local feedback. [M36 Rep Counting Validation](docs/M36_REP_COUNT_VALIDATION.md)
adds a **local, consented, manually annotated session audit**:
compare independently counted whole-set repetitions with
the M34 website's observed cycles, for Squat, OHP and
Barbell Row across front/side/diagonal camera views.
It reports per-set MAE/bias/RMSE, exact count agreement,
within-one agreement, explicit tracking abstentions,
and anonymous per-exercise/view aggregates.

Session count agreement **cannot** establish rep-level
precision/recall/F1 or form correctness. This tool
therefore never invents those figures. Optional
descriptive bootstrap intervals resample whole people;
a single-user pilot remains exploratory. All input
sessions are stored locally within Git-ignored
`data/external/` and only aggregate results in
`data/processed/`. No model, original VAL/TEST,
hosted inference or web camera UX is changed.

---

### M35 - Explainable movement feedback, not a form/safety verdict

The owner tested M34 observed cycle counts on their Ubuntu
webcam and reported they work reasonably well (informal
feedback, not annotated rep-count accuracy). M35 adds an
[explainable local observations panel](docs/M35_EXPLAINABLE_MOVEMENT_FEEDBACK.md):
tracking/abstention guidance, movement-cycle summaries and
relative timing variation for the last up to five observed
cycles. The three exercises retain separate camera
instructions; projected angle/time variation are **descriptive
only**, not signs of good/bad form.

The optional six original Fitness-AQA research RF outputs
remain separate and are explicitly identified as
publisher-annotation associations. Low model scores
cannot establish correct technique; Push-up stays
local-only. No new model, threshold tune, dataset, video
upload, feature schema, original M20 VAL/TEST access or
deployment claim is introduced.

---

### M34 - Heuristic observed movement cycles, not validated exercise reps

[M34 local Rep & Motion Analyzer](docs/M34_REP_MOTION_ANALYZER.md)
extends the Squat/OHP/Row camera preview with a conservative
angle-event counter, recent-cycle duration and angular excursion.
Squat uses a 2D knee flexion cycle; OHP uses a 2D elbow cycle;
Barbell Row uses **elbow flexion**, not its potentially stable
torso inclination. Requires stable high-low-high transitions,
two observations per endpoint, tracking continuity and
explicit Reset set; an incomplete movement does not count.

Absolute angle thresholds are experimental detection heuristics,
NOT correct form standards or fitness recommendations. False or
missed cycles are expected for some people/camera views. This
component runs entirely in the browser, retains no source video,
never changes the six opt-in Fitness-AQA models, and has not
been externally validated. Public deployment is separate from CI.

---

### M33 - Local movement feedback for Squat, OHP and Barbell Row

M33 adds camera-plane Squat knee angle, OHP elbow angle and Row apparent torso tilt, plus a local 10-second observed range and camera guidance. These are measurements only, not validated correct/wrong form, injury warnings or rep counting. It does not transmit extra data or change the six opt-in original-source models. See [M33 feedback runbook](docs/M33_LOCAL_EXERCISE_FEEDBACK_UI.md). Production hosting status should be checked independently of CI.

---

### M32 - Frozen Bicep Curl negative-result research card

Actual owner M31 on 90 video-level clips from 15 original TRAIN
people verified exact M30 reproducibility. LOSO balanced
accuracy was **60.00% baseline 264**, versus 56.67%
kinematic 276, 58.89% tracking-proxy 268 and 57.78%
temporal 280. All three descriptive paired subject-bootstrap
LOSO BA intervals overlap zero; none improved bad-label recall.
Every alternative improved video-level accuracy for only **1
of 15 people** in the individual-subject comparison.

[M32 frozen research card](docs/M32_CURL_NEGATIVE_RESULT_CARD.md)
generates an aggregate-only Markdown evidence card from the
local M31 JSON, covering full original CV results, anonymous
per-subject direction counts, paired uncertainty and original
camera-view slices. No original VAL/TEST, source video,
training or web inference changes.

---

### M31 - Bicep Curl participant-paired stability of temporal features

Real owner M30 (15 original TRAIN subjects, 90 Bicep Curl clips):
baseline 264-feature RF yielded **60.00%** Balanced Accuracy
in both GroupKFold(5) and LOSO. Temporal 280 improved Group5
to **62.22%**, but worsened LOSO to **57.78%**, with lower
bad recall. Tracking-proxy 268 and kinematic 276 also
failed to improve LOSO versus baseline.

[M31 subject robustness audit](docs/M31_CURL_PAIRED_SUBJECT_STABILITY.md)
re-runs these fixed original TRAIN-only models on identical
video/participant-heldout folds, exactly verifies the earlier
M30 report, and reports approximate paired 1000-draw
participant-bootstrap differences and anonymous per-subject
better/equal/worse counts. Neither source video, original
VAL/TEST samples nor public backend are accessed.
No new model performance claim is made until locally run.

---

### M30 - Bicep Curl temporal TRAIN research + Push-up local web preview

**Push-up is now selectable on the website** as an on-device,
camera-only research preview showing live 2D elbow angle and
10-second elbow angle range. It **does not** run a Push-up
good/bad model, count reps, send landmark data to the research
API, or claim safe/correct form. The six original Squat,
OHP, Row source-label detector endpoints are unchanged.

[M30 Curl and Push-up technical/Ubuntu guide](docs/M30_BICEP_CURL_AND_PUSHUP_WEB_PREVIEW.md)
also includes a new local-only Bicep Curl temporal feature
experiment using the already downloaded Gym7 archive and
**only frozen M20 TRAIN people**, comparing 264, 268,
276 and 280 feature variants with grouped 5-fold and LOSO
cross-validation. Neither existing M20 VAL/TEST clips nor
private model bundles are accessed. New model metrics
will only be available after the owner runs the commands.

---

### M29 - Freeze independent Push-up video study before collecting new subjects

Real owner M28 TRAIN-only GroupKFold and LOSO both found directional
gains in motion-aware Push-up classifier balanced accuracy, but **every
paired 95% subject-bootstrap difference interval includes zero**.
The camera/pose-quality-only 68-feature model reversed ranking
across validation strategies and all LOSO candidates had the
same bad-label recall as the 64-feature baseline.

[M29 independent cohort preparation](docs/M29_INDEPENDENT_PUSHUP_COHORT_PROTOCOL.md)
predeclares the 64-feature reference, interpretable 76-feature motion
hypothesis and exploratory 80-feature model for future genuinely
new consented adult participants. The new
[`m29_external_pushup_cohort`](scripts/m29_external_pushup_cohort.py)
CLI creates a blank *private* manifest and validates video
deduplication, dual-independent-rater annotations, pseudonymous
consent references and source cohort balance. **No new real videos
have been collected or scored**; no original VAL/TEST video is
accessed and no production inference/model changed.

---

### M28 - TRAIN-only Push-up temporal robustness and feature-family ablation

Owner M27 real-data TRAIN-only GroupKFold on 83 clips and 14 subjects:
64-feature baseline BA **54.18%**, 80-feature temporal BA **58.91%**,
a **+4.73 percentage point** exploratory gain; F1 bad and AP bad
rose slightly, while **recall bad fell from 51.22% to 48.78%**.
No original M20 VAL/TEST subject was evaluated by M27.

[M28 temporal-family robustness](docs/M28_PUSHUP_TEMPORAL_ROBUSTNESS.md)
compares the original 64, motion-kinematics-only 76, pose-quality-proxy
68 and combined 80 feature sets on identical frozen TRAIN clips, both
GroupKFold(5) and Leave-One-Subject-Out, plus descriptive paired
subject-cluster bootstrap intervals. It uses the existing local CSVs,
no new media download, no TEST inspection and no public model release.
The previously consulted M20 VAL is NOT used for adaptive model selection.

---

### M27 - TRAIN-only Push-up motion trajectory hypothesis

Owner M26 original VAL observed: Full paired accepted **23/30**
Push-up clips and achieved **61.36%** BA on 23 paired clips;
Upper expanded accepted **30/30** but achieved **51.89%**
BA on the same clips. On seven upper-only recovered VAL
videos it missed **all three publisher bad labels**.
Full model abstains on those seven videos.
**No current Push-up model is fit for public form-quality advice.**

[M27 frozen TRAIN-only temporal feature study](docs/M27_PUSHUP_TEMPORAL_TRAIN_EXPERIMENT.md)
adds 16 motion trajectory descriptors to the original M23
64-feature upper-body clip schema, and compares identical video
and person-heldout `GroupKFold` subsets on the ORIGINAL M20
TRAIN subjects only. Original VAL/TEST video is untouched by M27.
This is post-VAL exploratory research, not a production model.

---

### M26 - Frozen original M20 validation comparison (no TEST)

Real owner M25 LOSO on original TRAIN: Full paired 54.96%
balanced accuracy, Upper paired 56.39%, Upper expanded 53.57%.
Paired subject-bootstrap interval estimates for both upper
variants' balanced accuracy differences cross zero, so no
form-label accuracy improvement is established. The upper-body
gate remains a **coverage** improvement, not a validated coaching
model.

[M26 predeclared Push-up VAL protocol](docs/M26_FROZEN_PUSHUP_VALIDATION.md)
uses the fixed original M20 participant split, extracts **only
original VAL Push-up video clips**, and compares fixed 200-tree
full-body-trained and upper-expanded RF candidates on identical
paired VAL clips using the same 64-feature schema. Recovered
upper-only clips and gate coverage are reported separately.
No original TEST video, inference endpoint or hosted model is
accessed or updated.

---

### M25 - Push-up TRAIN-only LOSO reliability and subject uncertainty

The owner's M24 grouped TRAIN cross-validation showed increased
Push-up upper-body coverage (83 vs 69 admitted clips) without an
improvement in matched-video Balanced Accuracy: 56.22% full vs
47.69% upper-paired vs 49.24% upper-expanded on 69 matched
held-out TRAIN clips. Upper-expanded improved bad-label recall
(47.06% vs 35.29%) but nearly doubled false-positive bad
predictions (17 vs 8 out of 35 good clips).

[M25 original TRAIN participant LOSO and cluster uncertainty](docs/M25_PUSHUP_LOSO_ROBUSTNESS.md)
repeats an individual-subject-heldout comparison and estimates
approximate paired subject-bootstrap intervals on fixed out-of-fold
predictions. Data provenance checks and privacy-safe aggregate
output are built in. Original M20 validation and final TEST
remain untouched, and no new Push-up model is deployed.

---

### M24 - Compare new Push-up RFs on exactly matched TRAIN-only clips

Real M23 owner extraction recovered **14 previously rejected front-view
Push-up clips** in original TRAIN: **69/84** accepted using full-body
admission, versus **83/84** using upper-body admission, with no file
failures. Both original publisher good/bad labels gained seven
front-view clips.

[M24 participant-heldout paired RF evaluation](docs/M24_PUSHUP_GROUPED_CV.md)
compares original full-admitted video features against upper-admitted
video features on **identical held-out common clips**, using 5-fold
subject-separated cross-validation exclusively within the original
frozen TRAIN subjects. Expanded upper-gate training also uses
newly recovered TRAIN-only clips; their extra held-out predictions
are reported separately instead of changing the common denominator.

This is an offline analysis and **does not touch the original M20
validation/TEST scoring or deployed Fitness-AQA detectors**.

---

### M23 - Frozen Push-up participant split and new upper-body clip schema

Owner M22 TRAIN ablation: baseline original full-body accepted
**24/513 frames, 0/15 rejected clips**; baseline upper-body accepted
**258/513, 14/15**, with heavy/lower-threshold detection providing only
modest additional frame coverage.

[M23 frozen TRAIN-only Push-up features](docs/M23_TRAIN_ONLY_PUSHUP_SCHEMA_PILOT.md)
builds paired full/upper admission data using a NEW
`gym7-pushup-upper32-clip64-v1` schema with masked invisible joints
and the original M20 seed-42 frozen TRAIN participant membership.
Validation and TEST participant media are not decoded.
**No M23 Push-up model is trained or deployed yet**.

---

### M22 - Pairwise Push-up detector and upper-body gate ablation

Real owner M21 TRAIN-only diagnostics recorded **262/513** frames rejected
for complete-side limb visibility, **227/513** with no detectable pose,
and **24/513** accepted. Ankles and knees were most often invisible.

[M22 TRAIN-only ablation](docs/M22_PUSHUP_TRAIN_GATE_DETECTOR_ABLATION.md)
compares three MediaPipe pose-detection configurations against the same
original complete-side and experimental upper-body admission gates on
identical sampled frames. It is diagnostic-only, includes synthetic
holdout isolation tests and does not retrain or deploy models. Independent
validation and TEST camera videos remain untouched.

---

### M21 - Diagnose front-view Push-up coverage loss (local, TRAIN only)

The owner extracted **266/294** Gym7 videos. All **28 rejected**
clips came from **front-view Push-up** (14 original good + 14 original
bad), where coverage was **20/48 (41.7%)**; all Curl and other
Push-up camera views had 100% pose clip extraction coverage.

Use the new [M21 local front camera pose audit](docs/M21_PUSHUP_FRONT_COVERAGE_AUDIT.md)
to measure whether the failures stem from no detected person, the
original same-side limb visibility rule, degenerate torso scale or
too few valid frames. The audit **decodes only original TRAIN-subject
rejected clips**, not VAL/TEST subjects, and does not change the deployed
six source-RF models or the M20 baseline.

---

### M20 - additional gym exercises (local research pipeline, not yet deployed)

The scientific aim is **exercise movement-quality research**, not merely
drawing pose skeletons. The deployed 6 RFs perform real source-label
association for Squat, OHP and Row; their outputs are NOT independently
validated technique-correctness or injury-safety judgements.

A new **opt-in local** research pipeline downloads one 2.9 GB publisher
Mendeley Gym7 ZIP, builds owner-reviewable subject labels, extracts clip
features one video at a time, and trains grouped **Push-up and Bicep Curl**
good/bad *video-level* Random Forest pilots. Original per-video good/bad
labels are kept separate from the original named Fitness-AQA error labels.
TEST is locked by default, training files are ignored by Git, and new
models are NOT silently offered in the live single-frame scoring API.
This repository cannot write directly to the owner's Linux computer.

[Run the M20 local download, extraction and training guide](docs/M20_GYM7_LOCAL_TRAINING.md).

---

### M19 mobile cameras, premium studio UI and new dataset research

The mobile site now offers Front / Selfie and Rear / Observer camera
selection; mirror on/off flips **only the front preview**, not the numeric
MediaPipe landmarks supplied to the experimental source-label models.
Switching physical cameras stops the prior stream and resets research
inference consent. A full responsive movement-studio redesign highlights
camera controls, joint visibility and model status on mobile.

Tracked human-readable docs and source use simple ASCII hyphen instead of
Unicode en or em dashes. Valid CLI `--flags` and CSS `--properties` are
preserved. CI enforces this convention.

[M19 mobile camera and UX acceptance](docs/M19_MOBILE_CAMERA_WEB_REFRESH.md)
and [additional exercise dataset research](docs/M19_EXERCISE_DATASET_EXPANSION.md).
**No fourth exercise or newly trained model has been deployed.**

---

### M18 zero-cost deployment preparation (no paid Render plan)

The optional **free-tier** Render Blueprint (`render.yaml`) creates NO
persistent disk and uses Render's free 0.1 CPU / 512 MB instance. Owner-trained
six private `.joblib` artifacts are **never** included in public Git. The
owner can upload only those six files to a separate **private** Hugging Face
model repository; the backend fetches exactly six SHA256-verified files at
one pinned commit before trusting any model. No secrets are committed.

**No paid resource or cloud inference has been activated.** Render free
services sleep when idle and can run out of RAM; only a successful
real-owner 6/6 `/health` + `/ready` check verifies deployment.
Browser/simulation research accuracy still has not been established.

[Free Render setup, local private upload script, six-model bootstrap and
Vercel acceptance](docs/M18_RENDER_PRIVATE_MODEL_HOST.md).

---

### M18: six-model deployment readiness gate (pending owner cloud artifacts)

The Web v2 frontend and optional Python scoring API are integrated, but the
deployed Vercel site is **not yet connected to a publicly provisioned six-model
backend**. M18 now requires all six original-source research RFs before
enabling the user opt-in; `/ready` returns 503 when any is missing, and a
read-only HTTPS preflight checks exact model availability without transmitting
landmarks. Validation errors do not echo pose coordinates. This does **not**
fix the unvalidated browser/source pose distribution shift.

```bash
python -m scripts.check_web_inference_deployment --api-url http://localhost:8000
```

[Follow the M18 deployment acceptance and model-provisioning guide](docs/M18_SIX_MODEL_DEPLOYMENT_ACCEPTANCE.md).

---

### Web v2: opt-in six source-label RF inference API

The deployed browser camera and pose visibility workbench now has an
optional **real Python inference backend** integration for all six
original-source models (two Squat, two OHP, two Barbell Row). Users
must explicitly opt in before their **33 pose landmarks** leave the
browser; raw camera frames never leave. The backend reuses exact
existing `ModelRegistry` feature generation and thresholds, and
returns genuine experimental source-label scores or `ABSTAIN`.
There are **no invented scores or medical / safe-form judgments**.

```bash
python -m pip install -e '.[webapi,dev]'
python -m pytest -q
# Models stay on the owner's computer or separately provisioned private host.
POSE_TRUST_LOCAL_ARTIFACTS=1 POSE_MODEL_ROOT=. \
POSE_WEB_ORIGINS=http://localhost:5173 \
uvicorn gym_pose.web_api:app --host 127.0.0.1 --port 8000 --no-access-log
# In separate terminal:
cd web
VITE_INFERENCE_API_URL=http://localhost:8000 npm run dev
```

**Vercel static frontend does NOT magically contain the six local joblib
bundles.** Backend hosting with private model provisioning and a
build-time `VITE_INFERENCE_API_URL` is required before deployed
public website users can see real scores. Otherwise it stays a
browser-only visibility demo with model inference unavailable.
CORS, no-pose abstention, request constraints and tests are included;
public backend abuse protection and cross-MediaPipe task/solution
parity are still open research/ops requirements.

[How to provision the model API and connect Vercel](docs/WEB_REAL_INFERENCE_DEPLOYMENT.md).

---

### Web MVP: open camera in your browser (deployable on Vercel)

A new `web/` Vite project brings the privacy-conscious pose
**visibility viewer** to a browser, with no Python install, repo clone
or model pickle required for end users:

```bash
cd web
npm install
npm test
npm run dev
# Open http://localhost:5173
```

It supports webcam pose skeleton, Squat/OHP/Barbell Row exercise
selection, per-source-task required-joint visibility, a local camera
Start/Stop control and responsive desktop/mobile UI. Source videos
or camera frames are **not uploaded**.

**Important:** Six publisher source-error models are still only
available in the owner's trusted **local Python research registry**.
The web demo shows each RF as `NOT DEPLOYED`. No model probabilities
are fabricated; an explicit backend+source-parity validation or
browser ONNX parity is needed before serving those actual classifiers.

Vercel deployment: import GitHub repo and configure **Root Directory
`web`**, Vite `npm run build`, output `dist`. Browser webcam
requires HTTPS/localhost and permission.

[Web MVP setup, deployment and privacy boundaries](docs/WEB_MVP_VERCEL_DEPLOYMENT.md).

---

### Research release source integrity - M17B

The owner confirmed **218 tests passed** and all **six** trusted local
original-source error models loaded on M17A. M17B checks that
the source release has the six correct versioned task contracts,
the original TRAIN/VAL research reproduction scripts, and no newly
tracked private Fitness-AQA media or derived features. Run:

```bash
python -m pytest -q
python -m scripts.audit_research_release
```

The new audit **does not** load model pickle files, access a webcam
or evaluate publisher TEST. The repo still tracks two **historical**
notebook assets (`models/squat_form_rf.joblib` and `models/yolov8s.pt`);
they are disclosed as legacy warnings, not confused with the six
newly trained source-label RF artifacts or silently deleted.

[Release checklist, older OHP PR reconciliation, and handoff](docs/M17B_RELEASE_AUDIT_AND_HANDOFF.md).

### Research release candidate - M17

[Read the M17 experimental results, original source-label tables,
real webcam runtime evidence, privacy scope, historical TEST exposure,
and explicit release readiness/blockers](docs/M17_RESEARCH_REPORT_AND_RELEASE_READINESS.md).

```bash
python -m pip install -e '.[dev]'
python -m pytest -q

# Status-only; no joblib loading or camera:
python -m scripts.check_fitness_aqa_model_registry

# Only run with artifacts that YOU trained and trust (joblib uses pickle):
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts

# Optional local webcam demo, no image recording:
python -m pip install -e '.[camera,dev]'
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 --trust-local-artifacts
```

Source code/tests do not bundle Fitness-AQA original videos/images,
raw features, or the owner's local trained joblib artifacts. A historical
pilot previously inspected some publisher TEST clips, so **do not claim
the full project history kept original TEST pristine**. Independent
athlete-level or clinical claims have NOT been established.

---

# Historical implementation and milestone notes

## Milestone 13A: Overhead Press pose extraction pilot

OHP now has a **separate versioned 111-feature schema** (103 compatible
MediaPipe pose columns plus 8 OHP-specific upper-body geometry features),
with critical visibility checks for elbows and wrists.

```bash
# First 8 source train clips + 8 source validation clips, source TEST LOCKED:
python -m scripts.extract_fitness_aqa_ohp --split train-val --max-videos-per-split 8

# Once source video/prediction coverage has been inspected:
python -m scripts.extract_fitness_aqa_ohp --split train-val --max-videos-per-split 0
```

Outputs stay under `data/features/fitness_aqa_ohp/` (ignored by Git).
No OHP model performance claims until full train/val extraction and evaluation.
See [M13A OHP guide](docs/FITNESS_AQA_OHP_M13.md).

---

## Milestone 16D: scored-only RF runtime profiling

The owner's M16C webcam test admitted **47/180 Squat frames** and
generated **94 source-label scores**, up from zero in the initial
run. Both ankles are now occasionally visible, but the right ankle
remains below the source pose-quality cutoff in **131/180** frames.
The old average model-stage timing included many abstained frames,
so it was not a measure of real RF execution speed.

M16D now measures **separate** task-level successful inference
and abstention latencies, plus scored/abstained source-label
fractions per exercise/target, without collecting frames,
person data, source-label probabilities or changing model gates:

```bash
python -m pytest -q
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

Inspect `diagnostics.timing_ms.successful_model_task_inference`,
`diagnostics.timing_ms.abstained_model_task_call` and
`diagnostics.source_label_score_admission_by_target`.
Model score admission is **not** model accuracy or form-safety
assessment. This brings the local research viewer closer to M17
release/reproducibility work, not a production fitness coach.

[Milestone 16D scored inference performance audit](docs/M16D_SCORED_INFERENCE_LATENCY.md).

---

## Milestone 16C: webcam framing guide for low-visibility joints

On the owner's first 120-frame Squat webcam diagnostic, MediaPipe detected
a pose in **120/120 frames**, but both ankles and the right knee were
below visibility>=0.50 in **every** frame. The original source-model
quality gate correctly abstained on all 240 Squat detector attempts.
This is a **framing/joint visibility** blocker, not a model load failure.

The viewer now has a separate, display-only **joint visibility side
panel**. It shows which joints are `LOW`, `VISIBLE`, or `NO POSE`
so a user can move back and include their whole body without changing
the original 6 model schemas, gates or thresholds:

```bash
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

Use `--no-framing-guide` to hide the panel. No webcam frames are
recorded or uploaded. MediaPipe visibility is **not** a medical
assessment or confirmation of safe/correct exercise.

[Milestone 16C camera framing guide](docs/M16C_CAMERA_FRAMING_GUIDE.md).

---

## Milestone 16B: live pose abstention and performance diagnostics

The owner's first 60-frame Squat webcam run initialized correctly and
loaded all six source-error RF models but returned **0 source-label
scores / 120 abstentions** (two task scores per camera frame).
That alone did not distinguish *no MediaPipe pose* from unreliable
required joints or an inference error.

M16B reports **aggregate, privacy-safe** reasons and mean/p95
pose/model/frame timing + observed FPS; it keeps all quality gates and
thresholds unchanged:

```bash
python -m pytest -q
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 120 \
  --trust-local-artifacts
```

The viewer now displays no-pose and missing-joint hints and prints a
`diagnostics` summary at exit. No webcam video, pose coordinates,
per-frame predictions or publisher TEST data are stored. This is
diagnostics only, not a technique or safety coach.

[Milestone 16B diagnostics guide](docs/M16B_LIVE_POSE_DIAGNOSTICS.md).

---

## Milestone 16A: Local real-time *research* viewer (experimental)

The owner's M15C local run confirmed six source-label error-model
artifacts are loaded correctly (Squat, Overhead Press, Barbell Row).
However, source-validation precision/recall and pose acceptance remain
too weak to claim reliable technique or injury-prevention feedback.

An opt-in OpenCV webcam **research viewer** displays the manually
selected exercise's pose and two original-source *annotation scores*.
It never turns scores into coaching advice, safe/correct labels,
automatic movement alerts, rep counting, uploads or recordings.

```bash
python -m pytest -q
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
python -m scripts.run_fitness_aqa_research_viewer --exercise squat --camera 0 --trust-local-artifacts
```

Use keys `1` Squat, `2` OHP, `3` Barbell Row, `q` to exit.
Models must be your **own trusted local joblib** files (pickle can run
code). Missing or low-quality poses produce visible **ABSTAIN**, not
'correct form'. OpenCV GUI and MediaPipe camera extras are required.

[Milestone 16A real-time research viewer](docs/M16A_REALTIME_RESEARCH_VIEWER.md).

---

## Milestone 15C: Squat original source-error RF registry

The two genuine source-error targets `error_knees_forward` and
`error_knees_inward` are separate from the notebook's binary
`correct/incorrect` model. M15C packages new **103-feature Squat
source-label RFs** using source TRAIN video-group calibration and
publisher VAL reporting, without touching publisher TEST:

```bash
python -m scripts.train_fitness_aqa_squat_source_errors --seed 42 --trees 300
# Only load your OWN trusted local artifacts:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

Original source TRAIN/VAL pose CSVs are reused; no ZIP/media extraction.
Both Squat source-error slots can become available alongside OHP/Row,
with an explicit experimental-only result and fail-closed abstention.
Six software model slots do **not** mean the exercise coach is ready.

[M15C Squat source-error integration guide](docs/M15C_SQUAT_SOURCE_ERROR_REGISTRY.md).

---

## Milestone 15B: OHP registry feature adapter restored

M15A could already load both local Barbell Row models, but initially
reported `feature_schema_not_integrated` for OHP because its original
30-feature source module lived only on unmerged PR #18. M15B now
integrates those **exact** original OHP elbow/knee schemas on the
current main-based branch (no re-training or ZIP extraction required):

```bash
python -m pytest -q
python -m scripts.check_fitness_aqa_model_registry
# Only for your own trusted locally trained joblib files:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

After trusting the original locally created models, the two OHP and two
Barbell Row tasks can report `loaded_experimental_only`. Squat original knee-error model contracts are supplied by M15C after the user's local TRAIN-only run. This is
experimental source-label inference, **not** a trustworthy realtime
exercise-safety coach.

[Milestone 15B OHP adapter details](docs/M15B_OHP_REGISTRY_INTEGRATION.md).

---

## Milestone 15A: Offline multi-exercise model registry

An experimental **fail-closed** registry can load the owner's locally
trained OHP static and Barbell Row torso4 static RF artifacts, verify
original source target/feature schema/model class/operating threshold,
and score source labels without access to publisher TEST media:

```bash
python -m scripts.check_fitness_aqa_model_registry
# Only for your OWN trusted local joblib artifacts (pickle deserialization):
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

The Squat source knee-error slots are intentionally not automatically
mapped to the legacy generic correct/incorrect classifier. Original OHP
30-feature source schemas are integrated by M15B; the outstanding research
PR #18/#19 histories still require separate reconciliation before final release. Missing model/pose yields
`abstain`, not a "good exercise" classification. This is not a webcam
coach or medical safety assessment.

[Milestone 15A registry contracts](docs/M15_UNIFIED_MODEL_REGISTRY.md).

---

## Milestone 14D: Barbell Row static RF benchmark

Barbell Row full source train/validation now retains around 70% of
original per-task labeled images using the new torso4 20-feature
shoulder/hip geometry. The new benchmark trains a **separate** Static RF
for each publisher error label, selecting threshold from original
**TRAIN candidate-prefix groups only**:

```bash
python -m scripts.benchmark_fitness_aqa_barbell_row_torso4 \
  --seed 42 --trees 300 --bootstraps 500
```

It reads existing local torso4 CSVs (no ZIP/MediaPipe), rejects pilot,
split/prefix leakage and mismatched manifests, reports acceptance and
source-validation AP/precision/recall/FPR, and stores RF models under
ignored `artifacts/fitness_aqa_barbell_row_torso4_rf/`. Source TEST
image contents are never used. Candidate prefix is **NOT verified
video/subject provenance**, and 4-joint geometry does not measure spine
curvature.

[Milestone 14D training/evaluation protocol](docs/FITNESS_AQA_BARBELL_ROW_RF_M14D.md).

---

## Milestone 14C: Barbell Row shoulder-hip 20-feature extraction

The source TRAIN/VAL PR #22 audit established that requiring knees
and ankles was a major cause of the earlier **4-5%** Barbell Row pose
coverage. On equal-size label-stratified audit subsets, a strict
shoulder+hip four-joint gate admitted approximately **68-75%** of
images per task/split. These sample rates are **not** model accuracy.

A new independent `barbell-row-torso4-image-20-v1` schema uses only
the four reliable shoulders/hips as both feature inputs and quality
gate (>=0.50):

```bash
# Source JSON + ZIP already downloaded in M12; small smoke test first
python -m scripts.extract_fitness_aqa_barbell_row_torso4 --max-images-per-task-split 12
# Full original TRAIN+VAL, never publisher TEST
python -m scripts.extract_fitness_aqa_barbell_row_torso4 --max-images-per-task-split 0
```

The new output directory is `data/features/fitness_aqa_barbell_row_torso4/`
and the original 32-feature pipeline remains intact. This is an
**image-label classifier foundation**, not direct lumbar-spine
measurement or reliable real-time coaching.

[Milestone 14C torso4 extraction methodology](docs/FITNESS_AQA_BARBELL_ROW_TORSO4_M14.md).

---

## Milestone 14B prerequisite: Barbell Row Pose Coverage Audit

The original source Barbell Row TRAIN/VAL image extraction succeeded,
but strict 8-joint MediaPipe pose coverage was only **4-5%** across
the two tasks. The source lum-bar validation subset retained only
**8 positive error images**. Training a classifier now would risk
substantial source selection bias.

```bash
python -m scripts.audit_fitness_aqa_barbell_row_coverage --images-per-task-split 180 --seed 42
```

This is a train/validation-only, source-label-stratified diagnostic
comparing 8-joint strict/relaxed gates against hypothetical 6- and
4-joint gates on the **same source photos**. It reports per-joint
failures, positive/negative pose admission, and missing-pose counts;
no test photos decoded, no model retrained, no source archive edits.

[Barbell Row pose quality diagnostic](docs/FITNESS_AQA_BARBELL_ROW_POSE_COVERAGE_M14.md).

---

## Milestone 14A: Barbell Row image pose extraction

The publisher Barbell Row dataset is **images** with two separately
split binary error targets, `torso_angle` and `lumbar` (not video
time intervals). A dedicated 32-feature, quality-gated image pose
extractor processes original TRAIN/VAL images only:

```bash
# Source JSONs / images already verified in milestone 12.
python -m scripts.extract_fitness_aqa_barbell_row --max-images-per-task-split 12

# After pilot success, complete publisher TRAIN+VAL images for each task:
python -m scripts.extract_fitness_aqa_barbell_row --max-images-per-task-split 0
```

Outputs are local and ignored under `data/features/fitness_aqa_barbell_row/`.
**Publisher TEST pixels are never decoded.** The image-ID prefix is an
unverified clip identifier, and 8-joint MediaPipe geometry cannot directly
measure lumbar-spine curvature.

[Milestone 14 Barbell Row extraction guide](docs/FITNESS_AQA_BARBELL_ROW_M14.md).

---

## Milestone 12: Fitness-AQA multi-exercise dataset foundation

The research dataset foundation now supports the original publisher's **three
exercise subsets** without assuming interchangeable labels or globally
correct form:

- **Squat:** named knee error intervals from video ZIP (existing baseline).
- **Overhead Press (OHP):** elbow/knee error intervals from its own video ZIP.
- **Barbell Row:** torso-angle and lumbar **image** labels (0/1), with **separate
  publisher train/val/test splits per error target**.

```bash
# Validate downloaded small publisher JSON metadata, even before ZIP download:
python -m scripts.inspect_fitness_aqa_multi --exercise ohp --skip-archive
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row --skip-archive

# When archives are downloaded, verify exact file-stem IDs WITHOUT unzipping:
python -m scripts.inspect_fitness_aqa_multi --exercise ohp
python -m scripts.inspect_fitness_aqa_multi --exercise barbell_row
# or --exercise all after downloading all three datasets
```

The manifest saves locally under `artifacts/fitness_aqa_dataset/`, never
uploads source media, detects missing/duplicate ZIP keys and warns if the
Barbell Row image's two target-specific splits conflict. **No new OHP/Row
accuracy claims or model training** occur in this milestone; the publisher
test IDs are checked for integrity only. Source splits are not verified
independent by athlete.

[Milestone 12 datasets, exact download commands and research protocol](docs/FITNESS_AQA_MULTIBENCHMARK_M12.md).

---

## Milestone 2: Data provenance and subject-independent evaluation

The new commands validate a consented image manifest, extract traceable features, and
benchmark a Random Forest on **held-out subjects** with metrics per camera view.
The code ships without a real dataset or fabricated model accuracy.

```bash
gym-pose manifest-template
# Fill data/manifest.csv using real subject/session/view IDs and real image paths.
gym-pose validate-manifest
gym-pose extract-manifest
gym-pose benchmark
```

See [Milestone 2 protocol, outputs and data policy](docs/MILESTONE_2.md).
A random-frame validation split is **not** evidence of generalization across subjects.

---

## Milestone 1: Modular baseline (new)

The original notebook below remains available as a historical baseline. A new
installable package adds named 103-feature extraction, visibility checks, resettable
probability smoothing, model schema validation, tests and CLI workflows.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera,dev]'
python -m pytest -q

# Capture images for each class, then train an explicitly versioned baseline:
gym-pose capture --label correct --count 100
gym-pose capture --label incorrect --count 100
gym-pose extract
gym-pose train
gym-pose webcam
```

**Evaluation warning:** `gym-pose train` uses a random frame split for an exploratory
baseline only. For independent group evaluation, collect truthful `subject_id`/
`session_id` metadata and use `--group-column subject_id`. No real-world
form accuracy or rep-counting claims are made. Existing old joblib bundles are not
schema-versioned and must be retrained for the new CLI.

See [Milestone 1 setup, scope and limitations](docs/MILESTONE_1.md).

---

This project is structured for building and running a real-time gym form detection pipeline using:

- MediaPipe Pose for landmark extraction
- Scikit-learn for form classification
- OpenCV for webcam capture and live overlay
- YOLO weights available under `models/` for related detection experiments

The current notebook workflow supports one exercise at a time (default: squat), with two labels:

- `correct`
- `incorrect`

## Project Structure

```text
.
├── data/
│   ├── raw/                       # Captured training images by exercise/label
│   └── features/                  # Generated CSV feature files
├── models/                        # Trained classifiers and YOLO weights
├── notebooks/
│   └── gym_pose_detection.ipynb   # Main end-to-end notebook
├── README.md
└── requirements.txt
```

## Setup

Use your existing conda env or create a new one.

### Option A: Conda (recommended)

```bash
conda create -n gym_pose python=3.10 -y
conda activate gym_pose
pip install -r requirements.txt
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Workflow

Open:

- `notebooks/gym_pose_detection.ipynb`

Run cells in order:

1. Imports and path setup
2. Feature utility functions
3. Dataset capture
4. Feature CSV generation
5. Classifier training
6. Real-time webcam inference

### Data Collection

Capture images per label from webcam:

```python
collect_samples("correct", target_count=250)
collect_samples("incorrect", target_count=250)
```

Captured images are saved under:

- `data/raw/squat/correct`
- `data/raw/squat/incorrect`

### Build Features and Train

```python
build_feature_csv()
train_form_classifier()
```

Generated artifacts:

- Feature CSV: `data/features/squat_features.csv`
- Trained model: `models/squat_form_rf.joblib`

### Run Real-Time Form Detection

```python
run_realtime_form_check()
```

Controls:

- Press `q` to stop webcam inference.

## Labeling Guidance

To improve model quality:

- Keep camera angle consistent during one exercise session.
- Record different body positions, clothing, and lighting.
- Ensure each class has similar sample counts.
- Start with one exercise before adding more.

## Troubleshooting

- If webcam does not open, check camera permissions and `camera_index`.
- If MediaPipe import fails, verify package in active environment:

```bash
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
```

- If notebook paths fail, make sure you run from the repository root and use the updated notebook in `notebooks/`.