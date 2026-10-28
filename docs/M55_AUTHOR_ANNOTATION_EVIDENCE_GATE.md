# M55 - Author evidence, annotation semantics and claim gate

## Scope: public source review + private M54 aggregates, NO TEST

M55 documents what the original RepCount authors actually published
and distinguishes it from what the owner verified in the previous M54
TRAIN-only archive audit. It does NOT contact the authors, obtain a
dataset license, authenticate owner archive bytes, identify subjects,
relabel the original GT, inspect any video, or evaluate official TEST.

### Tier A: publicly documented by original authors

1. ShanghaiTech's [original dataset home](https://svip-lab.github.io/dataset/RepCount_dataset.html)
   states RepCount-A comprises 1,041 videos sourced from YouTube,
   that the labels include total repetition **count** and cycle
   **location along the time axis**, and that the full Part-A has
   TRAIN/VALID/TEST. The original project reads only TRAIN.
2. [Hu et al., TransRAC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html),
   Dataset annotation section: individual videos were assigned to
   two volunteer annotators; the start/end of each action cycle
   were labeled; annotation pairs were cross-validated and
   differences of MORE THAN ONE total count triggered review/
   revision. This is an author-described **protocol**, NOT proof
   that any particular owner's video had perfect annotation or
   anatomically standardized bottom/extension landmarks.
3. [Original TransRAC implementation](https://github.com/SvipRepetitionCounting/TransRAC)
   describes dataset use, cite instructions and the distinction
   between retrained comparisons of different research models.
   The presence of code/checkpoints and a dataset download is
   not blanket authorization to redistribute original YouTube
   source videos or use them commercially.

### Tier B: owner-verified M54 report (TRAIN-only)

The owner executed M54 against their local RepCountA.tar.gz and
reported:
- Exactly **101 TRAIN Squat annotation rows**, **zero zero-count
  Squat clips**. This is the original TRAIN Squat subset, NOT
  necessarily all Part-A actions, nor official publisher TEST.
- Original filename-hash development 20 clips: **312 GT cycles**,
  157 adjacent pairs with zero gap, 135 with positive gap,
  min/max annotated span 24/158 frames, 0 clips whose first
  cycle starts at frame zero.
- Original filename-hash replication 20 clips: **209 GT cycles**,
  155 adjacent pairs with zero gap, 34 with positive gap,
  min/max span 24/166 frames, 1 clip whose first cycle
  starts at frame zero.
- M54 original TRAIN CSV SHA matched M48 and passed the M49
  source/Git gate; TEST/VALID not opened, no video extraction.

M55's script verifies the owner's existing M54 JSON against M48
using exactly those aggregate invariants; it does not independently
verify a publisher archive checksum against an official release.

### Tier C: explicitly unresolved

| Question | Verified? | Why it matters |
| --- | --- | --- |
| Owner TAR byte-identical to publisher original | NO | M51 checks only against another owner-supplied copy |
| Legal right to redistribute original YouTube videos | NO | Publication of videos/screenshots can require additional permissions |
| Commercial training or deployment permission | NO | Research citation does not grant a commercial license |
| START/END mapped to knee flexion bottom or full extension | NO | General temporal action-cycle intervals are NOT anatomical phase labels |
| Original frame index end semantics (inclusive/half-open) | NO | Frozen parser convention is not established author policy |
| Participant/session/camera disjointness | NO | Filename hashes do not guarantee subject independence |
| Zero-rep false-positive performance | NO | 0/101 Squat TRAIN rows have count=0 |
| Official TEST generalization / new M45 F1 | NO | TEST remains untouched |

**Interpretation of gap geometry:** M54 identifies positive and
zero frame gaps between adjacent original annotated intervals. This
does not imply annotation error, exact transition phase, inter-rep
rest duration, or which physical posture occurs at GT boundaries.
Endpoint comparisons to 155-degree M45 returns therefore remain
**time matching to a publisher interval end**, not a guarantee of
matching a biomechanical movement phase.

## Reproducible local gate

From repository root:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m55_author_annotation_evidence_review.py -q

python -m scripts.m55_author_annotation_evidence_review \
  --m54-report data/processed/m54_train_annotation_provenance.json \
  --m48-lock data/processed/m48_train_evidence_freeze.json \
  --report data/processed/m55_author_evidence_review.json
~~~

The script uses ONLY the aggregate M54 report and M48 lock; does
not read the TAR/CSV/video, does not fetch online and refuses
overwrite. It writes a Git-ignored aggregate author-evidence
summary, no subject, video, filename or GT interval data.

## Author clarification questions (draft, not sent)

Use an original author contact channel, if one is publicly listed,
to seek a definitive answer in writing; do not invent recipients:

1. "For RepCount-A, do the CSV L1/L2... frame boundaries identify
    the full visual action cycle, and are endpoints inclusive or
    exclusive? Are starts/ends standardized to any particular
    exercise phase such as full extension or maximum knee flexion?"
2. "Is an official SHA256 checksum available for the Part-A TAR or
    original TRAIN annotation CSV, including release/version?"
3. "What are the reuse terms for original YouTube video contents
    in academic experiments, website screenshots, paper
    supplements and commercial product demonstrations?"
4. "Do published annotations include participant/session/camera
    grouping sufficient to establish subject-disjoint evaluation?"

Until independently documented: keep publisher TEST unrun,
do not redistribute dataset media, and avoid phase-specific
accuracy and subject-generalization claims.

## Next scientific gate

A later M56 can design an AUTHORIZED, ONE-SHOT official TEST
protocol only after provenance/licensing and endpoint semantics
are resolved to an acceptable degree. It should preregister the
exact TEST file manifest, evaluation tolerance, metric
definitions, zero-GT handling, subject independence caveats,
pose model version, inference cadence and locked thresholds,
without using TEST to tune M45. M55 does not authorize execution.
