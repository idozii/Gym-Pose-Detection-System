# M37 - Automated real-video Squat rep-event evaluation

## Scope and verified upstream references

This milestone advances the **no-manual-annotation** path requested by
the owner. It does **not** train or tune another Random Forest or
claim good/bad technique classification.

The original RepCount authors provide **RepCount-A**, a repeating-action
dataset with pre-existing action counts and paired frame positions for
the beginning/end of each rep:

- Official research project: <https://svip-lab.github.io/dataset/RepCount_dataset.html>
- Official TransRAC code (CVPR 2022): <https://github.com/SvipRepetitionCounting/TransRAC>
- Explicitly identified **third-party academic-use video mirror**:
  <https://huggingface.co/datasets/momo321654/RepCountA-videos>

The mirror states that there are 1,041 RepCount-A videos and
per-split JSON annotation files, with `video`, `action_type`,
`count` and flat `cycle_bounds` frame-index arrays, plus individual
MP4s. It attributes files to the official RepCount release.
These provenance claims have **not been independently byte-verified
against the original authors' archive** in this project.

M37 currently benchmarks only records whose source action is
`squat` or `squatting`. It does **not** assert that the RepCount
classes `front_raise`, `rowing` or `pull_up` are equivalent to
Gym Pose Lab's barbell Overhead Press or Barbell Row. Those remain
unvalidated and require a compatible independently annotated dataset.

### Small default subset rather than ~9.7GB full archive

The command downloads `validation_annotations.json` and only
**three deterministic Squat videos** from RepCount-A validation
by default. You can adjust the clip limit to 1-20; it never
downloads entire splits automatically. Clips are taken in sorted
source filename order, without choosing them based on outcomes.

Video and extracted landmark artifacts stay in local ignored
`data/external/m37_repcount/`. Raw landmark traces are temporary:
after replay, they are deleted and only aggregated metrics are
kept in `data/processed/m37_repcount_squat_event_results.json`.
Original Fitness-AQA and Gym7 M20 TEST are **never opened**.

## How the automatic benchmark works

1. Fetch a small selected set of **already annotated** source clips
   with `huggingface_hub`. The code checks action type, number of
   annotation pairs, non-overlapping integer frame intervals,
   original clip names and annotated cycle totals before use.
2. Decode each source MP4 via OpenCV. Reject bad FPS, oversized or
   oversized-duration clips, malformed pose samples, incomplete
   videos and annotations whose frame index exceeds the decoded
   clip length.
3. Extract MediaPipe **Solutions Python** body landmarks in
   chronological source order, targeting approximately **400ms**
   sampling to reflect the original M34 JS heuristic. Missing
   poses remain explicit `null` observations.
4. Replay those timestamped landmarks through the
   **existing unchanged browser `RepCycleCounter`** in
   `web/scripts/m37_eval_repcount_trace.js`, rather than
   rewriting its state machine in Python.
5. Detect the exact timestamps at which M34 increments the
   observed-cycle count. Match them to the source cycle **end**
   frame at a fixed maximum time discrepancy of
   `min(2500ms, max(750ms, 35% of annotated cycle duration))`.
   A source event may match at most one prediction.
6. Report event TP/FP/FN, precision, recall and F1 on the
   selected video subset, plus per-video count MAE,
   exact count fraction, within-one fraction, source
   pose-visibility sample coverage, number excluded
   and causes of exclusion. If **zero real clips** are
   successfully processed, M37 **refuses to produce a
   successful accuracy report**.

Important: unlike M36's artificial landmark regression tests,
M37 will only report real-video metrics **after actually decoding
third-party videos and replaying M34 outputs**. All Python/Node
GitHub CI tests use *synthetic local fixture files*, and CI
does not contact or download the dataset.

## M37 source access diagnostic (HTTP 404)

In October 2026 the owner encountered a **real 404** when the
Hugging Face client requested
`momo321654/RepCountA-videos/validation_annotations.json`.
A previously indexed Hugging Face dataset card described
that file, but old search results do not establish that the
repository and individual video/annotation files are still
accessible. A 404 might mean removed/moved, private, or
a renamed file; the error alone cannot distinguish these
causes. **Do not assume logging in will resolve it.**

The updated M37 checks **live repository metadata and exact
file paths before downloading any video**:

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate
python -m scripts.m37_repcount_video_benchmark --check-source
```

If the check reports inaccessible repository, no clip is
downloaded and no accuracy report is produced. Do not
try random URLs or unverified replacements; retrieve
the dataset via the official authors' permission-controlled
distribution if eligible:
<https://svip-lab.github.io/dataset/RepCount_dataset.html>.

If you already have the legally obtained RepCount-A files in
**the exact mirrored layout** with source frame annotations,
a one-command local fallback avoids Hugging Face entirely:

```text
/path/to/authorized-repcount-a/
    validation_annotations.json
    validation/
        stuX_Y.mp4
        ...
```

```bash
python -m scripts.m37_repcount_video_benchmark \
  --local-source /path/to/authorized-repcount-a \
  --max-videos 3
```

The paths above are schematic, **not an assertion the
dataset already exists on your machine**. The command
requires publisher-compatible original JSON annotation
fields (video, action_type, count, cycle_bounds frame
pairs) and paired files with exactly matching names.
It does not generate labels, request your own webcam
data, or automatically download large archives.
If the author's original dataset has a different layout,
a separately reviewed parser and source provenance checks
are required rather than guessing the alignment.

`--check-source` queries metadata only and never downloads
video. The normal default M37 command now runs the
same check automatically before initiating any download.
If the repo is restricted and you have **legitimate
authorized access**, follow its official terms; otherwise
leave the independent real-video evaluation pending.

## Run on the user's Ubuntu computer

Ensure Node 22 is installed (`node --version`). Use the
same research venv that worked for Gym7 MediaPipe extraction.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pip install "huggingface_hub>=0.27,<2"

python -m scripts.m37_repcount_video_benchmark --max-videos 3
```

The command runs end-to-end with no human repetition counting,
CSV creation, re-annotation or repeated manual webcam sessions.

If you have permission for a larger research pilot and
network/storage capacity, you may use:

```bash
python -m scripts.m37_repcount_video_benchmark --max-videos 10 \
  --report data/processed/m37_repcount_squat_event_results_10.json
```

No result may be called independent replication until the
third-party source annotations/video alignment is checked.

If Hugging Face returns **401/403**, the download is blocked
by access/authentication/terms. The script must **fail**:
do not bypass access restrictions, mirror raw video into GitHub,
scrape from alternate untrusted hosts, or present synthetic
results as though they were real-video results. Consult the
official authors' distribution and the mirror's academic-use
conditions; login where genuinely authorized.

If your system does not include OpenCV/MediaPipe Solutions,
install the already used project camera extras and verify
the MediaPipe version. Video download and pose extraction
require substantial local CPU/disk resources; the repo does
not run either in GitHub Actions.

## View the local report

```bash
python - <<'PY'
import json
with open("data/processed/m37_repcount_squat_event_results.json") as f:
    x = json.load(f)
print("Real Squat videos evaluated:", x["squat_evaluated_videos"])
print("Excluded:", x["excluded_videos"], x["exclusion_reasons"])
print("Annotated reps:", x["source_total_annotated_reps"])
print("M34 observed cycles:", x["m34_total_observed_cycles"])
print("Per-video count MAE:", x["session_mean_absolute_count_error"])
print("Event precision:", x["event_precision"])
print("Event recall:", x["event_recall"])
print("Event F1:", x["event_f1"])
print("Mean pose-sample coverage:", x["per_video_pose_sample_coverage_mean"])
print("Estimator parity validated:",
      x["browser_python_pose_estimator_parity_validated"])
PY
```

## Scientific limitations and release gates

- The 3-clip default is an exploratory smoke pilot, not a
  population/generalization result. The subset may contain
  camera perspectives incompatible with M34's original
  fixed 2D knee-angle thresholds, causing low accuracy
  or failure. **Do not tune M34 to this dataset and then
  report on the same clips as unbiased heldout performance.**
- The website uses browser MediaPipe Tasks Vision, whereas
  local M37 uses MediaPipe Solutions Python. Their pose
  estimator source parity **remains unverified**. Thus
  even if M37 performs well, it is not proof of actual
  hosted browser performance on new volunteers.
- Source cycle bounds are based on publisher frame numbers.
  M37 verifies those numbers fit inside the decoded clips,
  but has **not independently inspected each original
  video pair for frame offset, clipping or exact start/end
  semantics**. This may influence event matching.
- The matching-time tolerance is a frozen research
  convention, not a universal rep-duration criterion.
  Event score magnitudes depend on this tolerance.
  Strong results require reporting alternative tolerances
  without tuning them to a preferred outcome.
- This benchmark measures **rep counting only**, not
  squat depth quality, knee alignment, fitness suitability,
  clinical rehabilitation or injury prevention.
- Original RepCount Part-B volunteers may include
  schoolchildren and is **out of scope**. Use academic
  release/attribution terms of the original authors.
- No claim that OHP or Barbell Row have been independently
  tested on RepCount is warranted.

### Citation

Hu, H., Dong, S., Zhao, Y., Lian, D., Li, Z., & Gao, S.
(2022). *TransRAC: Encoding Multi-scale Temporal Correlation
with Transformers for Repetitive Action Counting*.
Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 19013-19022.

## What next?

If M37 actually processes annotated Squat videos and
reports reasonable event precision, recall and pose
coverage without excessive exclusions, next milestones
can compare web Tasks Vision versus Python Solutions
landmarks on the **same consented/source-permitted footage**,
then expand to the original OHP/Row with genuinely
matched event annotations. A validated "good/bad form"
model still requires separate exercise-specific fault
labels and independent subjects.
