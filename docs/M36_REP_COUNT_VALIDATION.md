# M36 - Rep Counting Validation: human-counted sessions

## Research goal

Owner observed M34 counting works "quite well" on their own
Ubuntu webcam. M35 introduced transparent movement feedback.
These are **informal observations, not measured rep detection
accuracy**.

M36 prepares an explicit offline evaluation against human-
counted complete repetitions for **Squat, Overhead Press
and Barbell Row**, stratified by front, side and diagonal
camera positions. The experimental M34 heuristic observed
cycle counter remains unchanged. This phase evaluates
**count totals per continuous set**, not technique quality.

Critically, **per-session total counts CANNOT identify
rep-level false positives or missed events**. For example,
ten actual reps and ten counted cycles could conceal two
missed reps and two false cycles. So M36 reports **mean
absolute count error, signed bias, RMSE, exact count
agreement, within-one agreement and tracking coverage**,
and deliberately does NOT report detection precision,
recall or F1. That will require independent timestamped
rep event annotations in a later phase.

## How to perform a small pilot safely

No re-download of Gym7, model training, API request or
new video collection is required.

1. Open the current M34/M35 app on localhost. For one
   continuous set, select exactly one of Squat, OHP, Row
   and one camera position. Click **Reset set**; make
   sure the counter is zero. Keep the same camera and
   exercise throughout.
2. Perform only reps you would ordinarily do, at a
   comfortable intensity. Do NOT deliberately perform
   unsafe reps to create error examples.
3. A second person should independently count completed
   repetitions **without seeing the website's counter**.
   If only you can count, note in your research write-up
   that observer independence is not established.
4. When the set ends, note the human count and the displayed
   **Observed cycles** count separately, plus whether
   the landmarks were stable / partial / insufficient.
   Do not reset or switch exercise until recording the
   result. Count a continuous set as **one row**, not
   one row per repetition.
5. Repeat in different camera positions only when easy to
   do so safely; do not force a frontal/diagonal position
   if the joints are not visible. Include quiet baseline
   sessions with zero human reps only if no exercise was
   being performed; false counts on such sessions are
   valuable to detect.

You can start with a **single-person pilot** and report
descriptive count error and coverage, but no valid
generalization claim follows. For a meaningful
expanded protocol, ideally include multiple **genuinely
new consented adults**, vary camera angle and exercise,
and record independent annotations. The script's
>=10-people/30-scorable-session tag is a *reporting
convenience*, NOT a power calculation or validation
guarantee. There is no requirement to do all combinations
in one workout.

Do not upload participant videos, personally identifying
information, exact street/home details or consent documents.
Use pseudonymous codes and keep private consent provenance
outside GitHub. If another person participates, obtain
their informed consent to the recording/analysis protocol.

## Ubuntu commands

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m36_rep_count_validation init
```

Creates a blank private CSV:

```text
data/external/m36_rep_validation/manual_session_counts.csv
```

The CSV has the following **exact** header:

```csv
session_id,participant_id,exercise,view,human_rep_count,website_cycle_count,tracking,consent_confirmed,protocol_version
```

An illustrative synthetic row (NOT a real experiment):

```csv
session_example01,person_example01,squat,side,8,7,stable,yes,m36-manual-count-pilot-v1
```

Input meaning:

- `session_id`: a new code starting with `session_` for
  each continuous set. Never reuse it.
- `participant_id`: stable pseudonym starting with
  `person_`. The same person must reuse it across
  their sessions. Do not use a name or an email.
- `exercise`: `squat`, `ohp`, or `barbell_row`.
  Push-up remains excluded from M34 cycle counting.
- `view`: `front`, `side`, or `diagonal`. Based on
  the camera position, not on the user's body rotation
  alone.
- `human_rep_count`: whole number 0..100 independently
  observed from the continuous set.
- `website_cycle_count`: whole number 0..100 from
  the M34 **Observed cycles** display.
- `tracking`: `stable`, `partial`, or `insufficient`.
  If tracking is insufficient, **leave website_cycle_count
  blank**, not zero. This represents abstention, so the
  evaluation does not hide failed camera observations.
- `consent_confirmed`: the literal `yes` only after
  appropriate research-use consent.
- `protocol_version`: always `m36-manual-count-pilot-v1`.

The repository's `.gitignore` excludes both
`data/external/` and `data/processed/`: no
participant-level session CSV is ever committed.

After recording genuine sessions:

```bash
python -m scripts.m36_rep_count_validation audit
```

Report (local, aggregate-only):

```text
data/processed/m36_rep_count_validation.json
```

To print the useful fields:

```bash
python - <<'PY'
import json
p="data/processed/m36_rep_count_validation.json"
with open(p) as f:
    r=json.load(f)
print("People:",r["distinct_pseudonymous_people"])
print("Sessions:",r["all_consented_sessions"])
print("Coverage:",round(r["counting_coverage"]*100,1),"%")
print("Overall:",r["overall"])
for exercise, data in r["by_exercise"].items():
    print(exercise, data)
for name, data in r["by_exercise_and_view"].items():
    if data["sessions"]:
        print(name, data)
print("Paired participant bootstrap:",r["bootstrap"])
print("Rep-level precision/recall available:",
      r["rep_level_precision_recall_f1_available"])
print("M20 VAL/TEST accessed:",
      r["original_M20_VAL_or_TEST_accessed"])
PY
```

The report contains counts, averages and (if >=3 distinct
people have scorable sessions) an exploratory 1,000-draw
participant-cluster percentile interval for MAE and
signed count error. **No session IDs, participant IDs,
video bytes, landmarks or individual predictions are
written**. The validator rejects non-consented rows,
duplicates, schema drift, untracked source groups,
non-integral count values or improper handling of
insufficient camera tracking. It refuses to overwrite
a previous output; store each frozen snapshot
separately if deliberately conducting a new study.

## Interpreting results

- **MAE = 0.5**: average absolute difference of half
  one cycle per eligible set, NOT 50% precision.
- **Signed error >0**: tendency to overcount. Below
  zero: tendency to undercount.
- **Exact agreement = 0.8**: in 80% of scorable
  sets, both total counts match; this says nothing
  about whether individual timestamps match.
- **Counting coverage** must be shown alongside
  conditional accuracy so camera failures are not
  concealed.
- Camera-view and exercise slices may have few
  examples; report sample counts and do not
  rank views as definitively best.

M36 does not justify a website badge such as
"correct form", "safe exercise" or "accurate reps".
A subsequent M37 could capture event timestamps
from the website **locally on explicit request**
and compare against independently annotated rep
timestamps, enabling true event TP/FP/FN,
rep-level precision, recall and F1.

The original M20 VAL/TEST and Mendeley Gym7 videos
are not opened, scored, or used to tune thresholds.
All original web model, host, camera and UI behavior
is unchanged by M36.
