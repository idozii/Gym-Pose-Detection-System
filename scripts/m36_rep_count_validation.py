"""M36: private, human-annotated session-count validation for M34.

Read ONLY a local CSV of manually observed repetition totals and the
M34 website's final observed-cycle count. No video/landmark processing,
machine-learning training or Fitness-AQA / Gym7 VAL/TEST usage.

The report contains anonymous aggregates only, including abstention/
tracking coverage. Net count error cannot determine rep-level TP/FP/FN,
so this script intentionally does NOT report precision, recall or F1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys

PROTOCOL = "m36-manual-count-pilot-v1"
COLUMNS = (
    "session_id", "participant_id", "exercise", "view",
    "human_rep_count", "website_cycle_count",
    "tracking", "consent_confirmed", "protocol_version"
)
EXERCISES = ("squat", "ohp", "barbell_row")
VIEWS = ("front", "side", "diagonal")
TRACKING = ("stable", "partial", "insufficient")
SAFE_SESSION_ID = re.compile(r"^session_[a-zA-Z0-9_-]{5,48}$")
SAFE_PERSON_ID = re.compile(r"^person_[a-zA-Z0-9_-]{5,48}$")
SEED = 42
BOOTSTRAP = 1000


def create_template(destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("M36 source annotations already exist; refusing to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(COLUMNS)
    return {"created": str(destination), "rows": 0, "protocol": PROTOCOL}


def _count(text: str, field: str) -> int:
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", text):
        raise ValueError(f"Invalid {field}: expected integer 0..100")
    count = int(text)
    if count > 100:
        raise ValueError(f"Invalid {field}: count above 100")
    return count


def read_sessions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unexpected M36 source schema; see docs/M36_REP_COUNT_VALIDATION.md")
        records = list(reader)
    if not records:
        raise ValueError("No real human-annotated M36 sessions yet. Template is intentionally blank")
    if len(records) > 10000:
        raise ValueError("Too many sessions for one M36 validation report")

    seen = set()
    results = []
    for raw in records:
        if None in raw or set(raw) != set(COLUMNS) or any(
            not isinstance(value, str) or value != value.strip()
            for value in raw.values()
        ):
            raise ValueError("Malformed, blank-column or padded M36 CSV row")
        sid = raw["session_id"]
        person = raw["participant_id"]
        if not SAFE_SESSION_ID.fullmatch(sid) or not SAFE_PERSON_ID.fullmatch(person):
            raise ValueError("Use pseudonymous session_/person_ IDs, not personal details")
        if sid in seen:
            raise ValueError("Duplicate session_id; each real-world set must be unique")
        seen.add(sid)
        if raw["exercise"] not in EXERCISES or raw["view"] not in VIEWS:
            raise ValueError("Unexpected M36 exercise or camera view")
        if raw["tracking"] not in TRACKING:
            raise ValueError("Tracking must be stable, partial, or insufficient")
        if raw["consent_confirmed"] != "yes":
            raise ValueError("All sessions must have research-use consent before inclusion")
        if raw["protocol_version"] != PROTOCOL:
            raise ValueError("Unexpected M36 frozen annotation protocol")
        human = _count(raw["human_rep_count"], "human_rep_count")
        # When tracking is insufficient, a final observed-cycle count
        # may be blank; do not equate it with observed zero.
        observed_text = raw["website_cycle_count"]
        if raw["tracking"] == "insufficient":
            if observed_text != "":
                raise ValueError("insufficient tracking must abstain: leave website_cycle_count blank")
            observed = None
        else:
            observed = _count(observed_text, "website_cycle_count")
        results.append({
            "participant": person,
            "exercise": raw["exercise"],
            "view": raw["view"],
            "tracking": raw["tracking"],
            "human": human,
            "observed": observed
        })
    return results


def _aggregate(scored: list[dict]) -> dict | None:
    if not scored:
        return None
    errors = [item["observed"] - item["human"] for item in scored]
    n = len(errors)
    return {
        "sessions": n,
        "observed_human_reps": sum(item["human"] for item in scored),
        "website_observed_cycles": sum(item["observed"] for item in scored),
        "mean_absolute_count_error": sum(abs(err) for err in errors) / n,
        "mean_signed_count_error": sum(errors) / n,
        "root_mean_squared_count_error": math.sqrt(
            sum(error * error for error in errors) / n
        ),
        "exact_session_count_fraction": sum(error == 0 for error in errors) / n,
        "within_one_cycle_fraction": sum(abs(error) <= 1 for error in errors) / n,
        "overcount_sessions": sum(error > 0 for error in errors),
        "undercount_sessions": sum(error < 0 for error in errors),
        "equal_count_sessions": sum(error == 0 for error in errors)
    }


def _bootstrap_by_person(scored: list[dict], *, draws: int = BOOTSTRAP,
                         seed: int = SEED) -> dict | None:
    """Resample complete participants, never individual sessions across people.

    Intervals are descriptive; results from a few persons are not
    population-level validation, even if intervals are narrow.
    """
    if draws < 100 or draws > 5000:
        raise ValueError("Bootstrap draws must be 100..5000")
    people = sorted({item["participant"] for item in scored})
    if len(people) < 3:
        return None
    import random
    rng = random.Random(seed)
    by_person = {
        person: [item for item in scored if item["participant"] == person]
        for person in people
    }
    samples = {"mae": [], "signed_error": []}
    for _ in range(draws):
        selected = [
            record for person in rng.choices(people, k=len(people))
            for record in by_person[person]
        ]
        summary = _aggregate(selected)
        samples["mae"].append(summary["mean_absolute_count_error"])
        samples["signed_error"].append(summary["mean_signed_count_error"])
    def interval(items):
        values = sorted(items)
        def percentile(p):
            index = p * (len(values) - 1)
            low = int(index)
            frac = index - low
            return values[low] * (1 - frac) + values[min(low + 1, len(values) - 1)] * frac
        return {
            "lower_95": percentile(.025),
            "median": percentile(.5),
            "upper_95": percentile(.975)
        }
    return {
        "resamples": draws,
        "unit": "person",
        "method": "exploratory participant-cluster bootstrap, not an independent accuracy guarantee",
        "mae": interval(samples["mae"]),
        "signed_error": interval(samples["signed_error"])
    }


def report_sessions(sessions: list[dict]) -> dict:
    if not sessions:
        raise ValueError("At least one consented annotation row is required")
    scored = [row for row in sessions if row["observed"] is not None]
    people = {row["participant"] for row in sessions}
    group = lambda exercise=None, view=None: [
        row for row in scored
        if (exercise is None or row["exercise"] == exercise)
        and (view is None or row["view"] == view)
    ]
    all_by_exercise = {exercise: [
        row for row in sessions if row["exercise"] == exercise
    ] for exercise in EXERCISES}
    return {
        "study": "M36 original website M34/M35 rep-count validation pilot",
        "protocol": PROTOCOL,
        "evidence_level": "manually recorded session totals; not per-repetition event validation",
        "all_consented_sessions": len(sessions),
        "distinct_pseudonymous_people": len(people),
        "scorable_sessions": len(scored),
        "abstained_sessions": len(sessions) - len(scored),
        "counting_coverage": len(scored) / len(sessions),
        "by_tracking": {tracking: sum(row["tracking"] == tracking for row in sessions)
                        for tracking in TRACKING},
        "overall": _aggregate(scored),
        "by_exercise": {
            exercise: {
                "sessions": len(all_by_exercise[exercise]),
                "abstained": sum(row["observed"] is None for row in all_by_exercise[exercise]),
                "scorable": _aggregate(group(exercise))
            }
            for exercise in EXERCISES
        },
        "by_exercise_and_view": {
            f"{exercise}/{view}": {
                "sessions": sum(row["exercise"] == exercise and row["view"] == view
                                for row in sessions),
                "scorable": _aggregate(group(exercise, view))
            } for exercise in EXERCISES for view in VIEWS
        },
        "bootstrap": _bootstrap_by_person(scored),
        "evaluation_on_new_independent_participants_verified": False,
        "original_M20_VAL_or_TEST_accessed": False,
        "video_or_landmarks_read": False,
        "individual_ids_in_report": False,
        "rep_level_precision_recall_f1_available": False,
        "limitations": (
            "Matching total counts can hide equal numbers of false and missed "
            "cycles; no per-rep TP/FP/FN or precision/recall/F1 is inferable. "
            "Self-counting/observer feedback and camera choice may bias this "
            "pilot. Insufficient tracking sessions are reported as abstentions, "
            "not silently scored as zero. New people and consent provenance "
            "require separate independent verification."
        ),
        "status": (
            "descriptive_pilot_only"
            if len(people) < 10 or len(scored) < 30
            else "expanded_descriptive_pilot_not_prospective_validation"
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="Create an empty private annotation CSV")
    init.add_argument("--csv", type=Path, default=Path(
        "data/external/m36_rep_validation/manual_session_counts.csv"))
    audit = sub.add_parser("audit", help="Report consented count errors by exercise/view")
    audit.add_argument("--csv", type=Path, default=Path(
        "data/external/m36_rep_validation/manual_session_counts.csv"))
    audit.add_argument("--output", type=Path, default=Path(
        "data/processed/m36_rep_count_validation.json"))
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            result = create_template(args.csv)
        else:
            if args.output.exists():
                raise FileExistsError("M36 report already exists; refusing to overwrite")
            sessions = read_sessions(args.csv)
            result = report_sessions(sessions)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, TypeError) as error:
        print(f"M36 {args.action} stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
