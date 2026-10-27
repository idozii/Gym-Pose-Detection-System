"""M31 Bicep Curl: paired participant-level robustness of M30 TRAIN CV.

Never opens dataset ZIP, video files, original M20 validation or TEST.
Retrains the same frozen M30 four RF configurations on the identical
TRAIN video IDs and subject-separated folds. Stores aggregate metrics
and paired participant-cluster intervals, never per-person predictions.

Research-only, not validated exercise-form correctness or safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from gym_pose.gym7_research import read_manifest
from gym_pose.gym7_training import load_clip_table
from gym_pose.gym7_m24_cv import _metrics
from scripts import m30_bicep_curl_temporal as m30

SEED = 42
TREES = 300
BASELINE = "baseline_264"
ALTERNATIVES = tuple(name for name in m30.FAMILIES if name != BASELINE)
STRATEGIES = {
    "group5": lambda: GroupKFold(n_splits=5),
    "loso": LeaveOneGroupOut,
}


def _load_verified_train(motion_path, manifest_path, original_path):
    manifest = read_manifest(manifest_path)
    original = load_clip_table(original_path)
    ids, train, source = m30.frozen_curl_train(manifest, original)
    temporal = m30._read_motion(motion_path)
    if set(temporal.index) != set(ids):
        raise ValueError("M31 original TRAIN clip IDs differ from frozen M30")
    train = train.set_index("video_id", drop=False).loc[ids]
    for vid in ids:
        if any(temporal.at[vid, key] != train.at[vid, key]
               for key in m30.META):
            raise ValueError("M31 original TRAIN person/view/label metadata mismatch")
        if source[vid]["subject_id"] != train.at[vid, "subject_id"]:
            raise ValueError("M31 original subject identity mismatch")
    data = train.loc[:, (*m30.META, *m30.FEATURES)].join(
        temporal.loc[ids, m30.MOTION_COLS], how="inner"
    )
    if (len(data) != len(ids)
        or data.subject_id.nunique() < 5
        or not np.isfinite(
            data.loc[:, m30.FAMILIES["temporal_280"]].to_numpy(dtype=float)
        ).all()):
        raise ValueError("Missing/nonfinite M30 features or too few TRAIN subjects")
    if set(data.exercise) != {"bicep_curl"} or set(data.label) != {"good", "bad"}:
        raise ValueError("M31 requires Bicep Curl publisher whole-video labels")
    return data, ids


def _fit_predict(data, columns, fit_idx, held_idx):
    train_people = set(data.iloc[fit_idx].subject_id)
    held_people = set(data.iloc[held_idx].subject_id)
    if train_people & held_people:
        raise AssertionError("M31 source participant leakage")
    labels = data.iloc[fit_idx].label.map(
        {"good": 0, "bad": 1}
    ).to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("M31 training fold does not have both publisher labels")
    rf = RandomForestClassifier(
        n_estimators=TREES, min_samples_leaf=2,
        class_weight="balanced", random_state=SEED, n_jobs=1
    )
    rf.fit(
        data.iloc[fit_idx].loc[:, columns].to_numpy(dtype=float),
        labels
    )
    if list(rf.classes_) != [0, 1]:
        raise ValueError("Unexpected publisher label encoding")
    return rf.predict_proba(
        data.iloc[held_idx].loc[:, columns].to_numpy(dtype=float)
    )[:, 1]


def _balanced(truth, scores):
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float) >= .5
    if set(truth) != {0, 1}:
        return None
    return float(.5 * (
        np.mean(~scores[truth == 0]) + np.mean(scores[truth == 1])
    ))


def _bad_recall(truth, scores):
    truth = np.asarray(truth, dtype=int)
    if not np.any(truth == 1):
        return None
    return float(np.mean((np.asarray(scores, dtype=float) >= .5)[truth == 1]))


def paired_bootstrap(truth_by_video, scores, people_by_video, *,
                     resamples=1000, seed=SEED):
    """Resample subjects together across all models (fixed OOF predictions).

    Percentile intervals are descriptive only and ignore model
    refitting/feature selection uncertainty. Not independent evidence.
    """
    if not 100 <= resamples <= 5000:
        raise ValueError("Paired bootstrap draws must be 100..5000")
    ids = sorted(truth_by_video)
    if set(scores) != set(m30.FAMILIES):
        raise ValueError("Every frozen M30 feature family must be present")
    if any(set(values) != set(ids) for values in scores.values()):
        raise ValueError("All models must score the same paired TRAIN videos")
    if set(people_by_video) != set(ids):
        raise ValueError("Every frozen TRAIN clip needs one subject group")
    people = sorted(set(people_by_video.values()))
    if len(people) < 5:
        raise ValueError("At least five independent TRAIN participants required")
    clips_by_person = {
        person: [vid for vid in ids if people_by_video[vid] == person]
        for person in people
    }
    rng = np.random.default_rng(seed)
    differences = {
        candidate: {"balanced_accuracy": [], "recall_bad": []}
        for candidate in ALTERNATIVES
    }
    skipped = 0
    for _ in range(resamples):
        sample = [
            vid for person in rng.choice(people, size=len(people), replace=True)
            for vid in clips_by_person[person]
        ]
        truth = [truth_by_video[vid] for vid in sample]
        if set(truth) != {0, 1}:
            skipped += 1
            continue
        reference = [scores[BASELINE][vid] for vid in sample]
        reference_bal = _balanced(truth, reference)
        reference_rec = _bad_recall(truth, reference)
        for candidate in ALTERNATIVES:
            proposed = [scores[candidate][vid] for vid in sample]
            differences[candidate]["balanced_accuracy"].append(
                _balanced(truth, proposed) - reference_bal
            )
            differences[candidate]["recall_bad"].append(
                _bad_recall(truth, proposed) - reference_rec
            )
    output = {}
    for candidate, series in differences.items():
        output[candidate] = {}
        for metric, values in series.items():
            if not values:
                raise ValueError("No valid paired bootstrap draws")
            lo, med, hi = np.quantile(values, [.025, .5, .975])
            output[candidate][metric] = {
                "lower_95": float(lo),
                "median": float(med),
                "upper_95": float(hi),
                "fraction_positive": float(np.mean(np.asarray(values) > 0))
            }
    return {
        "resamples_requested": resamples,
        "resamples_used": resamples - skipped,
        "resamples_skipped_single_class": skipped,
        "interval_method": (
            "Fixed OOF prediction participant-cluster bootstrap, paired by "
            "participant and source video; percentile intervals approximate, "
            "descriptive and NOT independent test inference."
        ),
        "deltas_vs_baseline_264": output
    }


def _subject_direction_counts(truth, scores, groups):
    """Anonymous tally of subjects with better/worse paired clip accuracy."""
    ids = sorted(truth)
    result = {}
    for candidate in ALTERNATIVES:
        counts = {"better_subjects": 0, "equal_subjects": 0,
                  "worse_subjects": 0}
        for person in sorted(set(groups.values())):
            subset = [vid for vid in ids if groups[vid] == person]
            baseline_correct = sum(
                int((scores[BASELINE][vid] >= .5) == (truth[vid] == 1))
                for vid in subset
            )
            candidate_correct = sum(
                int((scores[candidate][vid] >= .5) == (truth[vid] == 1))
                for vid in subset
            )
            if candidate_correct > baseline_correct:
                counts["better_subjects"] += 1
            elif candidate_correct < baseline_correct:
                counts["worse_subjects"] += 1
            else:
                counts["equal_subjects"] += 1
        result[candidate] = counts
    return result


def verify_m30_parity(report, previous_path):
    """Fail closed if recomputation drifts from owner's real M30 report."""
    if not previous_path.is_file():
        return "M30 prior report unavailable; exact reproducibility check skipped"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("paired_clips") != report["paired_train_clips"] or (
        previous.get("subjects") != report["train_subjects"]
    ):
        raise ValueError("Original M30 train cohort size drift")
    for cv in STRATEGIES:
        old = previous["models"][cv]
        new = report["strategies"][cv]
        if old["folds"] != new["folds"]:
            raise ValueError("M30 original participant fold count drift")
        for family in m30.FAMILIES:
            historical = old["models"][family]["all_paired"]
            current = new["models"][family]["all_paired"]
            for metric in (
                "balanced_accuracy", "f1_bad",
                "recall_bad", "average_precision_bad"
            ):
                if not np.isclose(
                    float(historical[metric]), float(current[metric]),
                    rtol=0, atol=1e-12
                ):
                    raise ValueError(
                        f"M31 failed to reproduce M30 {cv}/{family}/{metric}"
                    )
    return "Original M30 GROUP5 and LOSO model metrics reproduced exactly"


def run_study(motion_path, manifest_path, original_path, *,
              resamples=1000):
    data, ids = _load_verified_train(
        motion_path, manifest_path, original_path
    )
    truth = {vid: int(data.at[vid, "label"] == "bad") for vid in ids}
    people = {vid: str(data.at[vid, "subject_id"]) for vid in ids}
    results = {}
    uncertainty = {}
    for name, factory in STRATEGIES.items():
        predictions = {family: {} for family in m30.FAMILIES}
        fold_count = 0
        for fit_indices, hold_indices in factory().split(
            data, groups=data.subject_id
        ):
            fold_count += 1
            if set(data.iloc[fit_indices].subject_id) & set(
                data.iloc[hold_indices].subject_id
            ):
                raise AssertionError("M31 GroupCV person leakage")
            for family, features in m30.FAMILIES.items():
                predicted = _fit_predict(
                    data, features, fit_indices, hold_indices
                )
                for row_index, probability in zip(hold_indices, predicted):
                    vid = ids[int(row_index)]
                    if vid in predictions[family]:
                        raise AssertionError("Repeated TRAIN OOF scoring")
                    predictions[family][vid] = float(probability)
        if any(set(p) != set(ids) for p in predictions.values()):
            raise AssertionError("Incomplete original TRAIN out-of-fold scores")
        metrics = {}
        for family, probabilities in predictions.items():
            metrics[family] = {
                "all_paired": _metrics([
                    (truth[vid], probabilities[vid]) for vid in ids
                ]),
                "by_view": {
                    view: _metrics([
                        (truth[vid], probabilities[vid]) for vid in ids
                        if data.at[vid, "view"] == view
                    ]) for view in ("front", "side", "diagonal")
                }
            }
        results[name] = {
            "folds": fold_count,
            "paired_clips": len(ids),
            "models": metrics,
            "subjects_with_better_or_worse_accuracy_vs_baseline":
                _subject_direction_counts(truth, predictions, people)
        }
        uncertainty[name] = paired_bootstrap(
            truth, predictions, people, resamples=resamples
        )
    return {
        "study": "M31 Bicep Curl M30 paired subject-robustness audit",
        "original_seed": SEED,
        "rf_trees": TREES,
        "paired_train_clips": len(ids),
        "train_subjects": int(data.subject_id.nunique()),
        "publisher_quality_unit": "whole video good/bad only",
        "feature_sets": {k: len(v) for k, v in m30.FAMILIES.items()},
        "strategies": results,
        "cluster_bootstrap": uncertainty,
        "source_video_decoded": False,
        "original_VAL_or_TEST_scored": False,
        "production_model_published": False,
        "caution": (
            "All model comparisons remain exploratory within previously "
            "consulted Gym7 TRAIN. Participant bootstrap resamples fixed "
            "OOF predictions only. No clinically validated form verdict."
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=Path(
        "data/features/m30_bicep_curl_train_motion_v1.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    parser.add_argument("--original", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    parser.add_argument("--m30-report", type=Path, default=Path(
        "data/processed/m30_curl_train_cv.json"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/processed/m31_curl_paired_subject_stability.json"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        if args.report.exists():
            raise FileExistsError("M31 report exists; refuse to overwrite")
        report = run_study(
            args.motion, args.manifest, args.original,
            resamples=args.bootstrap
        )
        report["m30_reproducibility"] = verify_m30_parity(
            report, args.m30_report
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f"M31 aborted: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
