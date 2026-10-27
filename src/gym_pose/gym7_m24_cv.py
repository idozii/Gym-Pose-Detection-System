"""M24 - fair TRAIN-subject-only RF cross-validation for the new Push-up schema.

All split membership is reconstructed from the ORIGINAL accepted M20 CSV.
The 64-feature paired baseline and upper-body CSVs must come from M23.
Original validation/TEST subjects are prohibited from fitting or scoring.

No final RF is published here. All reported predictions are grouped
out-of-fold predictions from original TRAIN participants only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold

from gym_pose.gym7_research import read_manifest
from gym_pose.gym7_training import load_clip_table
from scripts.m23_train_only_pushup_features import (
    COLS, FEATURE_COLUMNS, META, SCHEMA, frozen_train_manifest,
)

MODELS = ("full_paired", "upper_paired", "upper_expanded")
SEED = 42


def _read_table(path: Path):
    table = pd.read_csv(path, keep_default_na=False)
    if tuple(table.columns) != COLS:
        raise ValueError("Expected exactly the M23 64-feature schema")
    if table.empty or table.video_id.duplicated().any():
        raise ValueError("Empty or repeated video IDs")
    if not table.exercise.eq("pushup").all():
        raise ValueError("M24 only permits Push-up video-quality data")
    if not table.label.isin(("good", "bad")).all():
        raise ValueError("Invalid publisher label in feature CSV")
    if not table.view.isin(("front", "side", "diagonal")).all():
        raise ValueError("Invalid original camera view")
    if not table.subject_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Missing subject IDs")
    features = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite feature input")
    return table.set_index("video_id", drop=False)


def verify_cohort(full: pd.DataFrame, upper: pd.DataFrame, original_manifest: list,
                  original_m20: pd.DataFrame):
    """No new participant, changed metadata, or CV fold contamination."""
    chosen, subjects = frozen_train_manifest(original_manifest, original_m20, seed=SEED)
    # M20 split contains 15 TRAIN people overall, but only 14 provided
    # Push-up clips in the owner's extracted data. Use only participants
    # with actual original TRAIN Push-up clips for Push-up CV folds.
    eligible_people = {r["subject_id"] for r in chosen}
    if not eligible_people.issubset(subjects):
        raise ValueError("Source participant not in original M20 TRAIN")
    source = {r["video_id"]: r for r in chosen}
    if len(source) != len(chosen):
        raise ValueError("Duplicated original video identity")
    if not set(full.index).issubset(set(upper.index)):
        raise ValueError("Upper admission must include every full-body clip")
    if not len(full) or not len(upper):
        raise ValueError("No training clips")
    if not set(upper.index).issubset(source):
        raise ValueError("M23 CSV has unknown/non-TRAIN videos")
    for gate, table in (("full", full), ("upper", upper)):
        for r in table.itertuples(index=False):
            meta = source[r.video_id]
            if any(getattr(r, k) != meta[k] for k in META):
                raise ValueError(f"{gate} metadata differs from original verified source")
            if r.subject_id not in subjects:
                raise ValueError("Crossing subject-disjoint TRAIN boundary")
    common_ids = sorted(full.index)
    if set(full.subject_id) != eligible_people:
        raise ValueError("Full gate must cover every available Push-up TRAIN participant")
    if len(set(full.subject_id)) < 5:
        raise ValueError("Require at least five original TRAIN subjects")
    return common_ids, sorted(upper.index), sorted(eligible_people), len(chosen)


def _fit(train: pd.DataFrame, *, n_trees=200):
    X = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train.label.map({"good": 0, "bad": 1}).to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Training fold missing an original publisher label")
    classifier = RandomForestClassifier(
        n_estimators=n_trees, min_samples_leaf=2,
        class_weight="balanced", random_state=SEED, n_jobs=1
    )
    classifier.fit(X, y)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected model class order")
    return classifier


def _predict(model, eval_table: pd.DataFrame):
    return model.predict_proba(
        eval_table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    )[:, 1]


def _metrics(rows, *, threshold=.5):
    if not rows:
        return {"n": 0, "status": "no_predictions"}
    y = np.asarray([item[0] for item in rows], dtype=int)
    prob = np.asarray([item[1] for item in rows], dtype=float)
    pred = (prob >= threshold).astype(int)
    data = {
        "n": int(len(y)),
        "n_good": int(np.sum(y == 0)),
        "n_bad": int(np.sum(y == 1)),
        "confusion_matrix_good_bad": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
        "precision_bad": float(precision_score(y, pred, zero_division=0)),
        "recall_bad": float(recall_score(y, pred, zero_division=0)),
        "f1_bad": float(f1_score(y, pred, zero_division=0)),
    }
    if set(y) == {0, 1}:
        data["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        data["average_precision_bad"] = float(average_precision_score(y, prob))
    else:
        data["balanced_accuracy"] = None
        data["average_precision_bad"] = None
        data["caution"] = "Only one publisher label in group, no balanced accuracy"
    return data


def compare_cv(full: pd.DataFrame, upper: pd.DataFrame, original_manifest: list,
               original_m20: pd.DataFrame, *, folds=5, trees=200):
    if folds < 2 or trees < 25 or trees > 1000:
        raise ValueError("Invalid CV folds or model size")
    common_ids, upper_ids, subjects, selected = verify_cohort(
        full, upper, original_manifest, original_m20
    )
    matched_full = full.loc[common_ids]
    matched_upper = upper.loc[common_ids]
    if matched_full.subject_id.nunique() < folds:
        raise ValueError("Too few original TRAIN participants for GroupKFold")
    crossval = GroupKFold(n_splits=folds)
    common_truth = {}
    out_of_fold = {mode: {} for mode in MODELS}
    expanded_truth = {}

    for fold_id, (train_idx, hold_idx) in enumerate(
        crossval.split(matched_full, groups=matched_full.subject_id), start=1
    ):
        train_ids = [common_ids[i] for i in train_idx]
        hold_ids = [common_ids[i] for i in hold_idx]
        train_people = set(matched_full.loc[train_ids].subject_id)
        hold_people = set(matched_full.loc[hold_ids].subject_id)
        if train_people & hold_people:
            raise AssertionError("GroupKFold participant leakage")
        if train_people | hold_people != set(subjects):
            raise ValueError("Full-body common clips do not cover every subject")

        expanded_train_ids = [
            ident for ident in upper_ids
            if upper.at[ident, "subject_id"] in train_people
        ]
        expanded_hold_ids = [
            ident for ident in upper_ids
            if upper.at[ident, "subject_id"] in hold_people
        ]
        assert not set(expanded_train_ids) & set(expanded_hold_ids)
        assert all(upper.at[i, "subject_id"] not in hold_people
                   for i in expanded_train_ids)

        train_tables = {
            "full_paired": full.loc[train_ids],
            "upper_paired": upper.loc[train_ids],
            "upper_expanded": upper.loc[expanded_train_ids],
        }
        eval_tables = {
            "full_paired": full.loc[hold_ids],
            "upper_paired": upper.loc[hold_ids],
            "upper_expanded": upper.loc[hold_ids],
        }
        for mode in MODELS:
            model = _fit(train_tables[mode], n_trees=trees)
            eval_rows = eval_tables[mode]
            prob = _predict(model, eval_rows)
            for ident, score in zip(hold_ids, prob):
                if ident in out_of_fold[mode]:
                    raise AssertionError("Same video predicted twice")
                out_of_fold[mode][ident] = float(score)
                common_truth[ident] = int(full.at[ident, "label"] == "bad")

            if mode == "upper_expanded":
                # Extra recovered held-out clips are a separate coverage-only
                # cohort; their quality metrics cannot be paired with full.
                extra_ids = sorted(set(expanded_hold_ids) - set(hold_ids))
                if extra_ids:
                    extra_prob = _predict(model, upper.loc[extra_ids])
                    for ident, score in zip(extra_ids, extra_prob):
                        if ident in expanded_truth:
                            raise AssertionError("Repeated recovered held-out video")
                        expanded_truth[ident] = (
                            int(upper.at[ident, "label"] == "bad"), float(score),
                            upper.at[ident, "view"]
                        )
    if any(set(out_of_fold[m]) != set(common_ids) for m in MODELS):
        raise AssertionError("Out-of-fold predictions incomplete")
    expected_recovered = set(upper_ids) - set(common_ids)
    if set(expanded_truth) != expected_recovered:
        raise AssertionError("Recovered samples missing in cross-validation")

    common_results = {}
    for mode, predictions in out_of_fold.items():
        ordered = [
            (common_truth[ident], predictions[ident]) for ident in common_ids
        ]
        per_view = {}
        for view in ("front", "side", "diagonal"):
            subset = [
                (common_truth[ident], predictions[ident])
                for ident in common_ids
                if full.at[ident, "view"] == view
            ]
            per_view[view] = _metrics(subset)
        common_results[mode] = {
            "pooled_group_oof": _metrics(ordered),
            "by_view": per_view,
        }

    recovered = [
        (truth, score) for truth, score, _view in expanded_truth.values()
    ]
    recovered_by_view = {
        view: _metrics([
            (truth, score) for truth, score, rec_view in expanded_truth.values()
            if rec_view == view
        ])
        for view in ("front", "side", "diagonal")
    }
    return {
        "study": "M24 original TRAIN subject GroupKFold research only",
        "source": "Mendeley Gym7 v3",
        "label_semantics": "publisher video good/bad, NOT named form errors",
        "schema": SCHEMA,
        "original_m20_seed": SEED,
        "frozen_original_train_subjects": len(subjects),
        "original_train_pushup_clips": selected,
        "full_gate_clips": len(common_ids),
        "upper_gate_clips": len(upper_ids),
        "newly_accepted_train_clips": len(expected_recovered),
        "paired_common_clips": len(common_ids),
        "group_cv_folds": folds,
        "n_estimators": trees,
        "heldout_m20_val_or_test_scored": False,
        "test_accessed": False,
        "metrics_on_identical_paired_clips": common_results,
        "recovered_unpaired_oof_only": {
            "note": "TRAIN-fold recovered clips have no matching full-body score",
            "all_views": _metrics(recovered),
            "by_view": recovered_by_view,
        },
        "conclusion_guard": (
            "This is internal TRAIN-only cross-validation, not independent "
            "form-quality accuracy. An increased clip count does not imply "
            "a better classifier. Original M20 validation and TEST subjects "
            "are unchanged and unscored."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_full_body.csv"))
    p.add_argument("--upper", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_upper_body.csv"))
    p.add_argument("--original-features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    p.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    p.add_argument("--output", type=Path, default=Path(
        "data/processed/m24_pushup_group_cv.json"))
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Existing M24 report will never be overwritten")
        full = _read_table(args.full)
        upper = _read_table(args.upper)
        original_m20 = load_clip_table(args.original_features)
        manifest = read_manifest(args.manifest)
        report = compare_cv(full, upper, manifest, original_m20,
                            folds=args.folds, trees=args.trees)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, RuntimeError, AssertionError) as e:
        print(f"M24 comparison failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
