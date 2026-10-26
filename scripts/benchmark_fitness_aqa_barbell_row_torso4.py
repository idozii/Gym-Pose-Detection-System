#!/usr/bin/env python3
"""Fitness-AQA Barbell Row torso4 STATIC RF baseline; original TEST locked.

Fits one separately calibrated RF for each publisher image-label task:
torso_angle, lumbar. Uses ONLY original source TRAIN and VAL 20-feature
CSV plus full extraction summary. No images/videos are opened and no
temporal ordering or subject independence is claimed.

Operating thresholds: selected via a held-out *TRAIN candidate prefix*
group split, then final RF is refit on all publisher TRAIN. Validation
is evaluated once at fixed 0.5 and train-selected thresholds. The prefix
is an unverified filename heuristic, NOT confirmed clip/subject ID.
Repeated use of validation across development makes scores exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from gym_pose.barbell_row_features import TARGETS
from gym_pose.barbell_row_torso4_features import (
    FEATURE_NAMES, SCHEMA_VERSION,
)
from scripts.compare_fitness_aqa_temporal import choose_threshold
from scripts.diagnose_fitness_aqa_thresholds import _metrics

META = ("image_id", "split", "target", "error_label", "candidate_clip_prefix")
REQUIRED_COLUMNS = (*META, *FEATURE_NAMES)
FULL_PUBLISHER_SPLIT_COUNTS = {
    "train": {"torso_angle": 11881, "lumbar": 10360},
    "val": {"torso_angle": 2564, "lumbar": 2227},
}


def _load(source_dir: Path, *, require_full: bool = True) -> tuple[dict, dict]:
    source_dir = Path(source_dir)
    summary = json.loads((
        source_dir / "fitness_aqa_barbell_row_torso4_extraction_summary.json"
    ).read_text(encoding="utf-8"))
    if (summary.get("status") != "barbell_row_image_pose_extraction_source_train_val_NO_TEST"
        or summary.get("feature_profile") != "torso4"
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("feature_names") != list(FEATURE_NAMES)
        or summary.get("feature_count") != len(FEATURE_NAMES)
        or summary.get("source_test_images_decoded") != 0
        or summary.get("cross_target_split_id_conflicts") != 0):
        raise ValueError("Manifest is not the expected locked-test 20-feature source extraction")
    if (not math.isclose(summary.get("min_visibility", -1), .5)
        or summary.get("source_split_type") !=
            "publisher_TASK_SPECIFIC_image_ID_NOT_verified_subject_or_clip_disjoint"):
        raise ValueError("Unexpected source split type or pose visibility setting")
    if require_full and summary.get("max_images_per_task_split") != 0:
        raise ValueError("Pilot extraction is not accepted: run full --max-images-per-task-split 0")
    if set(summary.get("splits", {})) != {"train", "val"}:
        raise ValueError("Only complete original TRAIN+VAL publisher splits permitted")

    rows = {task: {} for task in TARGETS}
    for split in ("train", "val"):
        stats = summary["splits"][split]
        if (stats.get("image_decode_failures") != 0
            or stats.get("images_decoded") != stats.get("unique_images_to_decode")):
            raise ValueError(f"{split}: source image decoding did not complete")
        for task in TARGETS:
            task_stats = stats["tasks"][task]
            if (require_full and
                task_stats.get("selected_images") != FULL_PUBLISHER_SPLIT_COUNTS[split][task]):
                raise ValueError(f"{task} {split}: official complete image split expected")
            if task_stats["pose_accepted"] + task_stats["pose_rejected"] != task_stats["selected_images"]:
                raise ValueError(f"{task} {split}: pose acceptance counts inconsistent")
            file = source_dir / f"fitness_aqa_barbell_row_torso4_{task}_{split}_pose.csv"
            df = pd.read_csv(file, dtype={
                "image_id": "string",
                "split": "string",
                "target": "string",
                "candidate_clip_prefix": "string",
            })
            if df.empty or list(df) != list(REQUIRED_COLUMNS):
                raise ValueError(f"{file}: empty/changed torso4 CSV schema")
            if df[list(REQUIRED_COLUMNS)].isna().any().any():
                raise ValueError(f"{file}: missing metadata or feature values")
            if not df["split"].eq(split).all() or not df["target"].eq(task).all():
                raise ValueError(f"{file}: source split/task mismatch")
            if df.image_id.duplicated().any():
                raise ValueError(f"{file}: duplicated publisher image IDs")
            if not df["error_label"].isin([0, 1]).all():
                raise ValueError(f"{file}: expected binary 0/1 task annotations")
            if not (df["image_id"].str.rsplit("_", n=1).str[0] ==
                    df["candidate_clip_prefix"]).all():
                raise ValueError(f"{file}: candidate prefix/source ID mismatch")
            features = df.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
            if not np.isfinite(features).all():
                raise ValueError(f"{file}: nonfinite numeric pose features")
            retained = len(df)
            positives = int(df.error_label.sum())
            if (retained != task_stats["pose_accepted"]
                or positives != task_stats["positive_accepted"]
                or retained - positives != task_stats["negative_accepted"]
                or positives > task_stats["positive_selected"]):
                raise ValueError(f"{file}: CSV labels disagree with source extraction summary")
            if set(df.error_label) != {0, 1}:
                raise ValueError(f"{file}: positive/negative class both required")
            rows[task][split] = df

    for task in TARGETS:
        tr, va = rows[task]["train"], rows[task]["val"]
        if set(tr.image_id) & set(va.image_id):
            raise ValueError(f"{task}: publisher train/val IMAGE ID overlap")
        if set(tr.candidate_clip_prefix) & set(va.candidate_clip_prefix):
            raise ValueError(
                f"{task}: candidate clip prefixes cross source train/val; evaluation unsafe")
        source_audit = summary.get("candidate_clip_prefix_audit", {}).get(task, {})
        if source_audit.get("candidate_prefix_overlap_train_val") != 0:
            raise ValueError(f"{task}: source prefix audit says train/val overlap")
    return rows, summary


def train_only_prefix_calibration(
    df: pd.DataFrame, *, seed: int,
    min_fit_positive_groups: int = 5, min_cal_positive_groups: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition training IMAGEs by unverified candidate prefix, not individual rows."""
    if df.candidate_clip_prefix.nunique() < 12:
        raise ValueError("Too few candidate source groups for train-only calibration")
    splitter = GroupShuffleSplit(n_splits=96, test_size=.25, random_state=seed)
    for fit_idx, cal_idx in splitter.split(
        df, groups=df["candidate_clip_prefix"]):
        fit = df.iloc[fit_idx]
        cal = df.iloc[cal_idx]
        fit_groups = set(fit.candidate_clip_prefix)
        cal_groups = set(cal.candidate_clip_prefix)
        if fit_groups & cal_groups:
            raise AssertionError("TRAIN fit/calibration prefix groups overlap")
        if (set(fit.error_label) == {0, 1}
            and set(cal.error_label) == {0, 1}
            and fit.loc[fit.error_label.eq(1), "candidate_clip_prefix"].nunique()
                >= min_fit_positive_groups
            and cal.loc[cal.error_label.eq(1), "candidate_clip_prefix"].nunique()
                >= min_cal_positive_groups):
            return fit_idx, cal_idx
    raise ValueError("No train-only prefix-group partition with sufficient positive groups")


def make_rf(*, seed: int, trees: int) -> RandomForestClassifier:
    if trees < 1:
        raise ValueError("trees must be positive")
    return RandomForestClassifier(
        n_estimators=trees, random_state=seed,
        min_samples_leaf=2, class_weight="balanced", n_jobs=-1)


def probabilities(fitted, x: pd.DataFrame) -> np.ndarray:
    if not np.array_equal(fitted.classes_, [0, 1]):
        raise ValueError("Both source error classes required")
    fitted.n_jobs = 1  # deterministic probability reduction across trees
    probs = np.asarray(fitted.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Invalid RF error probability")
    return probs


def prefix_group_bootstrap(
    y: np.ndarray, probs: np.ndarray, groups: np.ndarray,
    threshold: float, *, repetitions: int, seed: int,
) -> dict:
    """Resample candidate prefix blocks, NEVER assert they are true athlete IDs."""
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if (len(y) != len(groups) or len(y) != len(probs)
        or set(y) != {0, 1}):
        raise ValueError("Bootstrap needs binary labels and aligned group predictions")
    unique = np.unique(groups)
    index = {k: np.flatnonzero(groups == k) for k in unique}
    rng = np.random.default_rng(seed)
    ap, f1, skipped = [], [], 0
    for _ in range(repetitions):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[k] for k in drawn])
        if set(y[idx]) != {0, 1}:
            skipped += 1
            continue
        ap.append(float(np.clip(average_precision_score(y[idx], probs[idx]), 0, 1)))
        f1.append(float(f1_score(
            y[idx], probs[idx] >= threshold, average="macro",
            labels=[0, 1], zero_division=0)))
    if len(ap) < repetitions * .8:
        raise ValueError("Too few valid two-class bootstrap samples")
    return {
        "resampling_unit": "UNVERIFIED_filename_candidate_clip_prefix",
        "resamples_requested": repetitions, "resamples_valid": len(ap),
        "resamples_single_class_discarded": skipped,
        "validation_candidate_prefix_count": len(unique),
        "average_precision": {
            "percentile_2_5": float(np.percentile(ap, 2.5)),
            "percentile_97_5": float(np.percentile(ap, 97.5)),
        },
        "macro_f1_train_threshold": {
            "percentile_2_5": float(np.percentile(f1, 2.5)),
            "percentile_97_5": float(np.percentile(f1, 97.5)),
        },
        "warning": "Exploratory source-prefix resampling, not verified clip or subject-level CI",
    }


def benchmark(
    feature_dir: str | Path, output_dir: str | Path, *,
    seed: int = 42, trees: int = 300, bootstraps: int = 500,
    require_full: bool = True,
) -> dict:
    if trees < 1 or bootstraps < 100:
        raise ValueError("Invalid trees/bootstraps")
    source, manifest = _load(Path(feature_dir), require_full=require_full)
    report = {
        "status": "barbell_row_torso4_STATIC_RF_source_train_calibrated_validation_exploratory_NO_TEST",
        "source": "Fitness-AQA Barbell Row independently split per-image labels",
        "source_test_images_opened": False,
        "source_training_only_threshold_selection": True,
        "seed": seed, "trees": trees,
        "schema": SCHEMA_VERSION, "features": list(FEATURE_NAMES),
        "tasks": {},
        "limitations": [
            "The source uses per-IMAGE binary 0/1 error labels, not ordered video frames",
            "Publisher splits are task-specific and exact image-ID-disjoint, NOT verified subject-independent",
            "Filename candidate prefixes are UNVERIFIED clip/athlete grouping identifiers",
            "Sample duplicates/near-duplicates may remain across publisher splits",
            "Data pose admission (~70%) excludes a non-random subset; results are accepted-image-only",
            "A negative source annotation is absence of a named error, NOT globally correct lift",
            "Shoulder-hip landmarks CANNOT directly measure lumbar curvature or spine health",
            "Validation scores are exploratory; this split is not an independent production holdout",
            "No original publisher TEST pixels accessed; no real-time trainer integration or safety claim",
        ],
    }
    final_models = {}
    for task_i, task in enumerate(TARGETS):
        train, val = source[task]["train"], source[task]["val"]
        if int(val.error_label.sum()) < 20:
            raise ValueError(f"{task}: too few held-out positive image labels")
        fit_idx, cal_idx = train_only_prefix_calibration(
            train, seed=seed + task_i)
        xtrain = train.loc[:, list(FEATURE_NAMES)].astype(float)
        xval = val.loc[:, list(FEATURE_NAMES)].astype(float)
        ytrain = train.error_label.to_numpy(dtype=int)
        yval = val.error_label.to_numpy(dtype=int)
        temporary = make_rf(seed=seed, trees=trees)
        temporary.fit(xtrain.iloc[fit_idx], ytrain[fit_idx])
        calibration_probs = probabilities(temporary, xtrain.iloc[cal_idx])
        selection = choose_threshold(ytrain[cal_idx], calibration_probs)
        threshold = float(selection["threshold"])
        trained = make_rf(seed=seed, trees=trees)
        trained.fit(xtrain, ytrain)
        p = probabilities(trained, xval)
        fit_groups = sorted(set(train.iloc[fit_idx].candidate_clip_prefix))
        cal_groups = sorted(set(train.iloc[cal_idx].candidate_clip_prefix))
        item = {
            "schema_version": SCHEMA_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "train_source_images": int(manifest["splits"]["train"]["tasks"][task]["selected_images"]),
            "train_pose_accepted": len(train),
            "train_positive_accepted": int(ytrain.sum()),
            "train_positive_prevalence": float(ytrain.mean()),
            "train_pose_coverage": float(manifest["splits"]["train"]["tasks"][task]["pose_coverage_selected"]),
            "validation_source_images": int(manifest["splits"]["val"]["tasks"][task]["selected_images"]),
            "validation_pose_accepted": len(val),
            "validation_positive_accepted": int(yval.sum()),
            "validation_positive_prevalence": float(yval.mean()),
            "validation_pose_coverage": float(manifest["splits"]["val"]["tasks"][task]["pose_coverage_selected"]),
            "candidate_prefix_groups_train": int(train.candidate_clip_prefix.nunique()),
            "candidate_prefix_groups_val": int(val.candidate_clip_prefix.nunique()),
            "source_train_val_candidate_prefix_disjoint": True,
            "calibration": {
                "grouping": "unverified_image_filename_prefix",
                "fit_images": len(fit_idx), "calibration_images": len(cal_idx),
                "fit_prefixes": len(fit_groups),
                "calibration_prefixes": len(cal_groups),
                "fit_positive_prefixes": int(
                    train.iloc[fit_idx].loc[
                        train.iloc[fit_idx].error_label.eq(1),
                        "candidate_clip_prefix"].nunique()),
                "calibration_positive_prefixes": int(
                    train.iloc[cal_idx].loc[
                        train.iloc[cal_idx].error_label.eq(1),
                        "candidate_clip_prefix"].nunique()),
                "prefix_group_disjoint": True,
                "group_hash_sha256": hashlib.sha256(
                    ("|".join(fit_groups) + "--" + "|".join(cal_groups)).encode()
                ).hexdigest(),
                "train_only_selected_threshold": threshold,
                "train_calibration_macro_f1": float(selection["macro_f1"]),
            },
            "validation_fixed_0_5": _metrics(yval, p, .5),
            "validation_train_selected_threshold": _metrics(yval, p, threshold),
            "validation_average_precision": float(
                np.clip(average_precision_score(yval, p), 0, 1)),
            "random_ranker_AP_reference": float(yval.mean()),
            "validation_prefix_bootstrap": prefix_group_bootstrap(
                yval, p, val.candidate_clip_prefix.to_numpy(dtype=str),
                threshold, repetitions=bootstraps, seed=seed + task_i + 100),
        }
        report["tasks"][task] = item
        final_models[task] = {
            "model": trained, "exercise": "barbell_row",
            "target": task, "source_train_only": True,
            "publisher_test_untouched": True,
            "feature_names": list(FEATURE_NAMES), "schema_version": SCHEMA_VERSION,
            "threshold_train_prefix_calibration": threshold,
            "no_direct_lumbar_spine_measurement": True,
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for task, payload in final_models.items():
        joblib.dump(payload, output / f"barbell_row_{task}_torso4_static_rf.joblib")
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path(
        "data/features/fitness_aqa_barbell_row_torso4"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/fitness_aqa_barbell_row_torso4_rf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--allow-pilot", action="store_true",
                        help="For synthetic testing only: pilot scores are NOT valid benchmarks")
    args = parser.parse_args()
    try:
        result = benchmark(
            args.features_dir, args.output_dir,
            seed=args.seed, trees=args.trees, bootstraps=args.bootstraps,
            require_full=not args.allow_pilot)
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError) as exc:
        parser.exit(1, f"Barbell Row Static RF benchmark failed: {exc}\n")
    compact = {
        "status": result["status"],
        "publisher_TEST_images_opened": result["source_test_images_opened"],
        "tasks": {
            task: {
                "train_pose_accepted": data["train_pose_accepted"],
                "validation_pose_accepted": data["validation_pose_accepted"],
                "validation_positive_accepted": data["validation_positive_accepted"],
                "coverage_train": data["train_pose_coverage"],
                "coverage_validation": data["validation_pose_coverage"],
                "threshold_selected_on_source_train": (
                    data["calibration"]["train_only_selected_threshold"]),
                "validation_AP": data["validation_average_precision"],
                "validation_AP_baseline_prevalence": data["random_ranker_AP_reference"],
                "validation_macro_F1": (
                    data["validation_train_selected_threshold"]["macro_f1"]),
                "validation_precision": (
                    data["validation_train_selected_threshold"]["precision_error"]),
                "validation_recall": (
                    data["validation_train_selected_threshold"]["recall_error"]),
                "validation_FPR": (
                    data["validation_train_selected_threshold"]["false_positive_rate"]),
                "bootstrap_AP_interval": data["validation_prefix_bootstrap"]["average_precision"],
            } for task, data in result["tasks"].items()
        },
        "report_file": str(Path(args.output_dir) / "comparison.json"),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
