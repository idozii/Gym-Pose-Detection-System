#!/usr/bin/env python3
"""Package genuine Fitness-AQA Squat publisher knee-error RFs for the registry.

This is NOT the old 'correct/incorrect' notebook binary classifier.
Both labels are from named original source video timestamp annotations.
Only the already-extracted original publisher TRAIN/VAL pose CSVs and
TRAIN/VAL extraction summary are read; test CSV/video is NEVER opened.
One TRAIN clip-group holdout calibrates threshold; final model refits
on all source TRAIN. Validation is descriptive and repeatedly used.
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames
from scripts.compare_fitness_aqa_temporal import choose_threshold
from scripts.diagnose_fitness_aqa_thresholds import _metrics

FULL_VIDEO_COUNTS = {"train": 1136, "val": 243}
MANIFEST_NAME = "fitness_aqa_extraction_summary.json"
META_COLUMNS = (
    "sample_id", "video_id", "split", "timestamp_s", "frame_index",
    "source_fps", "error_knees_forward", "error_knees_inward",
    "any_of_two_annotated_errors",
)


def load_source(features_dir: str | Path, *, require_full: bool = True) -> tuple[dict, dict]:
    """Validate source provenance/CSV accounting without reading publisher TEST."""
    root = Path(features_dir)
    summary = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (summary.get("dataset") != "Fitness-AQA squat labeled"
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("split_type") != "publisher_video_id_split_NOT_subject_disjoint"):
        raise ValueError("Expected original publisher Squat source extraction schema")
    # Refuse an extraction manifest recording already-processed TEST media.
    if set(summary.get("splits", {})) != {"train", "val"}:
        raise ValueError("Expected only TRAIN+VAL extraction; source TEST must be untouched")
    if (not math.isclose(summary.get("sampling_fps", -1), 2.)
        or not math.isclose(summary.get("boundary_margin_seconds", -1), .1)
        or not math.isclose(summary.get("min_pose_visibility", -1), .5)):
        raise ValueError("Source pose sampling/visibility contract mismatch")
    if require_full and summary.get("limit_per_split") != 0:
        raise ValueError("Pilot not eligible for official source-error model packaging")
    data = {}
    all_ids = {}
    for split in ("train", "val"):
        stats = summary["splits"][split]
        if (stats.get("videos_failed", -1) != 0
            or stats.get("videos_processed") != stats.get("selected_videos")
            or (require_full and stats.get("selected_videos") != FULL_VIDEO_COUNTS[split])):
            raise ValueError(f"{split}: incomplete original publisher clip extraction")
        df = load_frames(root / f"fitness_aqa_{split}_pose.csv", split)
        if list(df.columns) != list((*META_COLUMNS, *FEATURE_NAMES)):
            raise ValueError(f"{split}: unexpected feature names/order")
        if len(df) != stats.get("pose_accepted"):
            raise ValueError(f"{split}: pose accepted count disagrees with source CSV")
        eligible = stats.get("frames_sampled", 0) - stats.get("boundary_rejected", 0)
        if (eligible < 1 or len(df) + stats.get("pose_rejected", -1) != eligible):
            raise ValueError(f"{split}: selected/eligible source frame counts inconsistent")
        if (int(df.error_knees_forward.sum()) != stats.get("annotated_knees_forward_frames")
            or int(df.error_knees_inward.sum()) != stats.get("annotated_knees_inward_frames")):
            raise ValueError(f"{split}: positive source annotations disagree with manifest")
        expected_sample_ids = (
            df.video_id.astype(str) + "_f" +
            df.frame_index.astype(int).map(lambda index: f"{index:07d}")
        )
        if not df.sample_id.eq(expected_sample_ids).all():
            raise ValueError(f"{split}: source frame/sample ID mismatch")
        all_ids[split] = set(df.video_id)
        data[split] = df
    if all_ids["train"] & all_ids["val"]:
        raise ValueError("Publisher source train/val VIDEO ID overlap")
    for target in TARGETS:
        for split in ("train", "val"):
            df = data[split]
            positives = int(df[target].sum())
            positive_clips = int(df.loc[df[target].eq(1), "video_id"].nunique())
            if not 20 <= positives < len(df) or positive_clips < 5:
                raise ValueError(f"{split} {target}: insufficient original positive clips/frames")
    return data, summary


def train_group_holdout(
    train: pd.DataFrame, target: str, *, seed: int = 42,
    min_fit_positive_videos: int = 5,
    min_cal_positive_videos: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ONLY publisher TRAIN source videos; never individual adjacent frames."""
    if target not in TARGETS:
        raise ValueError("Unknown Squat publisher error annotation")
    if train.video_id.nunique() < 15:
        raise ValueError("Insufficient TRAIN video groups")
    splitter = GroupShuffleSplit(n_splits=96, test_size=.25, random_state=seed)
    for fit, cal in splitter.split(train, groups=train.video_id):
        a, b = train.iloc[fit], train.iloc[cal]
        if set(a.video_id) & set(b.video_id):
            raise AssertionError("TRAIN group leakage")
        if (set(a[target]) == {0, 1} and set(b[target]) == {0, 1}
            and a.loc[a[target].eq(1), "video_id"].nunique() >= min_fit_positive_videos
            and b.loc[b[target].eq(1), "video_id"].nunique() >= min_cal_positive_videos):
            return fit, cal
    raise ValueError(f"{target}: insufficient TRAIN error-positive VIDEO groups for calibration")


def rf_model(*, seed: int, trees: int):
    if trees < 1:
        raise ValueError("Need at least one RF tree")
    return RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2,
        class_weight="balanced", random_state=seed, n_jobs=-1)


def predict_probs(fitted, features: pd.DataFrame) -> np.ndarray:
    if not np.array_equal(fitted.classes_, [0, 1]):
        raise ValueError("Model must have ordered original error labels [0,1]")
    fitted.n_jobs = 1
    p = np.asarray(fitted.predict_proba(features)[:, 1], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid classifier probability")
    return p


def train_models(
    features_dir: str | Path, output_dir: str | Path, *,
    seed: int = 42, trees: int = 300, require_full: bool = True,
) -> dict:
    if (not isinstance(seed, int) or seed < 0 or seed >= 2**32 - 2
        or trees < 1):
        raise ValueError("Invalid RF seed or tree count")
    frames, manifest = load_source(features_dir, require_full=require_full)
    train, val = frames["train"], frames["val"]
    x_train = train.loc[:, list(FEATURE_NAMES)].astype(float)
    x_val = val.loc[:, list(FEATURE_NAMES)].astype(float)
    report = {
        "status": "squat_original_source_errors_TRAIN_group_calibrated_VAL_exploratory_NO_TEST",
        "source": "Fitness-AQA publisher Squat video-ID TRAIN/VAL",
        "original_publisher_TEST_media_or_CSV_accessed": False,
        "thresholds_selected_on_source_TRAIN_only": True,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "rf_trees": trees, "seed": seed,
        "source_pose_coverage_train": (
            manifest["splits"]["train"]["pose_accepted"] /
            (manifest["splits"]["train"]["frames_sampled"] -
             manifest["splits"]["train"]["boundary_rejected"])),
        "source_pose_coverage_validation": (
            manifest["splits"]["val"]["pose_accepted"] /
            (manifest["splits"]["val"]["frames_sampled"] -
             manifest["splits"]["val"]["boundary_rejected"])),
        "tasks": {},
        "limitations": [
            "Error labels are original named time intervals, NOT the notebook's correct/incorrect labels",
            "Test CSV/videos are never opened; only original publisher source TRAIN and VAL",
            "Publisher clips are video-ID-disjoint, NOT verified subject-independent",
            "TRAIN-only clip groups calibrate thresholds; validation has been repeatedly consulted",
            "Static 103 features only; no causal temporal state or video-time continuity in registry",
            "Errors unannotated at a timestamp do NOT prove correct form",
            "Pose quality abstains on some source frames; evaluation only includes accepted poses",
            "These are research source-label correlations, NOT safe form or medical risk indicators",
        ],
    }
    bundles = {}
    for index, target in enumerate(TARGETS):
        fit, cal = train_group_holdout(train, target, seed=seed + index)
        y_train = train[target].to_numpy(dtype=int)
        y_val = val[target].to_numpy(dtype=int)
        provisional = rf_model(seed=seed, trees=trees)
        provisional.fit(x_train.iloc[fit], y_train[fit])
        cal_probability = predict_probs(provisional, x_train.iloc[cal])
        operating_point = choose_threshold(y_train[cal], cal_probability)
        threshold = float(operating_point["threshold"])
        final = rf_model(seed=seed, trees=trees)
        final.fit(x_train, y_train)
        val_probability = predict_probs(final, x_val)
        fit_groups = sorted(set(train.iloc[fit].video_id))
        cal_groups = sorted(set(train.iloc[cal].video_id))
        report["tasks"][target] = {
            "target": target,
            "train_frames": int(len(train)),
            "validation_frames": int(len(val)),
            "train_positive_frames": int(y_train.sum()),
            "train_positive_videos": int(train.loc[train[target].eq(1), "video_id"].nunique()),
            "validation_positive_frames": int(y_val.sum()),
            "validation_positive_videos": int(val.loc[val[target].eq(1), "video_id"].nunique()),
            "train_calibration": {
                "threshold": threshold,
                "fit_frames": int(len(fit)),
                "calibration_frames": int(len(cal)),
                "fit_videos": len(fit_groups),
                "calibration_videos": len(cal_groups),
                "fit_positive_videos": int(train.iloc[fit].loc[
                    train.iloc[fit][target].eq(1), "video_id"].nunique()),
                "calibration_positive_videos": int(train.iloc[cal].loc[
                    train.iloc[cal][target].eq(1), "video_id"].nunique()),
                "video_groups_disjoint": True,
                "clip_group_sha256": hashlib.sha256((
                    "|".join(fit_groups) + "--" + "|".join(cal_groups)
                ).encode()).hexdigest(),
            },
            "validation_source_label_prevalence": float(y_val.mean()),
            "validation_AP": float(np.clip(average_precision_score(y_val, val_probability), 0, 1)),
            "validation_fixed_0_5": _metrics(y_val, val_probability, .5),
            "validation_train_calibrated": _metrics(y_val, val_probability, threshold),
        }
        bundles[target] = {
            "model": final,
            "exercise": "squat", "target": target,
            "variant": "static",
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "classes": [int(c) for c in final.classes_],
            "source_train_only": True,
            "publisher_test_untouched": True,
            "threshold_selected_on_train_groups_only": threshold,
            "annotation_semantics": "publisher_named_error_interval_NOT_correctness",
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for target, bundle in bundles.items():
        joblib.dump(bundle, output / f"rf_{target}_static_source.joblib")
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/fitness_aqa_squat_source_rf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--allow-pilot", action="store_true",
                        help="Synthetic tests ONLY; never use pilot scores as real benchmark")
    args = parser.parse_args()
    try:
        report = train_models(
            args.features_dir, args.output_dir, seed=args.seed, trees=args.trees,
            require_full=not args.allow_pilot)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Squat source error model packaging failed: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "publisher_test_accessed": report["original_publisher_TEST_media_or_CSV_accessed"],
        "source_pose_coverage_train": report["source_pose_coverage_train"],
        "source_pose_coverage_validation": report["source_pose_coverage_validation"],
        "tasks": {
            target: {
                "train_positive_videos": entry["train_positive_videos"],
                "validation_positive_frames": entry["validation_positive_frames"],
                "validation_positive_videos": entry["validation_positive_videos"],
                "threshold_source_train": entry["train_calibration"]["threshold"],
                "validation_AP": entry["validation_AP"],
                "validation_AP_random_reference": entry["validation_source_label_prevalence"],
                "validation_macro_F1": entry["validation_train_calibrated"]["macro_f1"],
                "validation_precision": entry["validation_train_calibrated"]["precision_error"],
                "validation_recall": entry["validation_train_calibrated"]["recall_error"],
                "validation_FPR": entry["validation_train_calibrated"]["false_positive_rate"],
            } for target, entry in report["tasks"].items()
        },
        "report_file": str(Path(args.output_dir) / "validation_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
