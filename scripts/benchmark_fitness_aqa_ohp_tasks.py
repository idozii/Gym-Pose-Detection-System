#!/usr/bin/env python3
"""OHP error-task static vs strictly causal temporal Random Forest benchmark.

Input: M13C independent, task-specific 30-feature OHP source TRAIN and VAL
CSVs plus their extraction summary. Original source TEST is NEVER opened.
Calibrates each model threshold on held-out source TRAIN VIDEO groups only.
Each comparison is performed on the same accepted validation frame IDs
WITHIN that task. Across tasks, frame populations are different.
Validation is repeatedly examined during development, so even bootstrap
intervals are *exploratory*, NOT an independent generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from gym_pose.ohp_task_features import FEATURES_BY_TASK, SCHEMA_BY_TASK, TASKS
from scripts.diagnose_fitness_aqa_thresholds import _metrics, _clip_any_metrics
from scripts.compare_fitness_aqa_temporal import choose_threshold
from scripts.extract_fitness_aqa_ohp_tasks import META_COLUMNS

TEMPORAL_STATS = ("causal_mean3", "delta1")
SIGNALS_BY_TASK = {
    task: FEATURES_BY_TASK[task][-6:] for task in TASKS
}
TEMPORAL_NAMES_BY_TASK = {
    task: tuple(f"{name}_{stat}" for name in SIGNALS_BY_TASK[task]
                for stat in TEMPORAL_STATS)
    for task in TASKS
}
TEMPORAL_SCHEMA_BY_TASK = {
    task: SCHEMA_BY_TASK[task] + "-causal12-v1" for task in TASKS
}
VARIANTS = ("static", "causal_temporal")
EXPECTED_SPLITS = ("train", "val")


def _load_data(features_dir: Path, *, require_full: bool) -> tuple[dict, dict]:
    path = features_dir / "fitness_aqa_ohp_task_extraction_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary.get("status") != "OHP_task_specific_pose_extraction_NO_TEST"
        or summary.get("source_split_type") !=
            "publisher_video_ID_disjoint_NOT_subject_disjoint"
        or summary.get("source_test_videos_processed") != 0):
        raise ValueError("Expected publisher source OHP TRAIN+VAL extraction, test untouched")
    if summary.get("schema_by_task") != SCHEMA_BY_TASK:
        raise ValueError("Extraction feature schema mismatch")
    if summary.get("feature_count_by_task") != {task: 30 for task in TASKS}:
        raise ValueError("Extraction feature count mismatch")
    if not (summary.get("sampling_fps") == 2.
            and math.isclose(summary.get("boundary_margin_seconds", -1), .1)
            and math.isclose(summary.get("min_visibility_on_required_joints", -1), .5)):
        raise ValueError("Expected declared M13C source sampling, margin and visibility")
    if require_full and summary.get("limit_per_split") != 0:
        raise ValueError("Pilot data must not be used for benchmarking; run full --max-videos-per-split 0")
    if set(summary.get("splits", {})) != set(EXPECTED_SPLITS):
        raise ValueError("Benchmark requires BOTH train and validation; TEST forbidden")
    out = {task: {} for task in TASKS}
    all_seen = {"train": set(), "val": set()}
    for split in EXPECTED_SPLITS:
        s = summary["splits"][split]
        if (s.get("videos_failed") != 0
            or s.get("videos_processed") != s.get("selected_videos")
            or s.get("eligible_frames", -1) <= 0
            or s.get("selected_videos", 0) < 1):
            raise ValueError(f"{split}: incomplete or failed source extraction")
        if require_full and s["selected_videos"] != {"train": 1582, "val": 339}[split]:
            raise ValueError(f"{split}: full official OHP clip split not extracted")
        for task in TASKS:
            f = features_dir / f"fitness_aqa_ohp_{task}_{split}_pose.csv"
            df = pd.read_csv(f)
            columns = (*META_COLUMNS, *FEATURES_BY_TASK[task])
            if list(df.columns) != list(columns) or df.empty:
                raise ValueError(f"{f}: missing/reordered columns or empty extracted frame data")
            if not df["split"].eq(split).all():
                raise ValueError(f"{f}: source split mismatch")
            if df[list(columns)].isna().any().any():
                raise ValueError(f"{f}: missing feature/metadata values")
            if (df["sample_id"].duplicated().any() or
                df.duplicated(["video_id", "frame_index"]).any()):
                raise ValueError(f"{f}: duplicate accepted frame or video/frame")
            if not df["video_id"].map(lambda v: isinstance(v, str) and bool(v)).all():
                raise ValueError(f"{f}: missing video IDs")
            if not df["frame_index"].ge(0).all() or not df["timestamp_s"].ge(0).all():
                raise ValueError(f"{f}: invalid source timestamps or frame index")
            if not df["source_fps"].gt(0).all():
                raise ValueError(f"{f}: invalid source FPS")
            expected_ids = (
                df["video_id"] + "_f" + df["frame_index"].astype(int).map(
                    lambda n: f"{n:07d}")
            )
            if not (df["sample_id"] == expected_ids).all():
                raise ValueError(f"{f}: sample_id/source frame mismatch")
            for target in (*TASKS, "any_of_two_annotated_errors"):
                if not df[target].isin((0, 1)).all():
                    raise ValueError(f"{f}: invalid target {target}")
            if not (df[list(TASKS)].max(axis=1) ==
                    df["any_of_two_annotated_errors"]).all():
                raise ValueError(f"{f}: incorrect any-error annotation")
            x = df.loc[:, list(FEATURES_BY_TASK[task])].to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise ValueError(f"{f}: nonfinite engineered features")
            stats = s["tasks"][task]
            positives = int(df[task].sum())
            if (len(df) != stats["accepted_frames"]
                or positives != stats["positive_frames"]
                or len(df) - positives != stats["negative_frames"]
                or int(df.video_id.nunique()) != stats["videos_with_accepted_frames"]
                or int(df.loc[df[task].eq(1), "video_id"].nunique()) !=
                    stats["videos_with_positive_frames"]
                or len(df) + stats["rejected_frames"] != s["eligible_frames"]):
                raise ValueError(f"{f}: source summary and CSV disagree")
            if set(df[task]) != {0, 1}:
                raise ValueError(f"{f}: both named-error classes required")
            out[task][split] = df
            all_seen[split] |= set(df.video_id)
    if all_seen["train"] & all_seen["val"]:
        raise ValueError("Publisher train/validation video-ID leakage")
    for task in TASKS:
        if set(out[task]["train"].video_id) & set(out[task]["val"].video_id):
            raise ValueError(f"{task}: train/val leakage")
    return out, summary


def task_features(df: pd.DataFrame, task: str, *, temporal: bool,
                  max_gap_seconds: float = .75) -> pd.DataFrame:
    """12 additional features: causal mean3/delta1 of six task geometry signals.

    The temporal history never accesses validation labels or future frames,
    and resets after missing accepted poses create a gap >0.75 seconds.
    """
    if task not in TASKS or not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Unknown task or invalid max_gap_seconds")
    static = df.loc[:, list(FEATURES_BY_TASK[task])].astype(float).copy()
    if not temporal:
        return static
    more = pd.DataFrame(
        np.zeros((len(df), len(TEMPORAL_NAMES_BY_TASK[task])), dtype=float),
        columns=TEMPORAL_NAMES_BY_TASK[task], index=df.index)
    for clip_id, clip in df.groupby("video_id", sort=False):
        clip = clip.sort_values(
            ["timestamp_s", "frame_index", "sample_id"], kind="stable")
        if clip["timestamp_s"].duplicated().any():
            raise ValueError(f"{clip_id}: duplicate timestamps in causal history")
        breaks = (clip["timestamp_s"].diff().fillna(0.) >
                  max_gap_seconds).cumsum()
        for _, part in clip.groupby(breaks, sort=False):
            for signal in SIGNALS_BY_TASK[task]:
                v = static.loc[part.index, signal]
                more.loc[part.index, f"{signal}_causal_mean3"] = (
                    v.rolling(3, min_periods=1).mean().to_numpy())
                more.loc[part.index, f"{signal}_delta1"] = v.diff().fillna(0.).to_numpy()
    if not np.isfinite(more.to_numpy()).all():
        raise ValueError(f"{task}: nonfinite temporal features")
    return pd.concat((static, more), axis=1)


def group_calibration(df: pd.DataFrame, *, task: str, seed: int,
                      min_cal_positive_videos: int = 3,
                      min_fit_positive_videos: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """One video-group split per task, reused for both variants.

    Check positive *video* counts as well as frame classes; avoids accidental
    selection of an operating threshold from only one annotated video.
    """
    if (min_cal_positive_videos < 2 or min_fit_positive_videos < 2
        or df.video_id.nunique() < 15):
        raise ValueError(f"{task}: insufficient clip groups for calibration")
    splitter = GroupShuffleSplit(
        n_splits=96, test_size=.25, random_state=seed)
    for fit, cal in splitter.split(df, groups=df["video_id"]):
        fit_df, cal_df = df.iloc[fit], df.iloc[cal]
        if set(fit_df.video_id) & set(cal_df.video_id):
            raise AssertionError("TRAIN fitting/calibration groups overlap")
        if (all(set(part[task]) == {0, 1} for part in (fit_df, cal_df))
            and fit_df.loc[fit_df[task].eq(1), "video_id"].nunique() >= min_fit_positive_videos
            and cal_df.loc[cal_df[task].eq(1), "video_id"].nunique() >= min_cal_positive_videos):
            return fit, cal
    raise ValueError(
        f"{task}: train-only video-group calibration lacks enough positive videos")


def model(seed: int, trees: int) -> RandomForestClassifier:
    if trees < 1:
        raise ValueError("trees must be positive")
    return RandomForestClassifier(
        n_estimators=trees, random_state=seed, n_jobs=-1,
        min_samples_leaf=2, class_weight="balanced")


def predict_positive(model_obj, x: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model_obj.classes_)
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise ValueError("Both target classes must be present in the fit")
    p = np.asarray(model_obj.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid predicted probabilities")
    return p


def score(y, p, threshold, groups) -> dict:
    if len(y) == 0 or set(y) != {0, 1}:
        raise ValueError("Scoring requires both target classes")
    return {
        **_metrics(y, p, threshold),
        "average_precision": float(average_precision_score(y, p)),
        "positive_prevalence": float(np.mean(y)),
        "positive_video_count": int(len(set(groups[np.asarray(y)==1]))),
        "clip_any_on_retained_frames_only": _clip_any_metrics(
            y, p, groups, threshold),
    }


def paired_video_bootstrap(y, video_ids, variant_probs: dict, thresholds: dict, *,
                           repetitions: int = 500, seed: int = 42) -> dict:
    """Paired cluster bootstrap over original publisher VAL clips.

    Both variants are evaluated on the EXACT same retained frames per task.
    A video sampled twice contributes all its frames twice (no independent-
    frame approximation). Excludes draws lacking both classes; report count.
    """
    if repetitions < 100:
        raise ValueError("Need >=100 bootstrap iterations")
    y = np.asarray(y, dtype=int)
    groups = np.asarray(video_ids, dtype=str)
    if len(y) != len(groups) or set(y) != {0, 1}:
        raise ValueError("Paired bootstrap needs both classes and clip IDs")
    if set(variant_probs) != set(VARIANTS) or set(thresholds) != set(VARIANTS):
        raise ValueError("Both variants required for paired video bootstrap")
    for name in VARIANTS:
        if len(variant_probs[name]) != len(y):
            raise ValueError("Variant predictions have different validation frames")
    unique = np.unique(groups)
    indexed = {key: np.flatnonzero(groups == key) for key in unique}
    rng = np.random.default_rng(seed)
    rows = {name: {"ap": [], "macro_f1": []} for name in VARIANTS}
    rows["temporal_minus_static"] = {"ap": [], "macro_f1": []}
    skipped = 0
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indexed[key] for key in sampled])
        yy = y[idx]
        if set(yy) != {0, 1}:
            skipped += 1
            continue
        one = {}
        for name in VARIANTS:
            p = np.asarray(variant_probs[name], dtype=float)[idx]
            one[name] = {
                "ap": float(average_precision_score(yy, p)),
                "macro_f1": float(f1_score(
                    yy, p >= thresholds[name], average="macro",
                    labels=[0, 1], zero_division=0)),
            }
            for metric, value in one[name].items():
                rows[name][metric].append(value)
        for metric in ("ap", "macro_f1"):
            rows["temporal_minus_static"][metric].append(
                one["causal_temporal"][metric] - one["static"][metric])
    if repetitions - skipped < repetitions * .8:
        raise ValueError("Too many bootstrap resamples lack both classes")
    intervals = {
        name: {
            metric: {
                "percentile_2_5": float(np.percentile(values, 2.5)),
                "percentile_97_5": float(np.percentile(values, 97.5)),
                "median": float(np.median(values)),
            } for metric, values in data.items()
        }
        for name, data in rows.items()
    }
    return {
        "unit": "original_source_validation_video_cluster",
        "paired_same_frame_population": True,
        "requested_resamples": repetitions, "valid_resamples": repetitions - skipped,
        "skipped_single_class_resamples": skipped,
        "validation_video_count": int(len(unique)),
        "intervals": intervals,
        "warnings": [
            "Percentile bootstrap is exploratory; it cannot repair repeated development on validation",
            "Publisher subject identity unavailable; resampling clips is not subject-independent",
            "Thresholds were selected from a single disjoint source TRAIN calibration split",
        ],
    }


def benchmark(feature_dir: str | Path, output_dir: str | Path, *,
              seed: int = 42, trees: int = 300,
              max_gap_seconds: float = .75, bootstraps: int = 500,
              require_full: bool = True) -> dict:
    if trees < 1 or bootstraps < 100:
        raise ValueError("Invalid trees or bootstraps")
    frames, source = _load_data(Path(feature_dir), require_full=require_full)
    output_dir = Path(output_dir)
    report = {
        "status": "ohp_task_specific_static_vs_causal_temporal_train_calibrated_NO_TEST",
        "feature_count": {"static": 30, "causal_temporal": 42},
        "source": "Fitness-AQA overhead press train/validation only",
        "source_extraction_summary": "fitness_aqa_ohp_task_extraction_summary.json",
        "seed": seed, "trees": trees, "max_gap_seconds": max_gap_seconds,
        "validation_used_for_threshold_selection": False,
        "test_data_accessed": False,
        "tasks": {},
        "limitations": [
            "Same-frame static vs temporal comparison is paired only WITHIN each target's retained-frame population",
            "Elbow vs knee performance should not be directly compared: different gates, source frames, and prevalence",
            "Video-group calibration uses TRAIN labels only; not athlete/subject-disjoint",
            "Validation has been repeatedly examined in prior milestones: estimates are exploratory",
            "Frame errors are correlated within clips; video bootstrap helps but does not remove subject leakage",
            "Only two named annotation errors; zero means unannotated error, NOT globally correct technique",
            "Pose-abstained frames are NOT true negatives; report coverage separately",
            "Missing MediaPipe poses may bias class prevalence; test remains locked",
            "No production-ready fitness coaching or medical injury-prevention claim",
        ],
    }
    bundles = {}
    for task_index, task in enumerate(TASKS):
        train, val = frames[task]["train"], frames[task]["val"]
        if (train.loc[train[task].eq(1), "video_id"].nunique() < 15
            or val.loc[val[task].eq(1), "video_id"].nunique() < 5
            or int(val[task].sum()) < 10):
            raise ValueError(f"{task}: insufficient positive-source videos/frames for a benchmark")
        fit_idx, cal_idx = group_calibration(train, task=task, seed=seed + task_index)
        train_inputs = {
            variant: task_features(train, task, temporal=(variant != "static"),
                                   max_gap_seconds=max_gap_seconds)
            for variant in VARIANTS
        }
        val_inputs = {
            variant: task_features(val, task, temporal=(variant != "static"),
                                   max_gap_seconds=max_gap_seconds)
            for variant in VARIANTS
        }
        ytrain = train[task].to_numpy(dtype=int)
        yval = val[task].to_numpy(dtype=int)
        video_ids = val.video_id.to_numpy(dtype=str)
        thresholds, predictions = {}, {}
        result = {
            "schema_version": SCHEMA_BY_TASK[task],
            "train_frames": int(len(train)), "train_videos": int(train.video_id.nunique()),
            "train_positive_frames": int(ytrain.sum()),
            "train_positive_videos": int(train.loc[train[task].eq(1), "video_id"].nunique()),
            "validation_frames": int(len(val)), "validation_videos": int(val.video_id.nunique()),
            "validation_positive_frames": int(yval.sum()),
            "validation_positive_videos": int(val.loc[val[task].eq(1), "video_id"].nunique()),
            "validation_positive_prevalence": float(yval.mean()),
            "coverage_train": source["splits"]["train"]["tasks"][task]["pose_coverage_eligible"],
            "coverage_validation": source["splits"]["val"]["tasks"][task]["pose_coverage_eligible"],
            "calibration": {
                "fit_frames": int(len(fit_idx)), "calibration_frames": int(len(cal_idx)),
                "fit_videos": int(train.iloc[fit_idx].video_id.nunique()),
                "calibration_videos": int(train.iloc[cal_idx].video_id.nunique()),
                "fit_positive_videos": int(train.iloc[fit_idx].loc[
                    train.iloc[fit_idx][task].eq(1), "video_id"].nunique()),
                "calibration_positive_videos": int(train.iloc[cal_idx].loc[
                    train.iloc[cal_idx][task].eq(1), "video_id"].nunique()),
                "disjoint": True,
                "clip_group_hash_sha256": hashlib.sha256((
                    "|".join(sorted(set(train.iloc[fit_idx].video_id))) + "--" +
                    "|".join(sorted(set(train.iloc[cal_idx].video_id)))
                ).encode()).hexdigest(),
                "method": "single GroupShuffleSplit holdout of source TRAIN clip IDs",
            },
            "variants": {},
        }
        for variant in VARIANTS:
            xtrain, xval = train_inputs[variant], val_inputs[variant]
            provisional = model(seed, trees)
            provisional.fit(xtrain.iloc[fit_idx], ytrain[fit_idx])
            cal_prob = predict_positive(provisional, xtrain.iloc[cal_idx])
            calibration = choose_threshold(ytrain[cal_idx], cal_prob)
            threshold = float(calibration["threshold"])
            final = model(seed, trees)
            final.fit(xtrain, ytrain)
            prob = predict_positive(final, xval)
            thresholds[variant], predictions[variant] = threshold, prob
            result["variants"][variant] = {
                "schema_version": (
                    TEMPORAL_SCHEMA_BY_TASK[task]
                    if variant == "causal_temporal" else SCHEMA_BY_TASK[task]),
                "features_count": len(xtrain.columns),
                "feature_names": list(xtrain.columns),
                "train_only_threshold": threshold,
                "train_calibration_macro_f1": float(calibration["macro_f1"]),
                "validation_fixed_0_5": score(yval, prob, .5, video_ids),
                "validation_train_calibrated": score(yval, prob, threshold, video_ids),
            }
            bundles[f"{task}_{variant}"] = {
                "model": final, "exercise": "ohp",
                "target": task, "variant": variant,
                "feature_names": list(xtrain.columns),
                "schema_version": result["variants"][variant]["schema_version"],
                "classes": [int(c) for c in final.classes_],
                "threshold_selected_on_train_groups_only": threshold,
                "max_gap_seconds": max_gap_seconds if variant == "causal_temporal" else None,
                "source_train_only": True, "publisher_test_untouched": True,
            }
        result["paired_validation_cluster_bootstrap"] = paired_video_bootstrap(
            yval, video_ids, predictions, thresholds,
            repetitions=bootstraps, seed=seed + task_index + 100)
        report["tasks"][task] = result
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save only locally trusted fits in artifacts/, never to Git.
    for key, payload in bundles.items():
        joblib.dump(payload, output_dir / f"{key}.joblib")
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path,
                        default=Path("data/features/fitness_aqa_ohp_tasks"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/fitness_aqa_ohp_task_rf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-gap-seconds", type=float, default=.75)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--allow-pilot", action="store_true",
                        help="For synthetic tests only. Do NOT publish pilot-derived scores.")
    args = parser.parse_args()
    try:
        report = benchmark(
            args.features_dir, args.output_dir, seed=args.seed, trees=args.trees,
            max_gap_seconds=args.max_gap_seconds, bootstraps=args.bootstraps,
            require_full=not args.allow_pilot)
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError) as exc:
        parser.exit(1, f"OHP benchmark failed: {exc}\n")
    compact = {
        "status": report["status"],
        "test_data_accessed": report["test_data_accessed"],
        "tasks": {
            task: {
                "train_frames": entry["train_frames"],
                "train_positive_videos": entry["train_positive_videos"],
                "validation_frames": entry["validation_frames"],
                "validation_positive_frames": entry["validation_positive_frames"],
                "validation_positive_videos": entry["validation_positive_videos"],
                "coverage_train": entry["coverage_train"],
                "coverage_validation": entry["coverage_validation"],
                "calibration": entry["calibration"],
                "variants": {
                    name: {
                        "threshold_train": v["train_only_threshold"],
                        "validation_AP": v["validation_train_calibrated"]["average_precision"],
                        "validation_macro_F1": v["validation_train_calibrated"]["macro_f1"],
                        "validation_recall": v["validation_train_calibrated"]["recall_error"],
                        "validation_precision": v["validation_train_calibrated"]["precision_error"],
                        "validation_FPR": v["validation_train_calibrated"]["false_positive_rate"],
                    } for name, v in entry["variants"].items()},
                "paired_bootstrap_temporal_minus_static":
                    entry["paired_validation_cluster_bootstrap"]["intervals"]["temporal_minus_static"],
            } for task, entry in report["tasks"].items(),
        },
        "full_report": str(args.output_dir / "comparison.json"),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
