#!/usr/bin/env python3
"""Fitness-AQA V2 ablation: static poses vs causal short-history features.

Calibrate per-class thresholds on *train videos only* via a video-group holdout.
Fit final RFs on the entire publisher train split; report validation outcomes.
The source publisher's clip-disjoint splits are NOT verified subject-disjoint.
Never read the test CSV, never select thresholds by validation F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames
from scripts.diagnose_fitness_aqa_thresholds import _clip_any_metrics, _metrics, THRESHOLDS

SIGNALS = (
    "knee_angle_mean", "hip_angle_mean", "hip_to_shoulder_ratio",
    "left_knee_minus_ankle_x", "right_knee_minus_ankle_x",
    "knee_width_x", "torso_sin", "torso_cos",
)
TEMPORAL_NAMES = tuple(
    f"{name}_{stat}" for name in SIGNALS for stat in ("causal_mean3", "delta1")
)
TEMPORAL_SCHEMA = "squat-103-v1-plus-causal-16-v1"


def features(df: pd.DataFrame, *, temporal: bool, max_gap_seconds: float = 0.75) -> pd.DataFrame:
    """Rolling means/deltas use only current+past frames from the same clip/segment.

    Source extraction is nominally 2 FPS; time gaps >0.75 s reset the history
    to prevent a missing-pose interval looking like uninterrupted movement.
    """
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    x = df.loc[:, list(FEATURE_NAMES)].astype(float).copy()
    if not temporal:
        return x
    signals = pd.DataFrame(index=df.index)
    for label in ("knee_angle_mean", "hip_angle_mean", "hip_to_shoulder_ratio"):
        signals[label] = x[label]
    signals["left_knee_minus_ankle_x"] = x["lm_25_x"] - x["lm_27_x"]
    signals["right_knee_minus_ankle_x"] = x["lm_26_x"] - x["lm_28_x"]
    signals["knee_width_x"] = (x["lm_25_x"] - x["lm_26_x"]).abs()
    angle = np.deg2rad(x["torso_incline_deg"].to_numpy())
    signals["torso_sin"] = np.sin(angle)
    signals["torso_cos"] = np.cos(angle)
    derived = pd.DataFrame(np.zeros((len(df), len(TEMPORAL_NAMES)), dtype=float),
                           index=df.index, columns=TEMPORAL_NAMES)
    for _key, clip in df.groupby("video_id", sort=False):
        ordered = clip.sort_values(["timestamp_s", "frame_index", "sample_id"], kind="stable")
        times = ordered["timestamp_s"]
        if times.duplicated().any():
            raise ValueError("Duplicate timestamp in video; cannot build causal history")
        segments = (times.diff().fillna(0.0) > max_gap_seconds).cumsum()
        for _seg, part in ordered.groupby(segments, sort=False):
            ids = part.index
            for signal in SIGNALS:
                values = signals.loc[ids, signal]
                derived.loc[ids, f"{signal}_causal_mean3"] = (
                    values.rolling(3, min_periods=1).mean().to_numpy())
                derived.loc[ids, f"{signal}_delta1"] = (
                    values.diff().fillna(0.0).to_numpy())
    if not np.isfinite(derived.to_numpy()).all():
        raise ValueError("Temporal features contain non-finite values")
    return pd.concat([x, derived], axis=1)


def grouped_calibration_split(df: pd.DataFrame, *, seed: int = 42,
                              fraction: float = .25) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 1 or df.video_id.nunique() < 8:
        raise ValueError("Need >=8 videos and a valid calibration fraction")
    splitter = GroupShuffleSplit(n_splits=24, test_size=fraction, random_state=seed)
    for fit, cal in splitter.split(df, groups=df.video_id):
        if set(df.iloc[fit].video_id) & set(df.iloc[cal].video_id):
            raise AssertionError("Train/calibration clip overlap")
        if all(set(df.iloc[idx][target]) == {0, 1}
               for target in TARGETS for idx in (fit, cal)):
            return fit, cal
    raise ValueError("Could not create video-disjoint calibration with both classes for both targets")


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> dict:
    """Choose macro-F1 operating point on train-only calibration predictions."""
    candidates = sorted(set((*THRESHOLDS, .5)))
    rows = [_metrics(y, probabilities, t) for t in candidates]
    chosen = sorted(rows, key=lambda r: (-r["macro_f1"], abs(r["threshold"] - .5),
                                         r["threshold"]))[0]
    return chosen


def run(input_dir: str | Path, output_dir: str | Path, *, seed: int = 42,
        n_estimators: int = 300, max_gap_seconds: float = .75) -> dict:
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    train = load_frames(input_dir / "fitness_aqa_train_pose.csv", "train")
    val = load_frames(input_dir / "fitness_aqa_val_pose.csv", "val")
    if set(train.video_id) & set(val.video_id):
        raise ValueError("Train/val video overlap")
    fit_idx, cal_idx = grouped_calibration_split(train, seed=seed)
    if len(train) < 50 or len(val) < 20:
        raise ValueError("Too few observations for useful model comparison")
    result = {
        "status": "train_group_calibrated_validation_evaluation_ONLY",
        "schema_version": SCHEMA_VERSION, "temporal_schema": TEMPORAL_SCHEMA,
        "seed": seed, "n_estimators": n_estimators,
        "history": "3 causal valid frames; resets when timestamp gap exceeds max_gap_seconds",
        "max_gap_seconds": max_gap_seconds,
        "train_frames": len(train), "validation_frames": len(val),
        "train_videos": int(train.video_id.nunique()),
        "calibration_videos": int(train.iloc[cal_idx].video_id.nunique()),
        "fit_videos": int(train.iloc[fit_idx].video_id.nunique()),
        "validation_videos": int(val.video_id.nunique()),
        "calibration_and_fit_video_disjoint": True,
        "variants": {}, "warnings": [
            "A single group calibration holdout is high-variance; repeat across train seeds later",
            "Thresholds selected using train clips only, not validation",
            "Compare variants on validation descriptively; further model selection on validation causes overfitting",
            "Publisher video splits are not verified subject-independent",
            "Roughly half of eligible sampled frames were rejected by pose quality",
            "Clip-any labels only describe accepted-frame annotations, not all video error intervals",
            "Temporal features are causal across accepted frames but extracted using static-image MediaPipe detection",
            "Models here use error-specific labels, not global squat form; not compatible with existing webcam CLI",
        ],
    }
    models = {}
    for variant in ("static", "causal_temporal"):
        temporal = variant != "static"
        x_train = features(train, temporal=temporal, max_gap_seconds=max_gap_seconds)
        x_val = features(val, temporal=temporal, max_gap_seconds=max_gap_seconds)
        result["variants"][variant] = {"features_count": len(x_train.columns), "tasks": {}}
        for target in TARGETS:
            y = train[target].to_numpy(dtype=int)
            y_val = val[target].to_numpy(dtype=int)
            if any(set(y[idx]) != {0, 1} for idx in (fit_idx, cal_idx)):
                raise ValueError(f"{target}: invalid fit/calibration class balance")
            if set(y_val) != {0, 1}:
                raise ValueError(f"{target}: validation must include both classes")
            temp = RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )
            temp.fit(x_train.iloc[fit_idx], y[fit_idx])
            cal_prob = temp.predict_proba(x_train.iloc[cal_idx])[:, 1]
            selected = choose_threshold(y[cal_idx], cal_prob)
            threshold = float(selected["threshold"])
            final = RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )
            final.fit(x_train, y)
            val_prob = final.predict_proba(x_val)[:, 1]
            if not np.isfinite(val_prob).all():
                raise ValueError(f"{target}: invalid predicted probabilities")
            result["variants"][variant]["tasks"][target] = {
                "train_calibrated_threshold": threshold,
                "calibration_frame_metrics": selected,
                "validation_positive_frames": int(y_val.sum()),
                "validation_average_precision": float(average_precision_score(y_val, val_prob)),
                "validation_positive_prevalence": float(y_val.mean()),
                "validation_fixed_0_5": _metrics(y_val, val_prob, .5),
                "validation_train_calibrated_threshold": _metrics(y_val, val_prob, threshold),
                "validation_clip_any_fixed_0_5": _clip_any_metrics(
                    y_val, val_prob, val.video_id.to_numpy(dtype=str), .5),
                "validation_clip_any_calibrated": _clip_any_metrics(
                    y_val, val_prob, val.video_id.to_numpy(dtype=str), threshold),
            }
            models[f"{variant}_{target}"] = {
                "model": final, "target": target, "variant": variant,
                "feature_names": list(x_train.columns),
                "schema_version": SCHEMA_VERSION if not temporal else TEMPORAL_SCHEMA,
                "threshold_selected_from_train_only": threshold,
                "classes": list(map(int, final.classes_)),
                "max_gap_seconds": max_gap_seconds if temporal else None,
                "exercise": "squat", "evaluation_source": "Fitness-AQA source validation",
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in models.items():
        joblib.dump(bundle, output_dir / f"{name}.joblib")
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fitness_aqa_temporal"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gap-seconds", type=float, default=.75)
    args = parser.parse_args()
    try:
        report = run(args.features_dir, args.output_dir, seed=args.seed,
                     max_gap_seconds=args.max_gap_seconds)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    compact = {
        "status": report["status"], "train_frames": report["train_frames"],
        "validation_frames": report["validation_frames"],
        "calibration_videos": report["calibration_videos"],
        "variants": {
            name: {target: {
                "train_calibrated_threshold": task["train_calibrated_threshold"],
                "validation_ap": round(task["validation_average_precision"], 4),
                "validation_macro_f1": round(
                    task["validation_train_calibrated_threshold"]["macro_f1"], 4),
                "validation_recall": round(
                    task["validation_train_calibrated_threshold"]["recall_error"], 4),
                "validation_precision": round(
                    task["validation_train_calibrated_threshold"]["precision_error"], 4),
                "validation_false_positives": task[
                    "validation_train_calibrated_threshold"]["false_positive"],
            } for target, task in detail["tasks"].items()}
            for name, detail in report["variants"].items()},
        "report": str(args.output_dir / "comparison.json"),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
