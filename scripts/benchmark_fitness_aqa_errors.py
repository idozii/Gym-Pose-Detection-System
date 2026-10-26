#!/usr/bin/env python3
"""Train/evaluate two error-specific Random Forest baselines on train/validation only.

Never load test CSV. Do not use accuracy/ROC curves from a tiny pilot as
scientific evidence; minimum positive sample and positive clip counts enforced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames


def _evaluation(truth, guesses, confidence, video_ids) -> dict:
    """Report frame outcomes and clip-macro metrics (with their sample counts)."""
    actual = np.asarray(truth, dtype=int)
    predicted = np.asarray(guesses, dtype=int)
    prob = np.asarray(confidence, dtype=float)
    clips = np.asarray(video_ids, dtype=str)
    if len(actual) == 0 or len(actual) != len(predicted) or len(actual) != len(clips):
        raise ValueError("Predictions/labels must be nonempty and equal length")
    by_clip = []
    for key in sorted(set(clips)):
        use = clips == key
        by_clip.append({
            "video_id": key, "frames": int(use.sum()),
            "positive_frames": int(actual[use].sum()),
            "macro_f1": float(f1_score(actual[use], predicted[use], labels=[0, 1],
                                       average="macro", zero_division=0)),
        })
    return {
        "frames": len(actual), "videos": len(by_clip),
        "positives": int(actual.sum()), "negative_frames": int((actual == 0).sum()),
        "macro_f1_frame": float(f1_score(actual, predicted, labels=[0, 1], average="macro",
                                        zero_division=0)),
        "precision_error": float(precision_score(actual, predicted, zero_division=0)),
        "recall_error": float(recall_score(actual, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "average_precision": float(average_precision_score(actual, prob)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(actual, predicted, labels=[0, 1],
                                                       output_dict=True, zero_division=0),
        "mean_video_macro_f1": float(np.mean([r["macro_f1"] for r in by_clip])),
        "per_video_metrics": by_clip,
    }


def benchmark(features_dir: Path, output_dir: Path, *, seed: int = 42,
              min_positive_frames: int = 20, min_positive_videos: int = 5,
              n_estimators: int = 300) -> dict:
    if min_positive_frames < 1 or min_positive_videos < 1 or n_estimators < 1:
        raise ValueError("Positive sample/clip thresholds and tree count must be positive")
    train_path = features_dir / "fitness_aqa_train_pose.csv"
    val_path = features_dir / "fitness_aqa_val_pose.csv"
    train = load_frames(train_path, "train")
    val = load_frames(val_path, "val")
    if set(train.video_id) & set(val.video_id):
        raise ValueError("Leakage: train and validation share a video_id")
    if set(train.sample_id) & set(val.sample_id):
        raise ValueError("Leakage: train and validation share a sample_id")

    result = {
        "status": "validation_only_NO_test_used",
        "dataset": "Fitness-AQA squat labeled",
        "split": "publisher_video_id_split_NOT_subject_disjoint",
        "schema_version": SCHEMA_VERSION,
        "seed": seed, "train_frames": len(train), "validation_frames": len(val),
        "train_videos": int(train.video_id.nunique()),
        "validation_videos": int(val.video_id.nunique()),
        "tasks": {},
        "limitations": [
            "Error labels are time intervals, not overall exercise correctness",
            "No subject identity is verified in these official video splits",
            "Frame-level observations are correlated within video",
            "Model selection may use validation only; keep test untouched",
            "Pose visibility and annotation-boundary filtering change class coverage",
            "RF uses isolated frames; temporal modeling is outside this baseline",
        ],
    }
    # Fail before writing any model when even one task is too sparse, avoiding
    # seemingly impressive benchmarks obtained from the tiny pilot.
    for label in TARGETS:
        for name, df in (("train", train), ("val", val)):
            positive = int(df[label].sum())
            positive_videos = int(df.loc[df[label] == 1, "video_id"].nunique())
            negatives = int((df[label] == 0).sum())
            if positive < min_positive_frames or positive_videos < min_positive_videos:
                raise ValueError(
                    f"{label} {name}: {positive} positive frames across {positive_videos} "
                    f"videos. Need >= {min_positive_frames} frames, "
                    f">= {min_positive_videos} videos before training.")
            if negatives < min_positive_frames:
                raise ValueError(f"{label} {name}: too few negative frames ({negatives})")
    features_train = train.loc[:, list(FEATURE_NAMES)].astype(float)
    features_val = val.loc[:, list(FEATURE_NAMES)].astype(float)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in TARGETS:
        truth_train = train[label].astype(int)
        truth_val = val[label].astype(int)
        model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model.fit(features_train, truth_train)
        if list(model.classes_) != [0, 1]:
            raise ValueError(f"Expected binary ordered classes for {label}")
        probs = model.predict_proba(features_val)[:, 1]
        prediction = (probs >= .5).astype(int)
        report = _evaluation(truth_val, prediction, probs, val.video_id)
        result["tasks"][label] = report
        joblib.dump({
            "model": model, "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "classes": [int(c) for c in model.classes_],
            "target": label, "exercise": "squat",
            "training_source": "Fitness-AQA labeled squat train only",
            "decision_threshold": 0.5,
        }, output_dir / f"rf_{label}.joblib")
    (output_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fitness_aqa_rf"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = benchmark(args.features_dir, args.output_dir, seed=args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Training refused: {error}\n")
    brief = {
        "status": report["status"], "train_frames": report["train_frames"],
        "validation_frames": report["validation_frames"],
        "tasks": {name: {
            "frame_macro_f1": metrics["macro_f1_frame"],
            "video_macro_f1": metrics["mean_video_macro_f1"],
            "positive_frames": metrics["positives"],
            "precision_error": metrics["precision_error"],
            "recall_error": metrics["recall_error"],
        } for name, metrics in report["tasks"].items()},
        "full_report": str(args.output_dir / "validation_report.json"),
    }
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
