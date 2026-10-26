#!/usr/bin/env python3
"""Validation-only probability/threshold diagnostics for two Fitness-AQA RF models.

This never trains a classifier, never reads test CSV and never overwrites models.
Threshold candidates chosen on validation are EXPERIMENTAL; do not report their
validation F1 as an unbiased generalization estimate or silently deploy them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score)

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames

THRESHOLDS = tuple(round(float(x), 2) for x in np.linspace(.05, .95, 19))
QUANTILES = (0.1, 0.5, 0.9)


def _metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    predicted = (probs >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(
        labels, predicted, labels=[0, 1]).ravel())
    return {
        "threshold": float(threshold), "n": int(len(labels)),
        "predicted_positive": int(predicted.sum()),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision_error": float(precision_score(labels, predicted, zero_division=0)),
        "recall_error": float(recall_score(labels, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def _quantiles(arr: np.ndarray) -> dict:
    if not len(arr):
        return {"n": 0, "p10": None, "median": None, "p90": None,
                "max": None, "fraction_at_least_0_5": None}
    a, b, c = np.quantile(arr, QUANTILES)
    return {
        "n": int(len(arr)), "p10": float(a), "median": float(b),
        "p90": float(c), "max": float(arr.max()),
        "fraction_at_least_0_5": float((arr >= .5).mean()),
    }


def _clip_any_metrics(labels: np.ndarray, probs: np.ndarray,
                      video_ids: np.ndarray, threshold: float) -> dict:
    """Clip OR on accepted frames ONLY; not true full-video interval detection."""
    unique = np.unique(video_ids)
    clip_labels = np.asarray([labels[video_ids == key].max() for key in unique])
    clip_probs = np.asarray([probs[video_ids == key].max() for key in unique])
    result = _metrics(clip_labels, clip_probs, threshold)
    result["videos"] = len(unique)
    result["true_positive_videos_among_accepted_frames"] = int(clip_labels.sum())
    result["warning"] = "Clip labels refer only to retained pose frames, not all original video annotations"
    return result


def diagnostic_report(y_true: np.ndarray, probabilities: np.ndarray,
                      video_ids: np.ndarray, *, thresholds=THRESHOLDS) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    vids = np.asarray(video_ids, dtype=str)
    if not len(y) or y.shape != p.shape or y.shape != vids.shape:
        raise ValueError("Nonempty labels/probabilities/video IDs must have the same shape")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Binary targets and probabilities within [0,1] are required")
    if len(set(y)) != 2:
        raise ValueError("Both annotation classes required in validation")
    grid = tuple(sorted(set(float(t) for t in (*thresholds, .5))))
    if not all(0 < t < 1 for t in grid):
        raise ValueError("Threshold candidates must be inside (0,1)")
    rows = [_metrics(y, p, t) for t in grid]
    fixed = next(r for r in rows if r["threshold"] == .5)
    # Ties prefer a threshold closer to 0.5 for deterministic reporting.
    candidate = sorted(rows, key=lambda r: (-r["macro_f1"], abs(r["threshold"] - .5), r["threshold"]))[0]
    return {
        "positive_frames": int(y.sum()), "negative_frames": int((y == 0).sum()),
        "positive_prevalence": float(y.mean()),
        "average_precision_probability_ranking": float(average_precision_score(y, p)),
        "random_ranker_average_precision_reference": float(y.mean()),
        "probability_quantiles_by_truth": {
            "positive_error_frames": _quantiles(p[y == 1]),
            "negative_error_frames": _quantiles(p[y == 0]),
        },
        "fixed_threshold_0_5": fixed,
        "candidate_threshold_val_only_NOT_unbiased": candidate,
        "fixed_threshold_video_any_accepted_frame": _clip_any_metrics(y, p, vids, .5),
        "candidate_threshold_video_any_accepted_frame": _clip_any_metrics(
            y, p, vids, candidate["threshold"]),
        "threshold_grid": rows,
    }


def diagnose(feature_dir: str | Path, model_dir: str | Path,
             output_dir: str | Path) -> dict:
    feature_dir, model_dir, output_dir = map(Path, (feature_dir, model_dir, output_dir))
    # Training and test CSVs are intentionally never loaded.
    validation = load_frames(feature_dir / "fitness_aqa_val_pose.csv", "val")
    features = validation.loc[:, list(FEATURE_NAMES)].astype(float)
    result = {
        "status": "validation_only_threshold_diagnostics_NOT_model_retraining",
        "schema_version": SCHEMA_VERSION,
        "source": "Fitness-AQA publisher video-ID validation split",
        "n_validation_frames": len(validation),
        "n_validation_videos_with_accepted_frames": int(validation.video_id.nunique()),
        "tasks": {},
        "warnings": [
            "Selecting a threshold on validation makes resulting validation F1 optimistic for threshold selection",
            "Inspect average precision versus positive prevalence before changing threshold",
            "Adjacent frames in each video are correlated; no independent-subject claim",
            "Video-any metrics cover retained pose frames only, not unseen error windows",
            "Loading joblib requires trusted, locally generated model artifacts",
            "Test split remains locked until validation and choice of operating point are fixed",
        ],
    }
    grid_rows = []
    for target in TARGETS:
        # WARNING: joblib uses pickle; load only models produced locally by this repo.
        bundle = joblib.load(model_dir / f"rf_{target}.joblib")
        if (not isinstance(bundle, dict)
            or bundle.get("schema_version") != SCHEMA_VERSION
            or bundle.get("feature_names") != list(FEATURE_NAMES)
            or bundle.get("target") != target
            or bundle.get("classes") != [0, 1]
            or list(getattr(bundle.get("model"), "classes_", [])) != [0, 1]):
            raise ValueError(f"Wrong schema/label/model for task {target}; retrain trusted artifact")
        probabilities = np.asarray(bundle["model"].predict_proba(features)[:, 1], dtype=float)
        report = diagnostic_report(
            validation[target].to_numpy(dtype=int),
            probabilities,
            validation.video_id.to_numpy(dtype=str),
        )
        result["tasks"][target] = report
        for row in report["threshold_grid"]:
            grid_rows.append({"task": target, **row})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "threshold_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "threshold_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/fitness_aqa_rf"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fitness_aqa_diagnostics"))
    opts = parser.parse_args()
    try:
        report = diagnose(opts.features_dir, opts.model_dir, opts.output_dir)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.exit(1, f"Diagnosis failed: {exc}\n")
    short = {
        "status": report["status"],
        "validation_frames": report["n_validation_frames"],
        "tasks": {
            k: {
                "positive_prevalence": round(v["positive_prevalence"], 4),
                "average_precision": round(v["average_precision_probability_ranking"], 4),
                "positive_probability_quantiles": v["probability_quantiles_by_truth"]["positive_error_frames"],
                "negative_probability_quantiles": v["probability_quantiles_by_truth"]["negative_error_frames"],
                "fixed_0_5": v["fixed_threshold_0_5"],
                "candidate_val_only_not_unbiased": v["candidate_threshold_val_only_NOT_unbiased"],
                "clip_any_0_5": v["fixed_threshold_video_any_accepted_frame"],
            } for k, v in report["tasks"].items()
        },
        "saved_report": str(opts.output_dir / "threshold_diagnostics.json"),
        "saved_grid": str(opts.output_dir / "threshold_grid.csv"),
    }
    print(json.dumps(short, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
