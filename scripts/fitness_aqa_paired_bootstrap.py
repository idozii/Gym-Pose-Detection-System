#!/usr/bin/env python3
"""Paired, video-cluster bootstrap of existing Fitness-AQA static/temporal RF models.

Does not refit any model, tune thresholds, use the source test split, or alter
existing artifacts. This estimates **sampling variability across observed
publisher validation clips**, conditional on the trained models and thresholds.
It is not an independent-subject test or unbiased model-selection guarantee.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames
from scripts.compare_fitness_aqa_temporal import TEMPORAL_SCHEMA, features
from scripts.diagnose_fitness_aqa_thresholds import _metrics

METRICS = ("average_precision", "macro_f1", "recall_error", "precision_error", "false_positive_rate")
VARIANTS = ("static", "causal_temporal")


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    data = _metrics(y, p, threshold)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "macro_f1": data["macro_f1"],
        "recall_error": data["recall_error"],
        "precision_error": data["precision_error"],
        "false_positive_rate": data["false_positive_rate"],
        "true_positive": data["true_positive"],
        "false_positive": data["false_positive"],
        "false_negative": data["false_negative"],
        "true_negative": data["true_negative"],
    }


def paired_cluster_bootstrap(y, static_prob, temporal_prob, video_ids,
                             *, static_threshold: float, temporal_threshold: float,
                             repeats: int = 1000, seed: int = 42) -> dict:
    """Resample *whole videos* with replacement and compare the same frame rows.

    Both variants are scored on identical resampled video clusters. Means and CIs
    are empirical distributions of paired deltas, not iid-frame significance tests.
    """
    y = np.asarray(y, dtype=int)
    p0, p1 = np.asarray(static_prob, dtype=float), np.asarray(temporal_prob, dtype=float)
    videos = np.asarray(video_ids, dtype=str)
    if y.ndim != 1 or len(y) < 2 or any(a.shape != y.shape for a in (p0, p1, videos)):
        raise ValueError("Arrays must be aligned, one-dimensional and nonempty")
    if set(y) != {0, 1} or len(np.unique(videos)) < 2:
        raise ValueError("Need both labels and at least two distinct video clusters")
    if repeats < 10 or not 0 < static_threshold < 1 or not 0 < temporal_threshold < 1:
        raise ValueError("Invalid repetitions or frozen train thresholds")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("Nonfinite probabilities")
    if ((p0 < 0) | (p0 > 1)).any() or ((p1 < 0) | (p1 > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    unique = np.unique(videos)
    indices_by_video = [np.flatnonzero(videos == key) for key in unique]
    before = score(y, p0, static_threshold)
    after = score(y, p1, temporal_threshold)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in METRICS}
    skipped = 0
    for _ in range(repeats):
        selected_groups = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([indices_by_video[i] for i in selected_groups])
        labels = y[indices]
        if set(labels) != {0, 1}:
            skipped += 1
            continue
        m0 = score(labels, p0[indices], static_threshold)
        m1 = score(labels, p1[indices], temporal_threshold)
        for key in METRICS:
            samples[key].append(float(m1[key] - m0[key]))
    if repeats - skipped < max(10, repeats // 2):
        raise ValueError("Insufficient valid video-bootstrap resamples")
    comparisons = {}
    for key in METRICS:
        values = np.asarray(samples[key], dtype=float)
        lo, hi = np.quantile(values, [.025, .975])
        comparisons[key] = {
            "delta_temporal_minus_static": float(after[key] - before[key]),
            "cluster_bootstrap_95pct_interval": [float(lo), float(hi)],
            "fraction_bootstraps_temporal_greater": float((values > 0).mean()),
        }
    return {
        "static": before, "causal_temporal": after,
        "videos": len(unique), "frames": len(y),
        "positive_frames": int(y.sum()),
        "repeats_requested": repeats, "valid_bootstrap_samples": repeats - skipped,
        "bootstrap_samples_without_both_classes": skipped,
        "bootstrap_seed": seed, "paired_differences": comparisons,
    }


def _load_predict(model_dir: Path, variant: str, target: str, X):
    """Verify metadata against the feature construction before unpickling usage."""
    # IMPORTANT: joblib/pickle may execute code; only load models trained locally.
    bundle = joblib.load(model_dir / f"{variant}_{target}.joblib")
    expected_version = SCHEMA_VERSION if variant == "static" else TEMPORAL_SCHEMA
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if (not isinstance(bundle, dict) or bundle.get("target") != target
        or bundle.get("variant") != variant
        or bundle.get("schema_version") != expected_version
        or bundle.get("feature_names") != list(X.columns)
        or bundle.get("classes") != [0, 1]
        or list(getattr(model, "classes_", [])) != [0, 1]
        or bundle.get("threshold_selected_from_train_only") is None):
        raise ValueError(f"{variant}/{target}: model metadata mismatch; rerun comparison")
    threshold = float(bundle["threshold_selected_from_train_only"])
    prob = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return threshold, prob


def compare(feature_dir: str | Path, model_dir: str | Path, output_path: str | Path,
            *, seed: int = 42, repeats: int = 1000) -> dict:
    feature_dir, model_dir, output_path = Path(feature_dir), Path(model_dir), Path(output_path)
    previous = json.loads((model_dir / "comparison.json").read_text(encoding="utf-8"))
    if previous.get("status") != "train_group_calibrated_validation_evaluation_ONLY":
        raise ValueError("Expected prior train-calibrated V2 comparison, not a tuned test model")
    val = load_frames(feature_dir / "fitness_aqa_val_pose.csv", "val")
    if previous.get("validation_frames") != len(val):
        raise ValueError("Validation CSV count differs from original comparison")
    if previous.get("validation_videos") != int(val.video_id.nunique()):
        raise ValueError("Validation clip count differs from original comparison")
    gap = float(previous["max_gap_seconds"])
    inputs = {
        variant: features(val, temporal=variant != "static", max_gap_seconds=gap)
        for variant in VARIANTS
    }
    report = {
        "status": "validation_only_cluster_bootstrap_conditional_on_models",
        "source": "publisher video-ID validation, not verified independent subjects",
        "validation_frames": len(val), "validation_videos": int(val.video_id.nunique()),
        "repeats": repeats, "seed": seed,
        "comparison_seed": previous["seed"],
        "tasks": {},
        "limitations": [
            "CIs describe video-cluster sampling variability in *this* validation set only",
            "CIs do not include model-training variation, threshold-calibration variation or subject correlation",
            "Variants were already compared on validation, so interpreting a CI as independent confirmation is invalid",
            "Train-calibrated thresholds frozen, not retuned within bootstrap",
            "Video-level bootstrap preserves correlations among frames from the same video",
            "Original publisher test set is never read or used here",
            "No clinically validated squat form assessment or injury prevention claims",
        ],
    }
    for target in TARGETS:
        thresholds, probabilities = {}, {}
        for variant in VARIANTS:
            threshold, probabilities[variant] = _load_predict(
                model_dir, variant, target, inputs[variant])
            expected = previous["variants"][variant]["tasks"][target]["train_calibrated_threshold"]
            if not np.isclose(threshold, float(expected), atol=1e-10):
                raise ValueError(f"{variant}/{target}: threshold differs from prior report")
            thresholds[variant] = threshold
        report["tasks"][target] = {
            "thresholds_from_train": thresholds,
            **paired_cluster_bootstrap(
                val[target].to_numpy(dtype=int),
                probabilities["static"], probabilities["causal_temporal"],
                val.video_id.to_numpy(dtype=str),
                static_threshold=thresholds["static"],
                temporal_threshold=thresholds["causal_temporal"],
                repeats=repeats, seed=seed,
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path,
                        default=Path("data/features/fitness_aqa"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/fitness_aqa_temporal"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/fitness_aqa_robustness/paired_bootstrap.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    try:
        report = compare(args.features_dir, args.model_dir, args.output,
                         seed=args.seed, repeats=args.repeats)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Bootstrap diagnostic failed: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "validation_videos": report["validation_videos"],
        "tasks": {
            target: {
                key: val for key, val in details["paired_differences"].items()
                if key in ("average_precision", "macro_f1", "recall_error", "false_positive_rate")
            } for target, details in report["tasks"].items()
        },
        "full_report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
