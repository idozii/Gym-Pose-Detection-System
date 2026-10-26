#!/usr/bin/env python3
"""Train-only video-group calibration stability for Fitness-AQA static vs causal RF.

Hold final RandomForest parameters and random seed FIXED, and vary only the
video-group calibration split. Validation is used for reporting only (repeated
evaluation on one validation split is not independent confirmation). Test untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.audit_fitness_aqa_pose import TARGETS, load_frames
from scripts.compare_fitness_aqa_temporal import (
    features, grouped_calibration_split, choose_threshold, TEMPORAL_SCHEMA,
)
from scripts.diagnose_fitness_aqa_thresholds import _metrics

VARIANTS = ("static", "causal_temporal")
METRICS = ("macro_f1", "precision_error", "recall_error", "false_positive_rate")


def stable_summary(rows: list[dict]) -> dict:
    """Descriptive ranges across seeds, NOT confidence intervals or independent tests."""
    if not rows:
        raise ValueError("Cannot summarize empty runs")
    thresholds = np.asarray([r["threshold_from_train_calibration"] for r in rows], float)
    result = {
        "seeds": [int(r["calibration_seed"]) for r in rows],
        "thresholds": [float(v) for v in thresholds],
        "threshold_median": float(np.median(thresholds)),
        "threshold_range": [float(thresholds.min()), float(thresholds.max())],
        "n_runs": len(rows),
        "metrics": {},
    }
    for metric in METRICS:
        data = np.asarray([r["validation"][metric] for r in rows], dtype=float)
        result["metrics"][metric] = {
            "mean": float(data.mean()),
            "min": float(data.min()), "max": float(data.max()),
            "median": float(np.median(data)),
        }
    return result


def compare(feature_dir: str | Path, output_dir: str | Path, *,
            calibration_seeds: tuple[int, ...] = (42, 52, 62, 72, 82),
            forest_seed: int = 42, trees: int = 300,
            max_gap_seconds: float = .75) -> dict:
    if len(calibration_seeds) < 2 or len(set(calibration_seeds)) != len(calibration_seeds):
        raise ValueError("Supply >=2 unique train-group calibration seeds")
    if not all(isinstance(s, int) and s >= 0 for s in calibration_seeds):
        raise ValueError("Calibration seeds must be nonnegative integers")
    if trees < 1:
        raise ValueError("trees must be positive")
    base = Path(feature_dir)
    train = load_frames(base / "fitness_aqa_train_pose.csv", "train")
    validation = load_frames(base / "fitness_aqa_val_pose.csv", "val")
    if set(train.video_id) & set(validation.video_id):
        raise ValueError("Leakage: original train and validation share video IDs")
    if len(train) < 50 or len(validation) < 20:
        raise ValueError("Insufficient dataset for comparison")
    splits = {
        seed: grouped_calibration_split(train, seed=seed)
        for seed in calibration_seeds
    }
    output = {
        "status": "train_calibration_seed_stability_validation_descriptive_only",
        "source": "Fitness-AQA publisher train and val clip IDs; not verified subject-disjoint",
        "schema_version": SCHEMA_VERSION, "temporal_schema": TEMPORAL_SCHEMA,
        "forest_seed_fixed": forest_seed, "forest_trees": trees,
        "calibration_seeds": list(calibration_seeds),
        "train_frames": int(len(train)), "validation_frames": int(len(validation)),
        "calibration_video_counts": {
            str(s): int(train.iloc[indices[1]].video_id.nunique()) for s, indices in splits.items()
        },
        "variants": {}, "paired_seed_differences": {},
        "warnings": [
            "All seeds reuse the same original validation videos; ranges are descriptive, not CIs",
            "Validation already used for static/temporal comparison; no new independent confirmation",
            "Only group calibration split varies; final model random seed is deliberately fixed",
            "Train/val disjoint by video clip, not proven disjoint by athlete identity",
            "Frame-level pose acceptance near 50%; evaluate rejection bias separately",
            "Only annotation-specific error targets, not overall squat correctness",
            "Publisher test files and clips are never loaded or used",
        ],
    }
    for variant in VARIANTS:
        temporal = variant == "causal_temporal"
        train_x = features(train, temporal=temporal, max_gap_seconds=max_gap_seconds)
        val_x = features(validation, temporal=temporal, max_gap_seconds=max_gap_seconds)
        output["variants"][variant] = {"feature_count": train_x.shape[1], "tasks": {}}
        for target in TARGETS:
            y_train = train[target].to_numpy(dtype=int)
            y_val = validation[target].to_numpy(dtype=int)
            if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
                raise ValueError(f"{target}: both classes required in train and val")
            # Fit once on full source train. These exact validation probabilities
            # do NOT depend on calibration seed, only decision thresholds do.
            final = RandomForestClassifier(
                n_estimators=trees, min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=forest_seed,
            )
            final.fit(train_x, y_train)
            val_p = final.predict_proba(val_x)[:, 1]
            runs = []
            for cal_seed in calibration_seeds:
                fit, cal = splits[cal_seed]
                if set(train.iloc[fit].video_id) & set(train.iloc[cal].video_id):
                    raise AssertionError("Calibration clip leakage")
                if any(set(y_train[idx]) != {0, 1} for idx in (fit, cal)):
                    raise ValueError(f"Missing class in grouped fit/cal at seed {cal_seed}")
                temp = RandomForestClassifier(
                    n_estimators=trees, min_samples_leaf=2, class_weight="balanced",
                    n_jobs=-1, random_state=forest_seed,
                )
                temp.fit(train_x.iloc[fit], y_train[fit])
                cal_probs = temp.predict_proba(train_x.iloc[cal])[:, 1]
                operating_point = choose_threshold(y_train[cal], cal_probs)
                t = float(operating_point["threshold"])
                runs.append({
                    "calibration_seed": cal_seed,
                    "threshold_from_train_calibration": t,
                    "calibration": operating_point,
                    "validation": _metrics(y_val, val_p, t),
                })
            output["variants"][variant]["tasks"][target] = {
                "validation_ap_invariant_across_calibration_seeds": float(
                    average_precision_score(y_val, val_p)),
                "validation_positive_prevalence": float(y_val.mean()),
                "runs": runs,
                "seed_stability": stable_summary(runs),
            }
    for target in TARGETS:
        original = output["variants"]["static"]["tasks"][target]["runs"]
        temporal = output["variants"]["causal_temporal"]["tasks"][target]["runs"]
        paired = []
        for left, right in zip(original, temporal):
            if left["calibration_seed"] != right["calibration_seed"]:
                raise AssertionError("Seed pairing mismatch")
            paired.append({
                "calibration_seed": left["calibration_seed"],
                **{f"delta_{key}": float(right["validation"][key] - left["validation"][key])
                   for key in METRICS},
                "static_threshold": left["threshold_from_train_calibration"],
                "temporal_threshold": right["threshold_from_train_calibration"],
            })
        output["paired_seed_differences"][target] = {
            "runs": paired,
            "temporal_macro_f1_greater_count": sum(r["delta_macro_f1"] > 0 for r in paired),
            "temporal_recall_greater_count": sum(r["delta_recall_error"] > 0 for r in paired),
            "temporal_fpr_lower_count": sum(r["delta_false_positive_rate"] < 0 for r in paired),
        }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "multiseed_calibration.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fitness_aqa_robustness"))
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--forest-seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    try:
        report = compare(
            args.features_dir, args.output_dir,
            calibration_seeds=tuple(args.calibration_seeds),
            forest_seed=args.forest_seed, trees=args.trees,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Calibration stability failed: {exc}\n")
    print(json.dumps({
        "status": report["status"], "forest_seed_fixed": report["forest_seed_fixed"],
        "calibration_seeds": report["calibration_seeds"],
        "tasks": {
            target: {
                "static": report["variants"]["static"]["tasks"][target]["seed_stability"],
                "temporal": report["variants"]["causal_temporal"]["tasks"][target]["seed_stability"],
                "paired_counts": {
                    k: v for k, v in report["paired_seed_differences"][target].items() if k != "runs"
                },
                "ap_static": report["variants"]["static"]["tasks"][target][
                    "validation_ap_invariant_across_calibration_seeds"],
                "ap_temporal": report["variants"]["causal_temporal"]["tasks"][target][
                    "validation_ap_invariant_across_calibration_seeds"],
            } for target in TARGETS
        },
        "full_report": str(args.output_dir / "multiseed_calibration.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
