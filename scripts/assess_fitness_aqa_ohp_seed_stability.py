#!/usr/bin/env python3
"""Train-only-calibrated multi-seed OHP model/threshold stability study.

Stacks on M13D, does not read publisher TEST or source media, and never
modifies the original trained model artifacts. Validation has already been
used repeatedly during development, so outcomes are *exploratory*, not a
fresh holdout or basis for independently choosing a winning variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from gym_pose.ohp_task_features import TASKS
from scripts.benchmark_fitness_aqa_ohp_tasks import (
    VARIANTS, _load_data, group_calibration, model, predict_positive,
    task_features,
)
from scripts.compare_fitness_aqa_temporal import choose_threshold
from scripts.diagnose_fitness_aqa_thresholds import _metrics


DEFAULT_SEEDS = (41, 42, 43, 44, 45)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(p.strip()) for p in raw.split(","))
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated nonnegative integers") from exc
    if not 2 <= len(parts) <= 20 or len(set(parts)) != len(parts):
        raise ValueError("use 2–20 unique seeds to assess stability")
    if any(s < 0 or s >= (2**32 - 1) for s in parts):
        raise ValueError("each seed must be in range [0, 2**32 - 2]")
    return parts


def _distribution(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Cannot summarize empty/nonfinite score distribution")
    return {
        "min": float(a.min()), "median": float(np.median(a)),
        "max": float(a.max()), "mean": float(a.mean()),
        "sample_sd": float(a.std(ddof=1)) if len(a) >= 2 else None,
    }


def run(features_dir: str | Path, output: str | Path, *,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        trees: int = 150, max_gap_seconds: float = .75,
        require_full: bool = True) -> dict:
    if not 2 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds):
        raise ValueError("Require 2–20 distinct model/calibration seeds")
    if any(not isinstance(s, int) or s < 0 or s >= 2**32-1 for s in seeds):
        raise ValueError("Invalid seed value")
    if trees < 1 or not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Invalid trees or temporal gap")
    frames, source = _load_data(Path(features_dir), require_full=require_full)
    report = {
        "status": "ohp_multiseed_source_train_calibrated_VALIDATION_exploratory_NO_TEST",
        "seeds": list(seeds), "rf_trees": trees,
        "source": "Fitness-AQA publisher OHP train/val",
        "publisher_test_accessed": False,
        "source_train_group_calibration_only": True,
        "max_gap_seconds": max_gap_seconds,
        "tasks": {},
        "limitations": [
            "Seeds and train-group partitions are NOT independent replicated experiments",
            "Source validation has been examined repeatedly in this project: results are exploratory",
            "Changes between seeds mix random-forest stochasticity and train-only threshold variation",
            "Both variants share EXACTLY the same retained validation frames WITHIN each target",
            "Different OHP error targets have different gate-selected frame populations and prevalence",
            "Video IDs are disjoint by publisher split, subject IDs are not verified",
            "No TEST data, no raw videos, no external validation or reliable real-time coaching claim",
            "Model fit on full source train; train-only grouped holdout used only for threshold selection",
            "No deployment/model-selection change is made by this report",
        ],
    }
    for task_i, target in enumerate(TASKS):
        train, val = frames[target]["train"], frames[target]["val"]
        yt = train[target].to_numpy(dtype=int)
        yv = val[target].to_numpy(dtype=int)
        static_xtrain = task_features(train, target, temporal=False)
        static_xval = task_features(val, target, temporal=False)
        temporal_xtrain = task_features(train, target, temporal=True,
                                       max_gap_seconds=max_gap_seconds)
        temporal_xval = task_features(val, target, temporal=True,
                                     max_gap_seconds=max_gap_seconds)
        xtrain = {"static": static_xtrain, "causal_temporal": temporal_xtrain}
        xval = {"static": static_xval, "causal_temporal": temporal_xval}
        task_report = {
            "train_frames": len(train), "validation_frames": len(val),
            "train_positive_frames": int(yt.sum()),
            "train_positive_videos": int(train.loc[train[target].eq(1), "video_id"].nunique()),
            "validation_positive_frames": int(yv.sum()),
            "validation_positive_videos": int(val.loc[val[target].eq(1), "video_id"].nunique()),
            "validation_positive_prevalence": float(yv.mean()),
            "coverage_train": source["splits"]["train"]["tasks"][target]["pose_coverage_eligible"],
            "coverage_validation": source["splits"]["val"]["tasks"][target]["pose_coverage_eligible"],
            "results_by_seed": [],
        }
        for seed in seeds:
            fit_i, cal_i = group_calibration(
                train, task=target, seed=seed + task_i)
            if set(train.iloc[fit_i].video_id) & set(train.iloc[cal_i].video_id):
                raise AssertionError("Train fit/calibration video groups overlap")
            fit_ids = sorted(set(train.iloc[fit_i].video_id))
            cal_ids = sorted(set(train.iloc[cal_i].video_id))
            seed_result = {
                "seed": seed,
                "fit_video_count": len(fit_ids),
                "calibration_video_count": len(cal_ids),
                "fit_positive_video_count": int(train.iloc[fit_i].loc[
                    train.iloc[fit_i][target].eq(1), "video_id"].nunique()),
                "cal_positive_video_count": int(train.iloc[cal_i].loc[
                    train.iloc[cal_i][target].eq(1), "video_id"].nunique()),
                "video_groups_disjoint": True,
                "group_hash_sha256": hashlib.sha256(
                    ("|".join(fit_ids) + "--" + "|".join(cal_ids)).encode()
                ).hexdigest(),
                "variants": {},
            }
            for variant in VARIANTS:
                temp = model(seed, trees)
                temp.fit(xtrain[variant].iloc[fit_i], yt[fit_i])
                cal_prob = predict_positive(temp, xtrain[variant].iloc[cal_i])
                th = float(choose_threshold(yt[cal_i], cal_prob)["threshold"])
                final = model(seed, trees)
                final.fit(xtrain[variant], yt)
                p = predict_positive(final, xval[variant])
                seed_result["variants"][variant] = {
                    "feature_count": len(xtrain[variant].columns),
                    "threshold_from_train_groups_only": th,
                    "validation_average_precision": float(average_precision_score(yv, p)),
                    "validation_0_5": _metrics(yv, p, .5),
                    "validation_train_calibrated": _metrics(yv, p, th),
                }
            a = seed_result["variants"]["static"]
            b = seed_result["variants"]["causal_temporal"]
            seed_result["temporal_minus_static"] = {
                "average_precision": (b["validation_average_precision"] -
                                      a["validation_average_precision"]),
                "macro_f1_calibrated": (
                    b["validation_train_calibrated"]["macro_f1"] -
                    a["validation_train_calibrated"]["macro_f1"]),
                "recall_calibrated": (
                    b["validation_train_calibrated"]["recall_error"] -
                    a["validation_train_calibrated"]["recall_error"]),
                "precision_calibrated": (
                    b["validation_train_calibrated"]["precision_error"] -
                    a["validation_train_calibrated"]["precision_error"]),
            }
            task_report["results_by_seed"].append(seed_result)
        task_report["stability_summary"] = {}
        for variant in VARIANTS:
            runs = [s["variants"][variant] for s in task_report["results_by_seed"]]
            task_report["stability_summary"][variant] = {
                "train_calibrated_threshold": _distribution([
                    s["threshold_from_train_groups_only"] for s in runs]),
                "validation_AP": _distribution([
                    s["validation_average_precision"] for s in runs]),
                "validation_macro_F1_calibrated": _distribution([
                    s["validation_train_calibrated"]["macro_f1"] for s in runs]),
                "validation_recall_calibrated": _distribution([
                    s["validation_train_calibrated"]["recall_error"] for s in runs]),
                "validation_precision_calibrated": _distribution([
                    s["validation_train_calibrated"]["precision_error"] for s in runs]),
                "validation_FPR_calibrated": _distribution([
                    s["validation_train_calibrated"]["false_positive_rate"] for s in runs]),
            }
        deltas = [r["temporal_minus_static"] for r in task_report["results_by_seed"]]
        task_report["stability_summary"]["temporal_minus_static"] = {
            key: {
                "distribution": _distribution([d[key] for d in deltas]),
                "seeds_positive": sum(d[key] > 0 for d in deltas),
                "seeds_equal": sum(d[key] == 0 for d in deltas),
                "seeds_negative": sum(d[key] < 0 for d in deltas),
                "warning": "Seed signs are descriptive, not an independent statistical test",
            } for key in ("average_precision", "macro_f1_calibrated",
                          "recall_calibrated", "precision_calibrated")
        }
        report["tasks"][target] = task_report
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path,
                        default=Path("data/features/fitness_aqa_ohp_tasks"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/fitness_aqa_ohp_task_rf/seed_stability.json"))
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--max-gap-seconds", type=float, default=.75)
    parser.add_argument("--allow-pilot", action="store_true",
                        help="Synthetic tests ONLY; full official extraction is default")
    args = parser.parse_args()
    try:
        report = run(args.features_dir, args.output,
                     seeds=_parse_seeds(args.seeds),
                     trees=args.trees,
                     max_gap_seconds=args.max_gap_seconds,
                     require_full=not args.allow_pilot)
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError) as err:
        parser.exit(1, f"OHP seed stability failed: {err}\n")
    compact = {
        "status": report["status"], "seeds": report["seeds"],
        "publisher_test_accessed": report["publisher_test_accessed"],
        "tasks": {
            task: {
                "train_positive_videos": data["train_positive_videos"],
                "validation_positive_videos": data["validation_positive_videos"],
                "static_threshold_range": (
                    data["stability_summary"]["static"]["train_calibrated_threshold"]),
                "temporal_threshold_range": (
                    data["stability_summary"]["causal_temporal"]["train_calibrated_threshold"]),
                "static_AP": data["stability_summary"]["static"]["validation_AP"],
                "temporal_AP": data["stability_summary"]["causal_temporal"]["validation_AP"],
                "static_macro_F1": data["stability_summary"]["static"]["validation_macro_F1_calibrated"],
                "temporal_macro_F1": data["stability_summary"]["causal_temporal"]["validation_macro_F1_calibrated"],
                "temporal_minus_static": data["stability_summary"]["temporal_minus_static"],
            } for task, data in report["tasks"].items()
        },
        "report": str(args.output),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
