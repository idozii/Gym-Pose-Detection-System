#!/usr/bin/env python3
"""Train-only calibrated, clip-paired quality-gate sensitivity experiment.

Compares two independent extractions of the SAME Fitness-AQA publisher train/val
video split at pose visibility 0.50 vs 0.35. Never reads source test data.
Differences in the FULL retained validation sets combine model changes with
sample-selection changes. Compare models on the COMMON accepted validation
sample IDs to partly separate these effects.

MediaPipe is re-run separately for each extraction; shared sample IDs, labels
and feature-value deviations are audited. This is not a causal randomized trial.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from scripts.audit_fitness_aqa_pose import TARGETS, load_frames
from scripts.compare_fitness_aqa_temporal import features, choose_threshold
from scripts.diagnose_fitness_aqa_thresholds import _metrics
from gym_pose.features import FEATURE_NAMES

GATES = ("gate050", "gate035")


def _source_summary(folder: Path, split: str) -> dict:
    path = folder / "fitness_aqa_extraction_summary.json"
    if not path.is_file():
        raise ValueError(f"Required extraction summary missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split_type") != "publisher_video_id_split_NOT_subject_disjoint":
        raise ValueError(f"Invalid publisher clip split in {path}")
    if split not in data.get("splits", {}):
        raise ValueError(f"{path} has no {split} extraction summary")
    return data


def load_pair(original_dir: str | Path, relaxed_dir: str | Path) -> tuple[dict, dict]:
    paths = {"gate050": Path(original_dir), "gate035": Path(relaxed_dir)}
    summaries = {}
    frames = {}
    for gate, folder in paths.items():
        train_summary = _source_summary(folder, "train")
        for name in ("train", "val"):
            current = _source_summary(folder, name)
            if (current["sampling_fps"] != train_summary["sampling_fps"] or
                current["boundary_margin_seconds"] != train_summary["boundary_margin_seconds"]):
                raise ValueError("Sampling rate or boundary margin inconsistent")
        cutoff = 0.5 if gate == "gate050" else 0.35
        if not np.isclose(float(train_summary["min_pose_visibility"]), cutoff):
            raise ValueError(f"{gate}: expected cutoff {cutoff}")
        summaries[gate] = train_summary
        frames[gate] = {}
        for split in ("train", "val"):
            df = load_frames(folder / f"fitness_aqa_{split}_pose.csv", split)
            stats = train_summary["splits"][split]
            if len(df) != int(stats["pose_accepted"]):
                raise ValueError(f"{gate}/{split}: stale CSV or extraction summary")
            frames[gate][split] = df
    for gate in GATES:
        if set(frames[gate]["train"].video_id) & set(frames[gate]["val"].video_id):
            raise ValueError(f"{gate}: source train/validation video ID overlap")
    first, second = (summaries[g] for g in GATES)
    if first["sampling_fps"] != second["sampling_fps"] or first["boundary_margin_seconds"] != second["boundary_margin_seconds"]:
        raise ValueError("Extraction configurations must have same FPS and boundary margin")
    for split in ("train", "val"):
        aa, bb = (summaries[g]["splits"][split] for g in GATES)
        if (aa["selected_videos"] != bb["selected_videos"]
            or aa["frames_sampled"] != bb["frames_sampled"]
            or aa["boundary_rejected"] != bb["boundary_rejected"]):
            raise ValueError(f"{split}: sampled frame universe differs between gate runs")
    return frames, summaries


def _align(first: pd.DataFrame, second: pd.DataFrame, *, split: str) -> dict:
    a = first.set_index("sample_id", verify_integrity=True)
    b = second.set_index("sample_id", verify_integrity=True)
    keys = sorted(set(a.index) & set(b.index))
    if not keys:
        raise ValueError(f"{split}: no shared accepted frames")
    shared_a, shared_b = a.loc[keys], b.loc[keys]
    for key in ("video_id", "timestamp_s", "frame_index", *TARGETS):
        if not np.array_equal(shared_a[key].to_numpy(), shared_b[key].to_numpy()):
            raise ValueError(f"{split}: {key} differs on shared frames")
    deltas = np.abs(
        shared_a.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
        - shared_b.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    )
    return {
        "common_sample_ids": keys,
        "common_frame_count": len(keys),
        "original_only_count": int(len(a) - len(keys)),
        "relaxed_only_count": int(len(b) - len(keys)),
        "shared_feature_max_abs_delta": float(deltas.max()),
        "shared_feature_mean_abs_delta": float(deltas.mean()),
    }


def _shared_group_split(train_frames: dict, *, seed: int, max_trials: int = 32):
    """Use exactly the SAME fit/cal source clip IDs for both gate variants."""
    groups = np.array(sorted(set().union(*(set(train_frames[g].video_id) for g in GATES))))
    if len(groups) < 8:
        raise ValueError("At least eight distinct train videos required for calibration")
    splitter = GroupShuffleSplit(n_splits=max_trials, test_size=.25, random_state=seed)
    for fit_idx, cal_idx in splitter.split(groups, groups=groups):
        fit_ids, cal_ids = set(groups[fit_idx]), set(groups[cal_idx])
        good = True
        for gate in GATES:
            df = train_frames[gate]
            for ids in (fit_ids, cal_ids):
                labels = df.loc[df.video_id.isin(ids), list(TARGETS)]
                if len(labels) < 10 or any(set(labels[t]) != {0, 1} for t in TARGETS):
                    good = False
                    break
            if not good:
                break
        if good:
            return fit_ids, cal_ids
    raise ValueError("Could not find a shared video-disjoint calibration split containing both error classes in both gates")


def _model(seed, trees):
    return RandomForestClassifier(
        n_estimators=trees, random_state=seed, n_jobs=-1,
        min_samples_leaf=2, class_weight="balanced",
    )


def _score(y, prob, threshold) -> dict:
    if set(y) != {0, 1}:
        return {"status": "insufficient_both_classes", "n": len(y)}
    return {
        **_metrics(y, prob, threshold),
        "average_precision": float(average_precision_score(y, prob)),
        "positive_prevalence": float(np.mean(y)),
    }


def experiment(original_dir: str | Path, relaxed_dir: str | Path,
               output: str | Path, *, seed: int = 42,
               trees: int = 300, max_gap_seconds: float = .75) -> dict:
    if trees < 1:
        raise ValueError("trees must be positive")
    frames, summaries = load_pair(original_dir, relaxed_dir)
    alignments = {
        split: _align(frames["gate050"][split], frames["gate035"][split], split=split)
        for split in ("train", "val")
    }
    fit_ids, cal_ids = _shared_group_split({
        g: frames[g]["train"] for g in GATES
    }, seed=seed)
    result = {
        "status": "visibility_gate_ablation_train_calibrated_NO_test_read",
        "gates": {"gate050": 0.5, "gate035": 0.35},
        "seed": seed, "trees": trees,
        "feature_variant": "causal_temporal_119", "max_gap_seconds": max_gap_seconds,
        "matched_frame_audit": {
            split: {k: v for k, v in a.items() if k != "common_sample_ids"}
            for split, a in alignments.items()
        },
        "shared_train_calibration": {
            "fit_videos": len(fit_ids), "calibration_videos": len(cal_ids),
            "clip_ID_hash_sha256": hashlib.sha256(
                ("|".join(sorted(fit_ids)) + "--" + "|".join(sorted(cal_ids))).encode()
            ).hexdigest(),
            "groups_are_disjoint": fit_ids.isdisjoint(cal_ids),
        },
        "gate_coverage": {},
        "tasks": {},
        "limitations": [
            "Visibility relaxation may produce noisy landmarks; higher coverage is not proof of improvement",
            "Two separate MediaPipe extraction passes may differ on common frames; feature-delta audit provided",
            "Model comparison on full accepted samples is confounded by changing dataset selection",
            "Common-frame comparison isolates matched observations, but training samples and temporal history also differ",
            "Calibration uses identical disjoint source train video groups; source train and val never mixed",
            "This source video split is NOT established subject-independent",
            "Validation previously used to compare architecture; this experiment is exploratory, not independent confirmation",
            "Frames with no pose remain abstentions, not true negatives",
            "No original publisher test data are used",
        ],
    }
    for gate in GATES:
        result["gate_coverage"][gate] = {}
        for split in ("train", "val"):
            s = summaries[gate]["splits"][split]
            eligible = int(s["frames_sampled"]) - int(s["boundary_rejected"])
            accepted = int(s["pose_accepted"])
            result["gate_coverage"][gate][split] = {
                "eligible_frames": eligible, "accepted_frames": accepted,
                "accepted_rate": accepted / eligible,
                "videos_with_accepted_frames": int(frames[gate][split].video_id.nunique()),
                "positive_accepted_frames": {t: int(frames[gate][split][t].sum()) for t in TARGETS},
            }
    val_inputs = {
        gate: features(frames[gate]["val"], temporal=True, max_gap_seconds=max_gap_seconds)
        for gate in GATES
    }
    train_inputs = {
        gate: features(frames[gate]["train"], temporal=True, max_gap_seconds=max_gap_seconds)
        for gate in GATES
    }
    val_common = alignments["val"]["common_sample_ids"]
    for target in TARGETS:
        details = {}
        for gate in GATES:
            train = frames[gate]["train"]
            y = train[target].to_numpy(dtype=int)
            fit_mask = train.video_id.isin(fit_ids).to_numpy()
            cal_mask = train.video_id.isin(cal_ids).to_numpy()
            if not fit_mask.any() or not cal_mask.any() or set(y[fit_mask]) != {0,1} or set(y[cal_mask]) != {0,1}:
                raise ValueError(f"{gate}/{target}: invalid shared fit/cal label coverage")
            temp = _model(seed, trees)
            temp.fit(train_inputs[gate].loc[fit_mask], y[fit_mask])
            threshold = float(choose_threshold(
                y[cal_mask], temp.predict_proba(train_inputs[gate].loc[cal_mask])[:, 1]
            )["threshold"])
            final = _model(seed, trees)
            final.fit(train_inputs[gate], y)
            val = frames[gate]["val"]
            prob = final.predict_proba(val_inputs[gate])[:, 1]
            y_val = val[target].to_numpy(dtype=int)
            common_series = pd.Series(prob, index=val.sample_id)
            labels_series = pd.Series(y_val, index=val.sample_id)
            details[gate] = {
                "threshold_from_source_train_calibration": threshold,
                "all_accepted_validation_frames": _score(y_val, prob, threshold),
                "common_accepted_validation_frames": _score(
                    labels_series.loc[val_common].to_numpy(dtype=int),
                    common_series.loc[val_common].to_numpy(dtype=float), threshold),
            }
        result["tasks"][target] = details
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--relaxed-dir", type=Path, default=Path("data/features/fitness_aqa_gate035"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/fitness_aqa_gate_ablation/comparison.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    try:
        report = experiment(args.baseline_dir, args.relaxed_dir, args.output,
                            seed=args.seed, trees=args.trees)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Gate ablation failed: {exc}\n")
    short = {
        "status": report["status"],
        "coverage": report["gate_coverage"],
        "matched_frames": report["matched_frame_audit"],
        "tasks": {
            t: {g: {
                "train_threshold": details["threshold_from_source_train_calibration"],
                "validation_all": {
                    key: details["all_accepted_validation_frames"].get(key)
                    for key in ("macro_f1","average_precision","recall_error",
                                "precision_error","false_positive_rate")},
                "validation_common": {
                    key: details["common_accepted_validation_frames"].get(key)
                    for key in ("macro_f1","average_precision","recall_error",
                                "precision_error","false_positive_rate")},
            } for g, details in variants.items()}
            for t, variants in report["tasks"].items()},
        "full_report": str(args.output),
    }
    print(json.dumps(short, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
