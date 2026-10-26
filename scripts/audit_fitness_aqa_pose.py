#!/usr/bin/env python3
"""Audit Fitness-AQA extracted pose CSVs without accessing videos or test labels for training.

Targets mean error-specific *annotations* at timestamps; a zero is not proof
that the squat is globally correct. This audit does not measure model quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from gym_pose.features import FEATURE_NAMES
from scripts.extract_fitness_aqa_pose import META_COLUMNS

TARGETS = ("error_knees_forward", "error_knees_inward")
SPLITS = ("train", "val", "test")


def load_frames(path: Path, split: str) -> pd.DataFrame:
    """Read, type-check, and verify single publisher-split pose observations."""
    df = pd.read_csv(path)
    required = (*META_COLUMNS, *FEATURE_NAMES)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing[:8]}")
    if df.empty:
        raise ValueError(f"{path.name}: contains no accepted poses")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{path.name}: missing values")
    if not df["split"].eq(split).all():
        raise ValueError(f"{path.name}: split labels do not match {split}")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample IDs")
    if not df["video_id"].map(lambda k: isinstance(k, str) and bool(k)).all():
        raise ValueError(f"{path.name}: missing video IDs")
    for name in (*TARGETS, "any_of_two_annotated_errors"):
        if not df[name].isin([0, 1]).all():
            raise ValueError(f"{path.name}: nonbinary label {name}")
    calculated_or = df[TARGETS].max(axis=1)
    if not calculated_or.eq(df["any_of_two_annotated_errors"]).all():
        raise ValueError(f"{path.name}: any_of_two_annotated_errors mismatch")
    if (df["timestamp_s"] < 0).any() or (df["frame_index"] < 0).any():
        raise ValueError(f"{path.name}: negative timestamp/frame")
    if (df["source_fps"] <= 0).any():
        raise ValueError(f"{path.name}: invalid source_fps")
    if not np.isfinite(df[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: nonfinite features")
    if df.duplicated(["video_id", "frame_index"]).any():
        raise ValueError(f"{path.name}: repeated frame per video")
    return df


def audit(folder: str | Path, *, min_pose_coverage: float = .7,
          min_positive_frames: int = 20, min_positive_videos: int = 5) -> dict:
    folder = Path(folder)
    if not 0 <= min_pose_coverage <= 1 or min_positive_frames < 1 or min_positive_videos < 1:
        raise ValueError("Invalid QA thresholds")
    summary_path = folder / "fitness_aqa_extraction_summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    if source.get("split_type") != "publisher_video_id_split_NOT_subject_disjoint":
        raise ValueError("Expected publisher clip-ID splits, not synthetic subject IDs")
    result = {
        "status": "dataset_qa_only_NOT_model_performance",
        "source_report": str(summary_path),
        "split_type": source["split_type"],
        "splits": {}, "warnings": [],
    }
    all_ids = set()
    all_samples = set()
    for split in SPLITS:
        csv_path = folder / f"fitness_aqa_{split}_pose.csv"
        stats = source.get("splits", {}).get(split)
        if stats is None:
            if csv_path.exists():
                result["warnings"].append(f"stale_{split}_CSV_ignored_not_in_current_summary")
            continue
        if not csv_path.is_file():
            raise ValueError(f"Summary declares {split} but CSV is missing")
        df = load_frames(csv_path, split)
        ids = set(df["video_id"].astype(str))
        samples = set(df["sample_id"].astype(str))
        if all_ids & ids or all_samples & samples:
            raise ValueError(f"Split leakage: repeated video/sample IDs in {split}")
        all_ids.update(ids)
        all_samples.update(samples)
        if int(stats["pose_accepted"]) != len(df):
            raise ValueError(f"{split}: CSV row count does not equal extraction summary")
        sampled = int(stats["frames_sampled"])
        boundaries = int(stats["boundary_rejected"])
        eligible = sampled - boundaries
        accepted = int(stats["pose_accepted"])
        rejected = int(stats["pose_rejected"])
        if eligible < 1 or accepted + rejected != eligible:
            raise ValueError(f"{split}: extraction frame accounting is inconsistent")
        coverage = accepted / eligible
        detail = {
            "videos_with_accepted_frames": len(ids),
            "videos_selected": int(stats["selected_videos"]),
            "frames_sampled": sampled,
            "boundary_rejected": boundaries,
            "pose_rejected": rejected,
            "pose_accepted": accepted,
            "pose_coverage_eligible": round(coverage, 4),
            "targets": {},
            "warnings": [],
        }
        if coverage < min_pose_coverage:
            detail["warnings"].append("low_pose_coverage")
        for label in TARGETS:
            positive = int(df[label].sum())
            positive_videos = int(df.loc[df[label] == 1, "video_id"].nunique())
            detail["targets"][label] = {
                "positive_frames": positive,
                "negative_frames": len(df) - positive,
                "positive_rate_accepted": round(positive / len(df), 4),
                "positive_videos": positive_videos,
            }
            if split in ("train", "val"):
                if positive < min_positive_frames:
                    detail["warnings"].append(f"too_few_positive_frames:{label}")
                if positive_videos < min_positive_videos:
                    detail["warnings"].append(f"too_few_positive_videos:{label}")
        result["splits"][split] = detail
    if not result["splits"]:
        raise ValueError("No extracted split CSVs found")
    if not {"train", "val"} <= set(result["splits"]):
        result["warnings"].append("train_and_validation_both_required_before_training")
    result["warnings"].extend([
        "Publisher splits are video-ID-disjoint, not confirmed subject-disjoint",
        "A zero label means only absence of the corresponding annotated error",
        "Frame-level labels and adjacent frames are correlated; video-level metrics are needed",
    ])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/fitness_aqa"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fitness_aqa_qa.json"))
    options = parser.parse_args()
    try:
        report = audit(options.features_dir)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"QA failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
