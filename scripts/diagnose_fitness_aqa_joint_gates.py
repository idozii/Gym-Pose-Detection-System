#!/usr/bin/env python3
"""Counterfactual joint-selective visibility audit using EXISTING pose-QA CSVs.

No ZIP/video access, model retraining, test split access or feature edits.
Uses the 0.50-gate QA output: the minimum critical-joint visibility and
identities of all critical joints below 0.50. Consequently exact acceptance
for these specified 0.35 alternatives can be reconstructed *only* when QA
was performed with baseline gate 0.50.

These sampled videos were annotation-stratified; resulting coverage estimates
are not representative of the original source population.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts.audit_fitness_aqa_pose import TARGETS

KNEES = frozenset(("left_knee", "right_knee"))
ANKLES = frozenset(("left_ankle", "right_ankle"))
ALL_CRITICAL = frozenset((
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
))
GATES = {
    "all_050": frozenset(),
    "ankles_035_other_050": ANKLES,
    "knees_035_other_050": KNEES,
    "knees_ankles_035_other_050": KNEES | ANKLES,
    "all_035": ALL_CRITICAL,
}
VALID_REASONS = frozenset((
    "accepted", "no_pose_detected", "invalid_landmark_structure",
    "nonfinite_coordinates_or_visibility", "critical_joint_low_visibility",
    "degenerate_shoulder_distance", "other_feature_rejection",
))


def counterfactual_accept(reason: str, failing_names: str, min_visibility,
                          *, relaxed_joints: frozenset[str],
                          relaxed_cutoff: float = 0.35) -> bool:
    """Exact if original reason/joint names came from baseline 0.50 QA."""
    if not 0 <= relaxed_cutoff <= .5:
        raise ValueError("Cutoff must fall inside [0,0.5]")
    if not relaxed_joints <= ALL_CRITICAL:
        raise ValueError("Unknown critical joint names")
    if reason not in VALID_REASONS:
        raise ValueError(f"Unexpected rejection reason {reason!r}")
    if reason == "accepted":
        return True
    if reason != "critical_joint_low_visibility":
        return False
    bad = frozenset(n for n in str(failing_names).split("|") if n)
    if not bad or not bad <= ALL_CRITICAL:
        raise ValueError("Missing or unknown low-visibility joint names")
    if pd.isna(min_visibility):
        raise ValueError("Missing minimum critical joint visibility")
    minimum = float(min_visibility)
    if not math.isfinite(minimum):
        raise ValueError("Nonfinite minimum critical joint visibility")
    return bad <= relaxed_joints and minimum >= relaxed_cutoff


def summarize_split(df: pd.DataFrame, *, expected_050: int, expected_035: int) -> dict:
    if df.empty:
        raise ValueError("Quality CSV is empty")
    required = {
        "video_id", "sample_id", "rejection_reason", "low_visibility_joints",
        "minimum_critical_joint_visibility", *TARGETS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if not all(df[t].isin([0,1]).all() for t in TARGETS):
        raise ValueError("Expected binary annotation targets")
    per_gate = {}
    baseline_mask = df.rejection_reason.eq("accepted").to_numpy()
    for gate, relaxed in GATES.items():
        mask = df.apply(
            lambda r: counterfactual_accept(
                r["rejection_reason"], r["low_visibility_joints"],
                r["minimum_critical_joint_visibility"], relaxed_joints=relaxed),
            axis=1,
        ).to_numpy(dtype=bool)
        if (baseline_mask & ~mask).any():
            raise AssertionError("A relaxed gate must include all baseline accepted frames")
        targets = {}
        for target in TARGETS:
            y = df[target].to_numpy(dtype=int)
            positive = y == 1
            negative = y == 0
            targets[target] = {
                "positive_eligible_frames": int(positive.sum()),
                "positive_accepted": int((mask & positive).sum()),
                "new_positive_vs_050": int(((mask & ~baseline_mask) & positive).sum()),
                "positive_acceptance_fraction": (
                    float((mask & positive).sum() / positive.sum()) if positive.any() else None
                ),
                "negative_accepted": int((mask & negative).sum()),
                "new_negative_vs_050": int(((mask & ~baseline_mask) & negative).sum()),
                "negative_acceptance_fraction": (
                    float((mask & negative).sum() / negative.sum()) if negative.any() else None
                ),
            }
        per_gate[gate] = {
            "eligible_frames": len(df), "accepted_frames": int(mask.sum()),
            "accepted_fraction": float(mask.mean()),
            "additional_frames_over_all_050": int((mask & ~baseline_mask).sum()),
            "videos_with_accepted_frames": int(df.loc[mask, "video_id"].nunique()),
            "error_targets": targets,
        }
    if per_gate["all_050"]["accepted_frames"] != expected_050:
        raise ValueError("Baseline 0.50 count differs from original pose QA report")
    if per_gate["all_035"]["accepted_frames"] != expected_035:
        raise ValueError("All-joint 0.35 count differs from original QA sensitivity report")
    return per_gate


def diagnose(input_dir: str | Path, output_path: str | Path) -> dict:
    root, out = Path(input_dir), Path(output_path)
    source = json.loads((root / "pose_quality_report.json").read_text(encoding="utf-8"))
    if (source.get("status") != "pose_quality_DIAGNOSTIC_only_no_training_no_test"
        or source.get("baseline_min_visibility") != .5):
        raise ValueError("Require baseline-0.50 pose-quality audit, not a different experiment")
    result = {
        "status": "joint_selective_counterfactual_visibility_DIAGNOSTIC_only",
        "source": "Previously sampled, annotation-stratified train/val quality CSVs",
        "gate_relaxed_joint_names": {g: sorted(k) for g, k in GATES.items()},
        "splits": {},
        "warnings": [
            "Coverage simulated only; no model was trained or evaluated",
            "Original 90 videos were annotation-stratified, so their rates are NOT representative",
            "Source validation already repeatedly examined for exploratory research",
            "Higher visibility coverage may introduce noisy joint coordinates",
            "A newly accepted knee-occluded frame may harm inward-error recognition",
            "No test data or original ZIP/video accessed",
        ],
    }
    for split in ("train", "val"):
        path = root / f"quality_{split}_samples.csv"
        df = pd.read_csv(path, keep_default_na=True)
        # Original QA recorded frame_index but not sample_id. Verify no duplicates
        # by the (video_id, frame_index) pair and construct a stable local ID.
        if "sample_id" not in df.columns:
            if {"video_id","frame_index"} - set(df.columns):
                raise ValueError("CSV lacks video_id or frame_index")
            if df.duplicated(["video_id", "frame_index"]).any():
                raise ValueError("Duplicate frame indices inside a video")
            df["sample_id"] = df["video_id"].astype(str) + "_f" + df["frame_index"].astype(str)
        s = source["splits"][split]
        if len(df) != int(s["eligible_frames"]):
            raise ValueError(f"{split}: QA CSV length does not match report")
        baseline = int(s["accepted_frames"])
        cutoff = s["cutoff_accepted_counts_same_eligible_frames"]
        key = "0.35"
        if key not in cutoff:
            raise ValueError(f"{split}: original QA did not record threshold 0.35")
        result["splits"][split] = {
            "n_sampled_videos": int(s["selected_videos"]),
            "gates": summarize_split(df, expected_050=baseline,
                                     expected_035=int(cutoff[key]["accepted"])),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("artifacts/fitness_aqa_pose_quality"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/fitness_aqa_joint_gates/counterfactual.json"))
    args = parser.parse_args()
    try:
        report = diagnose(args.input_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Joint gate diagnosis failed: {exc}\n")
    print(json.dumps({
        "status": report["status"], "splits": report["splits"],
        "full_report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
