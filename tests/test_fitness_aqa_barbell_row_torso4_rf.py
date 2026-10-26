"""Synthetic tests for Barbell Row torso4 image-only RF benchmark.

No publisher source image bytes, MediaPipe or TEST-set content needed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.barbell_row_features import TARGETS
from gym_pose.barbell_row_torso4_features import FEATURE_NAMES, SCHEMA_VERSION
from scripts import benchmark_fitness_aqa_barbell_row_torso4 as row


def source_fixture(tmp_path, *, pilot=True):
    root = tmp_path / "features"
    root.mkdir()
    rng = np.random.default_rng(64)
    report = {
        "status": "barbell_row_image_pose_extraction_source_train_val_NO_TEST",
        "feature_profile": "torso4",
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": 20,
        "source_test_images_decoded": 0,
        "cross_target_split_id_conflicts": 0,
        "min_visibility": .5,
        "source_split_type": "publisher_TASK_SPECIFIC_image_ID_NOT_verified_subject_or_clip_disjoint",
        "max_images_per_task_split": 12 if pilot else 0,
        "candidate_clip_prefix_audit": {
            t: {"candidate_prefix_overlap_train_val": 0} for t in TARGETS
        },
        "splits": {},
    }
    for split, groups in (("train", 48), ("val", 18)):
        task_stats = {}
        for task_i, task in enumerate(TARGETS):
            records = []
            for g in range(groups):
                prefix = f"{100 + task_i if split == 'train' else 200 + task_i}_{g:03d}"
                for k in range(4):
                    label = int((g + task_i) % 3 == 0 or (g + task_i) % 7 == 1)
                    image_id = f"{prefix}_{k}"
                    features = {
                        f: float(rng.normal(.8 if label else .2, .2))
                        for f in FEATURE_NAMES
                    }
                    records.append({
                        "image_id": image_id, "split": split,
                        "target": task, "error_label": label,
                        "candidate_clip_prefix": prefix, **features,
                    })
            df = pd.DataFrame(records, columns=[*row.META, *FEATURE_NAMES])
            df.to_csv(
                root / f"fitness_aqa_barbell_row_torso4_{task}_{split}_pose.csv",
                index=False)
            n = len(df)
            positives = int(df.error_label.sum())
            task_stats[task] = {
                "selected_images": n,
                "pose_accepted": n,
                "pose_rejected": 0,
                "positive_accepted": positives,
                "negative_accepted": n - positives,
                "positive_selected": positives,
                "pose_coverage_selected": 1.0,
            }
        report["splits"][split] = {
            "unique_images_to_decode": groups * 4 * len(TARGETS),
            "images_decoded": groups * 4 * len(TARGETS),
            "image_decode_failures": 0,
            "tasks": task_stats,
        }
    (root / "fitness_aqa_barbell_row_torso4_extraction_summary.json").write_text(
        json.dumps(report))
    (root / "fitness_aqa_barbell_row_torso4_lumbar_test_pose.csv").write_text(
        "INVALID TEST DATA MUST NEVER BE OPENED")
    return root


def test_source_publisher_task_separation_and_no_test(tmp_path):
    root = source_fixture(tmp_path)
    records, summary = row._load(root, require_full=False)
    assert set(records) == set(TARGETS)
    assert summary["source_test_images_decoded"] == 0
    for task in TARGETS:
        train, val = records[task]["train"], records[task]["val"]
        assert set(train.image_id).isdisjoint(val.image_id)
        assert set(train.candidate_clip_prefix).isdisjoint(val.candidate_clip_prefix)
        assert set(train.error_label) == {0, 1}
        assert set(val.error_label) == {0, 1}


def test_refuses_pilot_for_user_benchmark(tmp_path):
    root = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="Pilot"):
        row._load(root, require_full=True)


def test_source_count_and_label_mismatch_detected(tmp_path):
    root = source_fixture(tmp_path)
    stats = root / "fitness_aqa_barbell_row_torso4_extraction_summary.json"
    payload = json.loads(stats.read_text())
    payload["splits"]["val"]["tasks"]["lumbar"]["positive_accepted"] += 1
    stats.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary"):
        row._load(root, require_full=False)


def test_grouped_train_calibration_uses_distinct_candidate_prefixes(tmp_path):
    root = source_fixture(tmp_path)
    frames, _ = row._load(root, require_full=False)
    for task in TARGETS:
        train = frames[task]["train"]
        fit, cal = row.train_only_prefix_calibration(train, seed=42)
        assert set(train.iloc[fit].candidate_clip_prefix).isdisjoint(
            train.iloc[cal].candidate_clip_prefix)
        assert train.iloc[fit].loc[
            train.iloc[fit].error_label.eq(1),
            "candidate_clip_prefix"].nunique() >= 5
        assert train.iloc[cal].loc[
            train.iloc[cal].error_label.eq(1),
            "candidate_clip_prefix"].nunique() >= 3


def test_train_val_candidate_prefix_leakage_halts(tmp_path):
    root = source_fixture(tmp_path)
    file = root / "fitness_aqa_barbell_row_torso4_lumbar_val_pose.csv"
    df = pd.read_csv(file)
    new_id = "101_000_0"  # same task's source train candidate prefix
    df.loc[0, "image_id"] = new_id
    df.loc[0, "candidate_clip_prefix"] = new_id.rsplit("_", 1)[0]
    df.to_csv(file, index=False)
    with pytest.raises(ValueError, match="candidate clip prefixes"):
        row._load(root, require_full=False)


def test_invalid_feature_or_duplicate_image_fails(tmp_path):
    root = source_fixture(tmp_path)
    file = root / "fitness_aqa_barbell_row_torso4_torso_angle_train_pose.csv"
    df = pd.read_csv(file)
    df.loc[1, "image_id"] = df.loc[0, "image_id"]
    df.to_csv(file, index=False)
    with pytest.raises(ValueError, match="duplicated publisher image"):
        row._load(root, require_full=False)


def test_end_to_end_rf_validation_without_loading_source_test(tmp_path):
    root = source_fixture(tmp_path)
    out = tmp_path / "artifacts"
    report = row.benchmark(
        root, out, trees=12, seed=42, bootstraps=100, require_full=False)
    assert report["source_test_images_opened"] is False
    assert report["source_training_only_threshold_selection"] is True
    assert set(report["tasks"]) == set(TARGETS)
    assert (out / "comparison.json").is_file()
    for task in TARGETS:
        assert (out / f"barbell_row_{task}_torso4_static_rf.joblib").is_file()
        entry = report["tasks"][task]
        assert entry["train_pose_accepted"] > entry["validation_pose_accepted"]
        assert entry["validation_positive_accepted"] > 20
        assert entry["calibration"]["prefix_group_disjoint"] is True
        assert entry["validation_prefix_bootstrap"]["resamples_valid"] >= 80
        assert 0 <= entry["validation_average_precision"] <= 1
        assert 0 < entry["calibration"]["train_only_selected_threshold"] < 1
        assert entry["validation_train_selected_threshold"]["macro_f1"] <= 1


def test_invalid_args_before_loading_source(tmp_path):
    with pytest.raises(ValueError, match="Invalid trees"):
        row.benchmark(tmp_path, tmp_path / "models", trees=0)
    with pytest.raises(ValueError, match="At least 100"):
        row.prefix_group_bootstrap(
            np.array([0, 1]), np.array([.1, .8]), np.array(["a", "b"]),
            .5, repetitions=5, seed=42)
