"""Synthetic original source Squat labels, groups, and new registry artifacts.

No publisher source ZIP, TEST frames or production joblib is read.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from gym_pose.model_registry import DEFAULT_PATHS, ModelRegistry
from scripts import train_fitness_aqa_squat_source_errors as squat


def fixtures(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    rng = np.random.default_rng(42)
    summary = {
        "dataset": "Fitness-AQA squat labeled",
        "schema_version": SCHEMA_VERSION,
        "split_type": "publisher_video_id_split_NOT_subject_disjoint",
        "sampling_fps": 2.0,
        "boundary_margin_seconds": 0.1,
        "min_pose_visibility": 0.5,
        "limit_per_split": 8,
        "splits": {},
    }
    for split, num_videos in (("train", 48), ("val", 18)):
        rows = []
        for video in range(num_videos):
            video_id = f"{1100 if split == 'train' else 2200 + video}_{video:03d}"
            for frame in range(4):
                fwd = int(video % 3 == 0 and frame in (1, 2, 3))
                inward = int(video % 4 == 0 and frame in (1, 2, 3))
                numeric = {
                    name: float(rng.normal(.3 + .3 * fwd + .1 * inward, .04))
                    for name in FEATURE_NAMES
                }
                rows.append({
                    "sample_id": f"{video_id}_f{frame:07d}",
                    "video_id": video_id, "split": split,
                    "timestamp_s": frame / 2.0,
                    "frame_index": frame,
                    "source_fps": 30.,
                    "error_knees_forward": fwd,
                    "error_knees_inward": inward,
                    "any_of_two_annotated_errors": int(fwd or inward),
                    **numeric,
                })
        df = pd.DataFrame(rows, columns=[
            *squat.META_COLUMNS, *FEATURE_NAMES,
        ])
        df.to_csv(root / f"fitness_aqa_{split}_pose.csv", index=False)
        summary["splits"][split] = {
            "selected_videos": num_videos, "videos_processed": num_videos,
            "videos_failed": 0,
            "pose_accepted": len(df), "pose_rejected": 0,
            "frames_sampled": len(df), "boundary_rejected": 0,
            "annotated_knees_forward_frames": int(df.error_knees_forward.sum()),
            "annotated_knees_inward_frames": int(df.error_knees_inward.sum()),
        }
    (root / squat.MANIFEST_NAME).write_text(json.dumps(summary))
    # Poison-pill legacy test CSV. If opened it will fail CSV shape checks.
    (root / "fitness_aqa_test_pose.csv").write_text(
        "THIS CONTENT IS SOURCE TEST AND MUST NEVER BE READ")
    return root


def landmarks(low=()):
    return [
        SimpleNamespace(x=.03 * i, y=.02 * (i + i % 2), z=.002 * i,
                        visibility=.1 if i in low else .99)
        for i in range(33)
    ]


def registry_with_only_synthetic_squat_paths(tmp_path, output):
    paths = {
        key: tmp_path / "absent" / key[0] / original.name
        for key, original in DEFAULT_PATHS.items()
    }
    for target in squat.TARGETS:
        paths["squat", target] = output / f"rf_{target}_static_source.joblib"
    return ModelRegistry(paths=paths)


def test_full_dataset_provenance_refuses_pilot(tmp_path):
    root = fixtures(tmp_path)
    with pytest.raises(ValueError, match="Pilot"):
        squat.load_source(root)
    x, summary = squat.load_source(root, require_full=False)
    assert set(x) == {"train", "val"}
    assert len(x["train"]) == 192
    assert len(x["val"]) == 72
    assert summary["splits"]["val"]["videos_failed"] == 0


def test_training_group_calibration_disjoint_by_video(tmp_path):
    root = fixtures(tmp_path)
    rows, _ = squat.load_source(root, require_full=False)
    for target in squat.TARGETS:
        fit, cal = squat.train_group_holdout(
            rows["train"], target, seed=42)
        left, right = rows["train"].iloc[fit], rows["train"].iloc[cal]
        assert set(left.video_id).isdisjoint(set(right.video_id))
        assert set(left[target]) == set(right[target]) == {0, 1}
        assert left.loc[left[target].eq(1), "video_id"].nunique() >= 5
        assert right.loc[right[target].eq(1), "video_id"].nunique() >= 3


def test_new_squat_models_end_to_end_load_both_and_abstain(tmp_path):
    root = fixtures(tmp_path)
    output = tmp_path / "new_models"
    report = squat.train_models(root, output, require_full=False,
                                seed=42, trees=14)
    assert report["original_publisher_TEST_media_or_CSV_accessed"] is False
    assert report["thresholds_selected_on_source_TRAIN_only"] is True
    assert set(report["tasks"]) == set(squat.TARGETS)
    assert (output / "validation_report.json").exists()
    for target in squat.TARGETS:
        task = report["tasks"][target]
        assert task["train_calibration"]["video_groups_disjoint"]
        assert task["train_positive_videos"] >= 5
        assert task["validation_positive_videos"] >= 5
        assert 0 < task["train_calibration"]["threshold"] < 1
        assert 0 <= task["validation_AP"] <= 1
        model = joblib.load(output / f"rf_{target}_static_source.joblib")
        assert model["annotation_semantics"] == "publisher_named_error_interval_NOT_correctness"
        assert model["schema_version"] == "squat-103-v1"
        assert len(model["feature_names"]) == 103
        assert model["target"] == target

    registry = registry_with_only_synthetic_squat_paths(tmp_path, output)
    with pytest.raises(ValueError, match="explicitly opt"):
        registry.load_local()
    statuses = registry.load_local(trust_local_pickle_files=True)
    assert all(statuses[f"squat/{name}"] == "loaded_experimental_only"
               for name in squat.TARGETS)
    assert statuses["ohp/error_elbows"] == "model_artifact_missing"
    assert statuses["barbell_row/lumbar"] == "model_artifact_missing"
    for name in squat.TARGETS:
        result = registry.score_landmarks("squat", name, landmarks())
        assert result["status"] == "experimental_source_label_score"
        assert result["feature_count"] == 103
        assert isinstance(result["source_label_flag"], bool)
        assert registry.score_landmarks("squat", name, landmarks(low=(25,)))["status"] == "abstain"


def test_wrong_squat_semantics_or_legacy_correctness_artifact_fail_closed(tmp_path):
    root = fixtures(tmp_path)
    output = tmp_path / "new_models"
    squat.train_models(root, output, trees=8, require_full=False)
    forward = output / "rf_error_knees_forward_static_source.joblib"
    payload = joblib.load(forward)
    payload["annotation_semantics"] = "notebook_correct_incorrect"
    joblib.dump(payload, forward)
    registry = registry_with_only_synthetic_squat_paths(tmp_path, output)
    with pytest.raises(ValueError, match="named publisher knee-error"):
        registry.load_local(trust_local_pickle_files=True)


def test_rejects_split_overlap_and_stale_positive_counts(tmp_path):
    root = fixtures(tmp_path)
    val = root / "fitness_aqa_val_pose.csv"
    df = pd.read_csv(val)
    df.loc[0, "video_id"] = "1100_000"
    df.loc[0, "sample_id"] = "1100_000_f0000000"
    df.to_csv(val, index=False)
    with pytest.raises(ValueError, match="VIDEO ID overlap"):
        squat.load_source(root, require_full=False)


def test_manifest_with_extracted_source_test_disallowed(tmp_path):
    root = fixtures(tmp_path)
    f = root / squat.MANIFEST_NAME
    data = json.loads(f.read_text())
    data["splits"]["test"] = {
        "selected_videos": 244, "videos_processed": 244,
    }
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source TEST"):
        squat.load_source(root, require_full=False)


def test_invalid_settings_and_annotation_failure_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid RF"):
        squat.train_models(tmp_path, tmp_path, trees=0)
    with pytest.raises(ValueError, match="Unknown Squat"):
        squat.train_group_holdout(pd.DataFrame(), "correct")
    root = fixtures(tmp_path)
    df = pd.read_csv(root / "fitness_aqa_train_pose.csv")
    df.loc[0, "error_knees_forward"] = 3
    df.to_csv(root / "fitness_aqa_train_pose.csv", index=False)
    with pytest.raises(ValueError, match="nonbinary"):
        squat.load_source(root, require_full=False)
