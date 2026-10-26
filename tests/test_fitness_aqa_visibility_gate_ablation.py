"""Synthetic-only regression tests for source-split pose gate sensitivity."""
import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES
from scripts.compare_fitness_aqa_visibility import (
    _align, _shared_group_split, experiment, load_pair,
)


def _create(tmp_path, gate, *, invalid_labels=False):
    root = tmp_path / gate
    root.mkdir()
    summary = {
        "split_type": "publisher_video_id_split_NOT_subject_disjoint",
        "min_pose_visibility": 0.5 if gate == "gate050" else 0.35,
        "sampling_fps": 2.0, "boundary_margin_seconds": .1,
        "splits": {},
    }
    rng = np.random.default_rng(123)
    for split, clips in (("train", 16), ("val", 8)):
        records = []
        for video in range(clips):
            for frame in range(8):
                if gate == "gate050" and frame < 2:
                    continue
                values = rng.normal(size=103)
                r = dict(zip(FEATURE_NAMES, values))
                r.update({
                    "sample_id": f"{split}_{video}_f{frame}",
                    "video_id": f"{split}_{video}",
                    "split": split, "timestamp_s": .5 * frame, "frame_index": frame,
                    "source_fps": 30.,
                    "error_knees_forward": int(frame in (3, 4)),
                    "error_knees_inward": int(frame in (5, 6)),
                })
                r["any_of_two_annotated_errors"] = max(
                    r["error_knees_forward"], r["error_knees_inward"])
                records.append(r)
        if invalid_labels and split == "val":
            records[2]["error_knees_inward"] = int(not records[2]["error_knees_inward"])
            records[2]["any_of_two_annotated_errors"] = max(
                records[2]["error_knees_forward"], records[2]["error_knees_inward"])
        pd.DataFrame(records).to_csv(root / f"fitness_aqa_{split}_pose.csv", index=False)
        summary["splits"][split] = {
            "selected_videos": clips, "frames_sampled": clips * 8,
            "boundary_rejected": 0, "pose_accepted": len(records),
            "pose_rejected": clips * 8 - len(records),
        }
    (root / "fitness_aqa_extraction_summary.json").write_text(json.dumps(summary))
    return root


def test_matching_and_original_only_counts(tmp_path):
    a = _create(tmp_path, "gate050")
    b = _create(tmp_path, "gate035")
    tables, _ = load_pair(a, b)
    matched = _align(tables["gate050"]["val"], tables["gate035"]["val"], split="val")
    assert matched["common_frame_count"] == 48
    assert matched["original_only_count"] == 0
    assert matched["relaxed_only_count"] == 16


def test_shared_calibration_uses_disjoint_source_video_ids(tmp_path):
    a = _create(tmp_path, "gate050")
    b = _create(tmp_path, "gate035")
    tables, _ = load_pair(a, b)
    fit, cal = _shared_group_split({k: v["train"] for k, v in tables.items()}, seed=42)
    assert fit.isdisjoint(cal)
    assert len(fit) + len(cal) == 16


def test_complete_experiment_never_reads_test(tmp_path):
    a = _create(tmp_path, "gate050")
    b = _create(tmp_path, "gate035")
    (a / "fitness_aqa_test_pose.csv").write_text("CORRUPT,TEST,INPUT\n")
    (b / "fitness_aqa_test_pose.csv").write_text("CORRUPT,TEST,INPUT\n")
    output = tmp_path / "reports" / "comparison.json"
    report = experiment(a, b, output, trees=8)
    assert report["status"] == "visibility_gate_ablation_train_calibrated_NO_test_read"
    assert report["matched_frame_audit"]["val"]["common_frame_count"] == 48
    assert report["gate_coverage"]["gate035"]["train"]["accepted_rate"] == 1.
    assert set(report["tasks"]) == {"error_knees_forward", "error_knees_inward"}
    for result in report["tasks"].values():
        for gate in ("gate050", "gate035"):
            assert result[gate]["all_accepted_validation_frames"]["n"] > 0
            assert result[gate]["common_accepted_validation_frames"]["n"] == 48
            assert 0 < result[gate]["threshold_from_source_train_calibration"] < 1
    assert output.exists()


def test_fails_on_inconsistent_annotation(tmp_path):
    a = _create(tmp_path, "gate050")
    b = _create(tmp_path, "gate035", invalid_labels=True)
    # Changed label at a shared frame should trigger the annotation mismatch.
    with pytest.raises(ValueError, match="differs on shared"):
        experiment(a, b, tmp_path / "report.json", trees=2)


def test_fails_if_cutoff_metadata_incorrect(tmp_path):
    a = _create(tmp_path, "gate050")
    b = _create(tmp_path, "gate035")
    summary = b / "fitness_aqa_extraction_summary.json"
    s = json.loads(summary.read_text())
    s["min_pose_visibility"] = .5
    summary.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="expected cutoff"):
        load_pair(a, b)
