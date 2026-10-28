"""M48 report freeze tests use synthetic JSON shaped like observed TRAIN data.

No publisher data, TEST video, validation media or real user documents.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

import scripts.m48_freeze_repcount_evidence as m48


def score(tp, fp, fn, mae):
    pred = tp + fp
    gt = tp + fn
    return {
        "videos": 20, "annotated_reps": gt, "predicted_reps": pred,
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "event_f1": 2 * tp / (2 * tp + fp + fn),
        "mean_absolute_count_error": mae,
        "mean_signed_count_error": -1.0
    }


def fixture_reports():
    m46 = {
        "protocol": m48.M45_PROTOCOL, "source_split": "train",
        "study_type": "prespecified_unseen_train_source_file_replication",
        "publisher_train_annotation_sha256": "f" * 64,
        "source_selection": {
            "policy": "stable_hash", "seed": 2026, "limit": 20,
            "source_hash_rank_offset": 20,
            "source_clip_filename_disjoint_from_m40_m45": True,
            "subject_or_view_disjointness_verified": False,
            "pose_observation_target_ms": 200,
            "400ms_subsample_shares_dense_tracker_history": True
        },
        "videos_selected": 20, "videos_evaluated": 20,
        "videos_excluded": 0, "annotated_reps": 209,
        "relative_excursion_rule": m48.EXPECTED_RULE,
        "no_validation_or_test_access": True,
        "deployed_rep_counter_unchanged": True,
        "conditions": {
            cond: {"methods": {
                name: score(*expected) for name, expected in methods.items()
            }} for cond, methods in m48.M46_OWNER_COUNTS.items()
        }
    }
    raw = json.dumps(m46, indent=2).encode("utf8")
    m47 = {
        "protocol": m48.M47_PROTOCOL, "source_split": "train",
        "original_m46_report_sha256": hashlib.sha256(raw).hexdigest(),
        "publisher_train_annotation_sha256": "f" * 64,
        "frozen_source": {
            "selection_policy": "stable_hash", "selection_seed": 2026,
            "hash_rank_offset": 20, "videos_evaluated": 20,
            "videos_excluded": 0, "annotated_gt_reps": 209,
            "source_filenames_disjoint_from_original_first_20": True,
            "subject_and_session_disjointness_verified": False
        },
        "analysis_settings": {
            "paired_cluster_unit": "source_video",
            "bootstrap_method": "percentile",
            "quantiles": [.025, .975],
            "draws": 5000, "seed": 2026,
            "difference_orientation": "relative_excursion_minus_low_to_high",
            "two_sided_test_or_p_value_performed": False
        },
        "rule_unchanged": m48.EXPECTED_RULE,
        "recomputed_frozen_m46_metrics": {
            cond: copy.deepcopy(item["methods"])
            for cond, item in m46["conditions"].items()
        },
        "paired_clip_bootstrap": {},
        "validation_and_test_untouched": True,
        "deployed_m34_unchanged": True
    }
    for cond, methods in m46["conditions"].items():
        baseline = methods["methods"]["low_to_high"]
        candidate = methods["methods"]["relative_excursion"]
        differences = {
            "event_f1": candidate["event_f1"] - baseline["event_f1"],
            "mean_absolute_count_error": (
                candidate["mean_absolute_count_error"] -
                baseline["mean_absolute_count_error"]
            ),
            "false_positive_events_per_video": (
                candidate["event_fp"] - baseline["event_fp"]
            ) / 20
        }
        m47["paired_clip_bootstrap"][cond] = {
            name: {
                "observed_relative_minus_low_to_high": delta,
                "percentile_95_interval": list(m48.M47_OWNER_INTERVALS[cond][name]),
                "bootstrap_draws_strictly_above_zero": 3000,
                "bootstrap_draws_equal_to_zero": 1000,
                "bootstrap_draws_strictly_below_zero": 1000
            } for name, delta in differences.items()
        }
    return m46, raw, m47


def test_m48_strict_owner_report_count_and_bootstrap_ci_anchor():
    m46, raw, m47 = fixture_reports()
    outcomes = m48.validate_evidence(m46, m47, raw)
    assert set(outcomes) == {"dense_200ms", "subsample_400ms"}
    assert pytest.approx(outcomes["dense_200ms"]["event_f1"][
        "observed_relative_minus_low_to_high"
    ]) == 0.04296203718169034
    assert outcomes["subsample_400ms"]["mean_absolute_count_error"][
        "video_bootstrap_percentile_95"
    ] == [-2.45, -0.20000000000000018]


def test_m48_refuses_tampered_original_scores_and_stats():
    m46, raw, m47 = fixture_reports()
    modified = copy.deepcopy(m46)
    modified["conditions"]["dense_200ms"]["methods"]["relative_excursion"][
        "event_tp"
    ] = 130
    with pytest.raises(ValueError, match="owner-observed count mismatch"):
        m48.validate_evidence(modified, m47, raw)
    modified = copy.deepcopy(m47)
    modified["original_m46_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source/protocol/reference"):
        m48.validate_evidence(m46, modified, raw)
    modified = copy.deepcopy(m47)
    modified["paired_clip_bootstrap"]["dense_200ms"]["event_f1"][
        "observed_relative_minus_low_to_high"
    ] = -0.1
    with pytest.raises(ValueError, match="owner-observed value mismatch"):
        m48.validate_evidence(m46, modified, raw)
    modified = copy.deepcopy(m47)
    modified["paired_clip_bootstrap"]["dense_200ms"]["event_f1"][
        "percentile_95_interval"
    ] = [-0.3, -0.1]
    with pytest.raises(ValueError, match="owner-observed value mismatch"):
        m48.validate_evidence(m46, modified, raw)
    modified = copy.deepcopy(m47)
    modified["paired_clip_bootstrap"]["dense_200ms"]["event_f1"][
        "bootstrap_draws_strictly_above_zero"
    ] += 1
    with pytest.raises(ValueError, match="do not sum to 5000"):
        m48.validate_evidence(m46, modified, raw)


def _repo(tmp_path):
    root = tmp_path / "repo"
    for path in m48.SOURCES:
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic dummy source for sha check " + path)
    return root


def test_m48_writes_only_fingerprints_and_verifies_read_only(tmp_path, monkeypatch):
    m46, raw, m47 = fixture_reports()
    m46_path = tmp_path / "m46.json"
    m47_path = tmp_path / "m47.json"
    lock_path = tmp_path / "m48.json"
    m46_path.write_bytes(raw)
    m47_path.write_text(json.dumps(m47))
    repo = _repo(tmp_path)
    monkeypatch.setattr(m48, "_revision", lambda root: "a" * 40)

    frozen = m48.run(m46_path=m46_path, m47_path=m47_path,
                     report_path=lock_path, repository_root=repo)
    assert frozen["source_split"] == "train"
    assert frozen["known_protocol"]["ground_truth_reps"] == 209
    assert frozen["known_protocol"]["bootstrap_draws"] == 5000
    assert frozen["known_protocol"]["single_predeclared_test_execution_permitted_by_this_script"] is False
    assert frozen["test_or_validation_media_scanned"] is False
    assert len(frozen["source_code_sha256"]) == len(m48.SOURCES)
    assert "video_path" not in lock_path.read_text()
    assert '"landmarks"' not in lock_path.read_text()
    assert '"frames"' not in lock_path.read_text()
    assert "source_squat.mp4" not in lock_path.read_text()
    checked = m48.verify_existing_lock(
        lock_path=lock_path, m46_path=m46_path, m47_path=m47_path,
        repository_root=repo
    )
    assert checked["verified_original_report_bytes"] is True
    assert checked["verified_frozen_source_bytes"] is True

    with pytest.raises(FileExistsError, match="refuses overwrite"):
        m48.run(m46_path=m46_path, m47_path=m47_path,
                report_path=lock_path, repository_root=repo)

    one_file = repo / m48.SOURCES[0]
    one_file.write_text(one_file.read_text() + "\nchanged")
    with pytest.raises(ValueError, match="source-code SHA256 changed"):
        m48.verify_existing_lock(
            lock_path=lock_path, m46_path=m46_path, m47_path=m47_path,
            repository_root=repo
        )
    one_file.write_text("synthetic dummy source for sha check " + m48.SOURCES[0])
    m47_path.write_text(json.dumps({**m47, "new_fake_claim": True}))
    with pytest.raises(ValueError, match="report SHA256 changed"):
        m48.verify_existing_lock(
            lock_path=lock_path, m46_path=m46_path, m47_path=m47_path,
            repository_root=repo
        )


def test_m48_fails_closed_without_reports_or_source_code(tmp_path, monkeypatch):
    m46, raw, m47 = fixture_reports()
    root = tmp_path / "repo"
    root.mkdir()
    p46 = tmp_path / "m46.json"
    p47 = tmp_path / "m47.json"
    p46.write_bytes(raw)
    p47.write_text(json.dumps(m47))
    monkeypatch.setattr(m48, "_revision", lambda root: "a" * 40)
    with pytest.raises(ValueError, match="research source file unavailable"):
        m48.run(m46_path=p46, m47_path=p47,
                report_path=tmp_path/"lock.json", repository_root=root)
    with pytest.raises(FileNotFoundError, match="requires existing"):
        m48.run(m46_path=tmp_path/"no.json", m47_path=p47,
                report_path=tmp_path/"lock.json", repository_root=root)
