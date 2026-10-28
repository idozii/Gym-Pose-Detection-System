"""M53 synthetic unit/integration tests. No publisher data or TEST accessed."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import m53_w6_train_metrics_parity as m53
from scripts.m39_repcount_train_baselines import _aggregate


def sample(gt, pred):
    tp = min(gt, pred)
    fp = pred - tp
    fn = gt - tp
    return {
        "annotated_count": gt, "predicted_count": pred,
        "absolute_count_error": abs(pred - gt),
        "signed_count_error": pred - gt,
        "event_tp": tp, "event_fp": fp, "event_fn": fn
    }


def pair(gt, base, candidate):
    return {
        "low_to_high": sample(gt, base),
        "relative_excursion": sample(gt, candidate)
    }


def test_m53_normalized_mae_divides_by_each_video_gt_not_global_total():
    items = [pair(2, 4, 2), pair(10, 9, 11)]
    got = m53.metrics_for_videos(items)
    base = got["low_to_high"]
    candidate = got["relative_excursion"]
    assert base["mean_normalized_absolute_count_error"] == pytest.approx(
        (2 / 2 + 1 / 10) / 2
    )
    assert candidate["mean_normalized_absolute_count_error"] == pytest.approx(
        (0 / 2 + 1 / 10) / 2
    )
    assert base["absolute_count_mae"] == 1.5
    assert candidate["absolute_count_mae"] == .5
    assert base["off_by_one_fraction"] == .5
    assert candidate["off_by_one_fraction"] == 1
    assert base["exact_count_fraction"] == 0
    assert candidate["exact_count_fraction"] == .5
    assert candidate["videos_with_exact_count"] == 1
    assert candidate["videos_with_off_by_one_or_better"] == 2
    assert base["annotated_gt_reps"] == 12


def test_m53_positive_gt_required_and_original_repetition_matches():
    for bad in [
        pair(0, 0, 0),
        pair(10, 10, 10) | {
            "relative_excursion": sample(11, 11)
        },
        pair(10, 10, 10) | {
            "relative_excursion": sample(10, 10) | {"event_tp": 999}
        },
        pair(10, 10, 10) | {
            "relative_excursion": sample(10, 10) |
                                  {"absolute_count_error": 999}
        },
        pair(10, 10, 10) | {
            "relative_excursion": sample(10, 10) |
                                  {"predicted_count": 999}
        }
    ]:
        with pytest.raises(ValueError):
            m53.metrics_for_videos([bad])
    with pytest.raises(ValueError):
        m53.metrics_for_videos([])
    with pytest.raises(ValueError):
        m53.metrics_for_videos([{"low_to_high": sample(3, 2)}])


def test_m53_paired_video_bootstrap_is_deterministic_and_conditioned_on_n20():
    items = [pair(2 + i % 9, 2 + i % 9 + 2,
                  2 + i % 9 + (i % 3 == 0)) for i in range(20)]
    a = m53.paired_video_bootstrap(items, repeats=100, seed=2026)
    b = m53.paired_video_bootstrap(items, repeats=100, seed=2026)
    assert a == b
    assert set(a) == set(m53.FIELDS)
    for metric, entry in a.items():
        assert set(entry) == {
            "candidate_minus_baseline", "percentile_95_interval"
        }
        assert len(entry["percentile_95_interval"]) == 2
        assert entry["percentile_95_interval"][0] <= entry[
            "percentile_95_interval"][1]
        assert entry["candidate_minus_baseline"] == pytest.approx(
            m53.metrics_for_videos(items)["relative_excursion"][metric] -
            m53.metrics_for_videos(items)["low_to_high"][metric]
        )
    with pytest.raises(ValueError):
        m53.paired_video_bootstrap(items[:19], repeats=100)


def make_w6_record(index):
    records = {}
    for name in m53.CONDITIONS:
        records[name] = {
            "source_frames": 20, "gate_sample_ms": 200,
            "gate_selected_frames": 20, "gate_dropped_frames": 0,
            "original_m45_event_count": 5,
            "browser_incremental_event_count": 5,
            "direct_same_input_event_disagreements": 0,
            "gate_reference_event_count": 5,
            "gate_browser_event_count": 5,
            "gate_same_input_event_disagreements": 0,
            "direct_exact_event_parity": True,
            "gated_exact_event_parity": True,
            "pose_estimator_parity_verified": False,
            "webpage_runtime_parity_verified": False
        }
    return {
        "protocol": m53.W6_PROTOCOL,
        "source_split": "train",
        "raw_frames_or_event_times_saved": False,
        "browser_or_python_pose_parity_claimed": False,
        "conditions": records
    }


def test_w6_aggregate_only_and_no_identity_retained():
    records = [make_w6_record(i) for i in range(20)]
    result = m53.summarize_w6(records)
    for cond in m53.CONDITIONS:
        row = result[cond]
        assert row["videos"] == 20
        assert row["source_frames"] == 400
        assert row["browser_incremental_event_count"] == 100
        assert row["all_same_landmark_event_times_equal"] is True
        assert row["all_gated_same_landmark_event_times_equal"] is True
        assert row["videos_with_direct_disagreement"] == 0
    with pytest.raises(ValueError):
        m53.summarize_w6(records[:-1])


def test_m53_w6_synthetic_end_to_end_owner_gate_m47_reference_and_no_raw_data(
        tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("m46.json", "m47.json", "m48.json")]
    for file in paths:
        file.write_text('{"synthetic":true}', encoding="utf8")
    out = tmp_path / "m53.json"
    w6 = tmp_path / "w6.json"
    monkeypatch.setattr(m53, "pretest_m49", lambda **_kwargs: {
        "research_source_provenance_check_passed": True,
        "official_test_executed_or_opened": False
    })
    rows = []
    for i in range(20):
        gt = 6 + i % 8
        base = sample(gt, gt + (i % 3) + 1)
        candidate = sample(gt, gt + (i % 2))
        rows.append({
            "annotated_rep_count": gt,
            "original_m44": {"m43": {"m41": {
                cond: {"low_to_high": dict(base)} for cond in m53.CONDITIONS
            }}},
            "conditions": {
                cond: {"relative_excursion": dict(candidate)}
                for cond in m53.CONDITIONS
            }
        })
    monkeypatch.setattr(m53.m47, "compare_original_m45", lambda path, out: (
        rows[int(path.stem.split("_")[-1])]
    ))
    monkeypatch.setattr(m53.m47, "validate_original_m45", lambda *_args: None)
    observed = {
        cond: {
            method: _aggregate([m53.m47._per_video(row, cond)
                                for row in rows], method)
            for method in m53.METHODS
        } for cond in m53.CONDITIONS
    }

    def fake_replay(**args):
        assert args["archive"] == tmp_path / "RepCountA.tar.gz"
        assert args["reference_path"] == paths[0]
        for i in range(20):
            args["compare_fn"](Path("pose_" + str(i) + ".json"),
                               tmp_path / ("result_" + str(i) + ".json"))
        return {
            "source_split": "train",
            "publisher_train_annotation_sha256": "a" * 64,
            "frozen_source": {
                "videos_evaluated": 20, "annotated_gt_reps":
                    sum(row["annotated_rep_count"] for row in rows),
                "subject_and_session_disjointness_verified": False
            },
            "validation_and_test_untouched": True,
            "recomputed_frozen_m46_metrics": observed
        }
    # Synthetic test deliberately differs from the frozen real-world
    # 209 GT total, so production run must reject rather than fake it.
    with pytest.raises(ValueError, match="post-replay aggregate mismatch"):
        m53.run(
            archive=tmp_path / "RepCountA.tar.gz", root=tmp_path / "temp",
            reference_path=paths[0], m47_path=paths[1],
            lock_path=paths[2], report_path=out,
            w6_report_path=w6, replay_fn=fake_replay,
            parity_fn=lambda _trace, _out: make_w6_record(0)
        )
    assert not out.exists()
    assert not w6.exists()
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest()  # original untouched


def test_m53_and_w6_reports_refuse_overwrite(tmp_path):
    a = tmp_path / "out.json"
    b = tmp_path / "w6.json"
    a.write_text("keep", encoding="utf8")
    with pytest.raises(FileExistsError):
        m53.run(archive=tmp_path / "data.tar.gz", root=tmp_path,
                reference_path=tmp_path / "m46.json",
                m47_path=tmp_path / "m47.json",
                lock_path=tmp_path / "m48.json", report_path=a,
                w6_report_path=b)
    assert a.read_text() == "keep"
