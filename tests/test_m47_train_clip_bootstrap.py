"""M47 tests on synthetic clips, no user or publisher media."""
from __future__ import annotations

import csv
import io
import json
import math
import tarfile

import pytest

import scripts.m47_train_clip_bootstrap as m47
from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m39_repcount_train_baselines import _aggregate
from scripts.m45_train_relative_motion_benchmark import (
    EXPECTED_RULE, PROTOCOL as M45_PROTOCOL
)


def method(gt=1, tp=0, fp=0):
    pred = tp + fp
    return {
        "annotated_count": gt,
        "predicted_count": pred,
        "absolute_count_error": abs(pred - gt),
        "signed_count_error": pred - gt,
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": gt - tp,
        "event_precision": tp / pred if pred else None,
        "event_recall": tp / gt if gt else None,
        "event_f1": (2 * tp / (2 * tp + fp + gt - tp)
                     if pred and gt else None)
    }


def paired(base, new):
    return {"low_to_high": base, "relative_excursion": new}


def test_bootstrap_is_exact_when_every_video_has_identical_frozen_outcomes():
    items = [paired(method(tp=0), method(tp=1)) for _ in range(20)]
    first = m47.paired_video_bootstrap(items, repeats=500, seed=2026)
    second = m47.paired_video_bootstrap(items, repeats=500, seed=2026)
    assert first == second
    assert first["event_f1"]["observed_relative_minus_low_to_high"] == 1
    assert first["event_f1"]["percentile_95_interval"] == [1, 1]
    assert first["mean_absolute_count_error"]["percentile_95_interval"] == [-1, -1]
    assert first["false_positive_events_per_video"]["percentile_95_interval"] == [0, 0]
    assert first["event_f1"]["bootstrap_draws_strictly_above_zero"] == 500
    assert first["event_f1"]["bootstrap_draws_strictly_below_zero"] == 0


def test_bootstrap_remains_pairwise_and_uses_same_clip_indices():
    a = [paired(method(gt=3, tp=1), method(gt=3, tp=2))
         for _ in range(10)]
    b = [paired(method(gt=4, tp=4), method(gt=4, tp=2, fp=1))
         for _ in range(10)]
    rows = a + b
    result = m47.paired_video_bootstrap(rows, repeats=1000, seed=99)
    assert result["event_f1"]["percentile_95_interval"][0] < (
        result["event_f1"]["percentile_95_interval"][1]
    )
    assert math.isclose(
        result["event_f1"]["observed_relative_minus_low_to_high"],
        m47._metric(rows, "event_f1")
    )
    assert result["mean_absolute_count_error"][
        "observed_relative_minus_low_to_high"
    ] == m47._metric(rows, "mean_absolute_count_error")
    assert sum(result["event_f1"][key] for key in (
        "bootstrap_draws_strictly_above_zero",
        "bootstrap_draws_equal_to_zero",
        "bootstrap_draws_strictly_below_zero"
    )) == 1000


def test_bootstrap_rejects_invalid_paired_clip_denominators():
    good = paired(method(), method(tp=1))
    bad = paired(method(gt=2), method(gt=1, tp=1))
    with pytest.raises(ValueError, match="different GT"):
        m47.paired_video_bootstrap([good, bad])
    bad = paired(method(gt=0), method(gt=0))
    with pytest.raises(ValueError, match="positive-GT"):
        m47.paired_video_bootstrap([good, bad])
    with pytest.raises(ValueError, match=">=2 paired"):
        m47.paired_video_bootstrap([good])
    with pytest.raises(ValueError, match=">=2 paired"):
        m47.paired_video_bootstrap([good, good], repeats=0)
    with pytest.raises(ValueError, match=">=2 paired"):
        m47.paired_video_bootstrap([good, good], seed=True)
    bad = paired(method(gt=1, tp=1), method(gt=1, tp=1))
    bad["relative_excursion"]["event_tp"] = 2
    with pytest.raises(ValueError, match="GT/event totals"):
        m47.paired_video_bootstrap([good, bad])


def _member(tf, path, content):
    t = tarfile.TarInfo(path)
    t.size = len(content)
    tf.addfile(t, io.BytesIO(content))


def _source(tmp_path):
    p = tmp_path / "RepCountA.tar.gz"
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "name", "type", "count", "L1", "L2"
    ])
    writer.writeheader()
    names = [f"squat_{k:03d}.mp4" for k in range(43)]
    for name in names:
        writer.writerow({"name": name, "type": "squat",
                         "count": 1, "L1": 0, "L2": 20})
    source = csv_buffer.getvalue().encode()
    with tarfile.open(p, "w:gz") as tf:
        _member(tf, "LLSP/annotation/train.csv", source)
        _member(tf, "LLSP/annotation/valid.csv", source)
        for name in names:
            _member(tf, f"LLSP/video/train/{name}", b"r" * 14000)
            _member(tf, f"LLSP/video/valid/{name}", b"v" * 14000)
            _member(tf, f"LLSP/video/test/{name}", b"t" * 14000)
    return p


def _frozen_row():
    original = method(gt=1, tp=0)
    novel = method(gt=1, tp=1)
    conditions = {}
    for name in m47.CONDITIONS:
        # 200ms novel succeeds; 400ms novel has same GT error as original.
        candidate = novel if name == "dense_200ms" else original
        conditions[name] = {"relative_excursion": candidate}
    return {
        "original_m44": {
            "m43": {
                "m41": {
                    "dense_200ms": {"low_to_high": original},
                    "subsample_400ms": {"low_to_high": original}
                }
            }
        },
        "conditions": conditions
    }


def _reference(source, tmp_path):
    _, h = selected_from_tar_archive(
        source, tmp_path / "prefetched_synthetic", split="train", limit=20,
        selection_policy="stable_hash", selection_seed=2026,
        selection_offset=20, require_explicit_split=True
    )
    base = _frozen_row()
    data = {}
    for cond in m47.CONDITIONS:
        clips = [m47._per_video(base, cond) for _ in range(20)]
        methods = {name: _aggregate(clips, name) for name in m47.METHODS}
        data[cond] = {"methods": methods}
    return {
        "protocol": M45_PROTOCOL, "source_split": "train",
        "source_selection": {
            "source_hash_rank_offset": 20, "policy": "stable_hash",
            "seed": 2026, "limit": 20,
            "source_clip_filename_disjoint_from_m40_m45": True,
            "subject_or_view_disjointness_verified": False
        },
        "relative_excursion_rule": EXPECTED_RULE,
        "publisher_train_annotation_sha256": h,
        "annotated_reps": 20, "videos_evaluated": 20,
        "videos_excluded": 0, "conditions": data
    }


def test_m47_pipeline_rechecks_m46_source_metrics_then_persists_only_aggregates(tmp_path, monkeypatch):
    source = _source(tmp_path)
    reference = _reference(source, tmp_path)
    ref_path = tmp_path / "m46-reference.json"
    ref_path.write_text(json.dumps(reference))
    called = []

    def fake_pose(path, annotation, *, sample_ms):
        assert sample_ms == 200
        called.append((path.read_bytes()[:1], annotation["annotated_count"]))
        return {"annotated_count": 1, "frames": [
            {"time_ms": i * 200, "landmarks": None} for i in range(20)
        ]}

    def fake_scoring(inp, out):
        assert json.loads(inp.read_text())["annotated_count"] == 1
        row = _frozen_row()
        out.write_text(json.dumps(row))
        return row

    checked = []
    def validate_fake(row, count):
        checked.append(count)
        assert row == _frozen_row() and count == 1

    # Prior M45 tests validate the full original replay contract; fake
    # comparator rows here deliberately omit nested M43/M44 metadata.
    monkeypatch.setattr(m47, "validate_original_m45", validate_fake)
    report = m47.run(
        archive=source, root=tmp_path / "m47-private-synthetic",
        report_path=tmp_path / "new-report.json",
        reference_path=ref_path, extract_fn=fake_pose,
        compare_fn=fake_scoring, repeats=100, seed=2026
    )
    assert len(called) == len(checked) == 20
    assert all(code == b"r" and gt == 1 for code, gt in called)
    assert report["publisher_train_annotation_sha256"] == (
        reference["publisher_train_annotation_sha256"]
    )
    assert report["frozen_source"]["hash_rank_offset"] == 20
    assert report["frozen_source"]["annotated_gt_reps"] == 20
    assert report["paired_clip_bootstrap"]["dense_200ms"]["event_f1"][
        "percentile_95_interval"
    ] == [1, 1]
    assert report["validation_and_test_untouched"] is True
    output = (tmp_path / "new-report.json").read_text()
    for phrase in (
        "squat_000", '"time_ms"', '"landmarks"', '"frames"',
        '"video_path"', '"per_video_metrics"', '"cycle_bounds_frames"'
    ):
        assert phrase not in output
    assert not list((tmp_path / "m47-private-synthetic").glob("m47-private-*"))
    assert not list((tmp_path / "m47-private-synthetic").rglob("test/*.mp4"))
    assert not list((tmp_path / "m47-private-synthetic").rglob("valid/*.mp4"))
    with pytest.raises(FileExistsError, match="already exists"):
        m47.run(
            archive=source, root=tmp_path / "m47-private-synthetic",
            report_path=tmp_path / "new-report.json",
            reference_path=ref_path
        )


def test_m47_refuses_original_m46_score_or_source_mismatch(tmp_path):
    source = _source(tmp_path)
    original = _reference(source, tmp_path)
    data = _frozen_row()
    observed = {}
    for cond in m47.CONDITIONS:
        clips = [m47._per_video(data, cond) for _ in range(20)]
        observed[cond] = {
            name: _aggregate(clips, name) for name in m47.METHODS
        }
    args = dict(
        annotation_sha=original["publisher_train_annotation_sha256"],
        annotated_reps=20, observed_methods=observed, videos=20
    )
    m47.validate_reference(original, **args)
    modified = json.loads(json.dumps(original))
    modified["conditions"]["dense_200ms"]["methods"]["low_to_high"][
        "event_tp"
    ] += 1
    with pytest.raises(ValueError, match="point estimates differ"):
        m47.validate_reference(modified, **args)
    modified = json.loads(json.dumps(original))
    modified["source_selection"]["source_hash_rank_offset"] = 0
    with pytest.raises(ValueError, match="source, rank"):
        m47.validate_reference(modified, **args)
    modified = json.loads(json.dumps(original))
    modified["publisher_train_annotation_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source, rank"):
        m47.validate_reference(modified, **args)
