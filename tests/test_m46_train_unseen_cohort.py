"""M46 exact source-level nonoverlap on frozen publisher TRAIN filename hash.

Synthetic archive tests only; no publisher data, validation or TEST inference.
"""
from __future__ import annotations

import csv
import hashlib
import io
import tarfile

import pytest

from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m45_train_relative_motion_benchmark import run


def _add(tar, name, content):
    item = tarfile.TarInfo(name)
    item.size = len(content)
    tar.addfile(item, io.BytesIO(content))


def _archive(tmp_path, n, *, reverse=False, missing=None, contradictory=False):
    base = tmp_path / ("reverse" if reverse else "forward")
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "RepCountA.tar.gz"
    names = [f"squat_{i:03d}.mp4" for i in range(n)]
    annotations = io.StringIO()
    writer = csv.DictWriter(annotations, fieldnames=[
        "name", "type", "count", "L1", "L2"
    ])
    writer.writeheader()
    for name in names:
        writer.writerow({
            "name": name, "type": "squat", "count": 1,
            "L1": 0, "L2": 12
        })
    with tarfile.open(archive, "w:gz") as tf:
        _add(tf, "LLSP/annotation/train.csv",
             annotations.getvalue().encode("utf8"))
        _add(tf, "LLSP/annotation/valid.csv",
             annotations.getvalue().encode("utf8"))
        for name in reversed(names) if reverse else names:
            if name == missing:
                continue
            _add(tf, f"LLSP/video/train/{name}",
                 (name * 900).encode("utf8"))
            # Identical filenames in author validation must never be decoded.
            _add(tf, f"LLSP/video/valid/{name}", b"v" * 13000)
        if contradictory:
            _add(tf, "LLSP/video/train/test/dummy.mp4", b"t" * 14000)
    return archive


def _select(archive, output, *, offset, limit=20):
    return selected_from_tar_archive(
        archive, output, split="train", limit=limit,
        require_explicit_split=True, selection_policy="stable_hash",
        selection_seed=2026, selection_offset=offset
    )


def test_m46_second_hash_twenty_are_fully_disjoint_and_archive_order_independent(tmp_path):
    first = _archive(tmp_path, 47)
    reversed_tar = _archive(tmp_path, 47, reverse=True)
    old_rows, old_hash = _select(first, tmp_path / "dev", offset=0)
    new_rows, new_hash = _select(first, tmp_path / "new", offset=20)
    permuted_rows, reversed_hash = _select(reversed_tar, tmp_path / "permute", offset=20)
    old_names = {item["video"] for item in old_rows}
    new_names = {item["video"] for item in new_rows}
    assert old_hash == new_hash == reversed_hash
    assert len(old_names) == len(new_names) == 20
    assert not old_names.intersection(new_names)
    assert {item["video"] for item in permuted_rows} == new_names
    assert all(item["annotated_count"] == 1 for item in new_rows)
    assert all(item["video_path"].read_bytes().startswith(b"squat")
               for item in new_rows)
    assert not list((tmp_path/"new").rglob("valid/*.mp4"))
    assert not list((tmp_path/"new").rglob("test/*.mp4"))


def test_m46_partial_preselected_replication_cohort_never_refills_from_first_twenty(tmp_path):
    source = _archive(tmp_path, 27)
    old, _ = _select(source, tmp_path/"old", offset=0)
    new, _ = _select(source, tmp_path/"new", offset=20)
    assert len(old) == 20
    assert len(new) == 7
    assert {x["video"] for x in old}.isdisjoint(
        {x["video"] for x in new}
    )


def test_m46_refuses_missing_unseen_train_source_and_silent_replacement(tmp_path):
    only20 = _archive(tmp_path, 20)
    with pytest.raises(ValueError, match="unseen TRAIN cohort unavailable"):
        _select(only20, tmp_path/"no_new", offset=20)
    # If publisher labels promise 20 ranked new clips but their source
    # video member is missing, the loader must refuse a smaller biased
    # selection rather than silently substituting previously used clips.
    names = [f"squat_{i:03d}.mp4" for i in range(41)]
    ranked = sorted(names, key=lambda name: (
        hashlib.sha256(f"repcount-m40-train-v1:2026:{name}".encode()).digest(),
        name
    ))
    source = _archive(tmp_path/"missing", 41, missing=ranked[20])
    with pytest.raises(ValueError, match="missing explicitly TRAIN-labeled"):
        _select(source, tmp_path/"missing_output", offset=20)


def test_m46_does_not_allow_arbitrary_post_hoc_selection_offsets(tmp_path):
    p = _archive(tmp_path, 41)
    for forbidden in (21, 5, -1, True, "20"):
        with pytest.raises(ValueError, match="offset"):
            run(archive=p, root=tmp_path/"private",
                report_path=tmp_path/"bad.json", limit=1,
                cohort_offset=forbidden)
    with pytest.raises(ValueError, match="Nonzero cohort offset"):
        selected_from_tar_archive(
            p, tmp_path/"old_mode", split="train", limit=1,
            selection_policy="archive_order",
            require_explicit_split=True, selection_offset=20
        )
    with pytest.raises(ValueError, match="Nonzero cohort offset"):
        selected_from_tar_archive(
            p, tmp_path/"invalid_split", split="validation", limit=1,
            selection_policy="stable_hash",
            require_explicit_split=True, selection_offset=20
        )
    with pytest.raises(ValueError, match="Selection offset"):
        selected_from_tar_archive(
            p, tmp_path/"negative", split="train", limit=1,
            selection_policy="stable_hash",
            require_explicit_split=True, selection_offset=-1
        )
