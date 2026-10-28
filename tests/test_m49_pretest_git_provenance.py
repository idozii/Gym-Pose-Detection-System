"""M49 deterministic synthetic Git-history provenance checks.

The tests create a disposable mini Git repository. They never open
publisher archives, original user media, validation or TEST annotations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import m48_freeze_repcount_evidence as m48
from scripts import m49_pretest_git_provenance as m49


def _git(root, *args):
    output = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True,
        check=False
    )
    assert output.returncode == 0, output.stderr
    return output.stdout.strip()


def _repo(tmp_path):
    root = tmp_path / "git_repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    hashes = {}
    for path in m48.SOURCES:
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        data = ("synthetic frozen research module " + path + "\n").encode("utf8")
        source.write_bytes(data)
        hashes[path] = hashlib.sha256(data).hexdigest()
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "frozen 17 research modules")
    commit = _git(root, "rev-parse", "HEAD")
    lock = {
        "protocol": m48.PROTOCOL,
        "git_head_commit_at_freeze": commit,
        "source_code_sha256": hashes,
        "publisher_train_annotation_sha256": "e" * 64
    }
    return root, lock


def test_m49_verifies_frozen_source_at_original_commit_and_newer_git_head(tmp_path):
    root, lock = _repo(tmp_path)
    first = m49.verify_git_history(lock, repository_root=root)
    assert first["original_m48_git_commit"] == first["current_git_commit"]
    assert first["frozen_source_paths_checked"] == 17
    assert first["source_bytes_match_original_m48_git_commit"] is True
    assert first["local_source_bytes_match_m48_lock"] is True

    # M49 additions (docs/new CLI) can change current HEAD while
    # deliberately leaving original 17 research files byte-identical.
    (root / "EXTRA_README.md").write_text("new notes only")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "add new nonfrozen notes")
    second = m49.verify_git_history(lock, repository_root=root)
    assert second["original_m48_git_commit"] != second["current_git_commit"]
    assert second["source_bytes_match_current_git_commit"] is True
    assert second["non_source_changes_since_m48_lock_allowed"] is True


def test_m49_detects_local_dirty_research_file_even_if_git_head_matches(tmp_path):
    root, lock = _repo(tmp_path)
    path = root / m48.SOURCES[0]
    path.write_bytes(path.read_bytes() + b"dirty")
    with pytest.raises(ValueError, match="local research source differs"):
        m49.verify_git_history(lock, repository_root=root)


def test_m49_detects_committed_change_to_frozen_research_file(tmp_path):
    root, lock = _repo(tmp_path)
    path = root / m48.SOURCES[0]
    path.write_bytes(path.read_bytes() + b"changed")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "unauthorized model source update")
    # Original M48 locked hash disagrees with newer committed working tree.
    with pytest.raises(ValueError, match="local research source differs"):
        m49.verify_git_history(lock, repository_root=root)
    # Even if someone tries to re-label changed working bytes as the
    # original lock, the ORIGINAL Git blob still disagrees and is caught.
    altered_lock = {
        **lock,
        "source_code_sha256": {
            **lock["source_code_sha256"],
            m48.SOURCES[0]: hashlib.sha256(path.read_bytes()).hexdigest()
        }
    }
    with pytest.raises(ValueError, match="differs from Git tracked"):
        m49.verify_git_history(altered_lock, repository_root=root)


def test_m49_detects_missing_original_git_revision_or_wrong_source_set(tmp_path):
    root, lock = _repo(tmp_path)
    broken = dict(lock)
    broken["git_head_commit_at_freeze"] = "a" * 40
    with pytest.raises(ValueError, match="cannot verify frozen source"):
        m49.verify_git_history(broken, repository_root=root)
    broken = dict(lock)
    broken["source_code_sha256"] = {}
    with pytest.raises(ValueError, match="17 source hashes missing"):
        m49.verify_git_history(broken, repository_root=root)
    broken = dict(lock)
    broken["source_code_sha256"] = {
        **lock["source_code_sha256"], m48.SOURCES[0]: "a" * 40
    }
    with pytest.raises(ValueError, match="invalid SHA256"):
        m49.verify_git_history(broken, repository_root=root)


def test_m49_check_only_and_persistent_report_never_authorize_test(tmp_path, monkeypatch):
    root, lock = _repo(tmp_path)
    lock_path = tmp_path / "original_m48.json"
    lock_path.write_text(json.dumps(lock))
    m46_path = tmp_path / "original_m46.json"
    m47_path = tmp_path / "original_m47.json"
    m46_path.write_text("{}")
    m47_path.write_text("{}")

    def fake_verified(*, lock_path, m46_path, m47_path, repository_root):
        assert lock_path.is_file()
        assert m46_path.is_file()
        assert m47_path.is_file()
        assert repository_root == root
        return {
            "verified_original_report_bytes": True,
            "verified_frozen_source_bytes": True
        }

    # M48 independently tests real report verification. This test
    # isolates M49's Git-byte and no-TEST behavior.
    monkeypatch.setattr(m49.m48, "verify_existing_lock", fake_verified)
    output = tmp_path / "m49.json"
    preview = m49.run(
        m46_path=m46_path, m47_path=m47_path, lock_path=lock_path,
        report_path=output, repository_root=root, check_only=True
    )
    assert not output.exists()
    assert preview["source_git_provenance"]["frozen_source_paths_checked"] == 17
    assert preview["external_test_readiness_approved"] is False
    assert preview["official_test_executed_or_opened"] is False
    assert preview["preregistered_future_test_protocol_without_execution"][
        "test_execution_authorized_by_this_tool"
    ] is False

    result = m49.run(
        m46_path=m46_path, m47_path=m47_path, lock_path=lock_path,
        report_path=output, repository_root=root
    )
    assert result == preview
    assert json.loads(output.read_text())["validation_or_test_media_scanned"] is False
    assert "landmarks" not in output.read_text()
    assert "cycle_bounds_frames" not in output.read_text()
    assert '"frames"' not in output.read_text()
    assert '"video_path"' not in output.read_text()
    assert "source_squat.mp4" not in output.read_text()
    with pytest.raises(FileExistsError, match="refuses overwrite"):
        m49.run(
            m46_path=m46_path, m47_path=m47_path, lock_path=lock_path,
            report_path=output, repository_root=root
        )
    # Check-only may revalidate an existing report path; no writes.
    assert m49.run(
        m46_path=m46_path, m47_path=m47_path, lock_path=lock_path,
        report_path=output, repository_root=root, check_only=True
    ) == result


def test_m49_rejects_corrupt_m48_verification(tmp_path, monkeypatch):
    root, lock = _repo(tmp_path)
    lock_path = tmp_path / "m48.json"
    lock_path.write_text(json.dumps(lock))
    for name in ("m46.json", "m47.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        m49.m48, "verify_existing_lock",
        lambda **kwargs: {"verified_original_report_bytes": False,
                          "verified_frozen_source_bytes": True}
    )
    with pytest.raises(ValueError, match="valid original M48"):
        m49.run(
            m46_path=tmp_path/"m46.json",
            m47_path=tmp_path/"m47.json",
            lock_path=lock_path,
            report_path=tmp_path/"out.json",
            repository_root=root
        )
    assert not (tmp_path/"out.json").exists()
