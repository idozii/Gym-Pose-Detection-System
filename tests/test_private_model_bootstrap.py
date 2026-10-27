"""Offline security tests: six fake byte strings, NEVER owner pickle or network."""
import hashlib
import json
from pathlib import Path

import pytest

from gym_pose.private_model_bootstrap import (
    EXPECTED, MANIFEST_NAME, MANIFEST_VERSION,
    bootstrap_models, main, make_manifest, validate_manifest,
)


def source_fixtures(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for i, name in enumerate(EXPECTED):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic-not-pickle-%d" % i).encode("ascii"))
    return source


def test_exactly_six_manifest_entries_and_private_retrieval(tmp_path: Path):
    source = source_fixtures(tmp_path)
    manifest = make_manifest(source)
    assert manifest["format"] == MANIFEST_VERSION
    assert [e["path"] for e in manifest["files"]] == list(EXPECTED)
    payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_file = source / MANIFEST_NAME
    manifest_file.write_bytes(payload)
    urls = []

    def downloader(**kwargs):
        urls.append(kwargs)
        return str(source / kwargs["filename"])

    target = tmp_path / "output"
    result = bootstrap_models(
        repo_id="owner/private-models", revision="a" * 40,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        token="fixture-secret-not-real", output_root=target,
        downloader=downloader,
    )
    assert tuple(result) == EXPECTED
    assert all((source / name).read_bytes() == (target / name).read_bytes() for name in EXPECTED)
    assert {c["filename"] for c in urls} == {MANIFEST_NAME, *EXPECTED}
    assert all(c["token"] == "fixture-secret-not-real" for c in urls)
    assert all(c["revision"] == "a" * 40 for c in urls)
    assert all(c["repo_type"] == "model" for c in urls)


def test_mismatched_pinned_manifest_never_downloads_any_model(tmp_path: Path):
    source = source_fixtures(tmp_path)
    (source / MANIFEST_NAME).write_text(json.dumps(make_manifest(source)))
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs["filename"])
        return str(source / kwargs["filename"])

    with pytest.raises(ValueError, match="digest mismatch"):
        bootstrap_models(
            repo_id="owner/private", revision="a"*40,
            manifest_sha256="0"*64, token="fixture",
            output_root=tmp_path / "result", downloader=downloader,
        )
    assert calls == [MANIFEST_NAME]
    assert not (tmp_path / "result").exists()


def test_bad_model_digest_never_installs_partial_files(tmp_path: Path):
    source = source_fixtures(tmp_path)
    data = (json.dumps(make_manifest(source)) + "\n").encode()
    (source / MANIFEST_NAME).write_bytes(data)
    (source / EXPECTED[-1]).write_bytes(b"tampered model")

    with pytest.raises(ValueError, match="digest mismatch"):
        bootstrap_models(
            repo_id="owner/private", revision="a"*40,
            manifest_sha256=hashlib.sha256(data).hexdigest(),
            token="fixture", output_root=tmp_path / "result",
            downloader=lambda **kw: str(source / kw["filename"]),
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("bad", [
    "x", "owner/repo/third", "https://huggingface.co/owner/repo",
])
def test_disallow_untrusted_repo_urls(tmp_path, bad):
    with pytest.raises(ValueError):
        bootstrap_models(
            repo_id=bad, revision="a"*40,
            manifest_sha256="0"*64, token="fixture",
            output_root=tmp_path,
            downloader=lambda **_: (_ for _ in ()).throw(RuntimeError("network")),
        )


def test_must_pin_full_commit_and_manifest(tmp_path):
    for sha, digest in [("main", "a"*64), ("a"*40, "latest"), ("a"*39, "b"*64)]:
        with pytest.raises(ValueError):
            bootstrap_models(
                repo_id="owner/repo", revision=sha, manifest_sha256=digest,
                token="secret", output_root=tmp_path,
            )


def test_reject_missing_extra_or_duplicate_entries(tmp_path):
    data = make_manifest(source_fixtures(tmp_path))
    for changed in [
        {**data, "files": data["files"][:-1]},
        {**data, "files": [*data["files"][:-1], data["files"][0]]},
        {**data, "files": [*data["files"], {"path": "../private", "sha256": "0"*64}]},
    ]:
        with pytest.raises(ValueError):
            validate_manifest(changed)


def test_untrusted_flags_do_not_download_or_unpickle(monkeypatch):
    monkeypatch.setenv("POSE_HF_REPO_ID", "owner/private")
    monkeypatch.setenv("POSE_TRUST_LOCAL_ARTIFACTS", "0")
    assert main() == 0
    monkeypatch.delenv("POSE_HF_REPO_ID")
    monkeypatch.setenv("POSE_TRUST_LOCAL_ARTIFACTS", "1")
    assert main() == 0
