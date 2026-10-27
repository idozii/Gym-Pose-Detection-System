"""Strict six-artifact bootstrap from an owner-owned PRIVATE HF model repository.

Render Free has ephemeral storage and no SSH/disk. The operator explicitly
opts into their trusted six joblib files using a pinned model-repo commit and
a SHA256-pinned manifest. Never fetch source videos, training data or TEST data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .model_registry import DEFAULT_PATHS, KNOWN_TASKS

MANIFEST_NAME = "gym_pose_source_rf_manifest.json"
MANIFEST_VERSION = "gym-pose-six-source-rf-v1"
EXPECTED = tuple(DEFAULT_PATHS[key].as_posix() for key in KNOWN_TASKS)
MAX_MANIFEST_BYTES = 16_384
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as data:
        for block in iter(lambda: data.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def make_manifest(source_root: Path) -> dict:
    """Read exactly six local model filenames without following extra folders."""
    files = []
    for relative in EXPECTED:
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError("One of the six required trusted model files is missing")
        files.append({"path": relative, "sha256": sha256_file(source)})
    return {"format": MANIFEST_VERSION, "files": files}


def validate_manifest(manifest: object) -> dict[str, str]:
    if not isinstance(manifest, dict) or set(manifest) != {"format", "files"}:
        raise ValueError("Invalid model manifest")
    if manifest["format"] != MANIFEST_VERSION or not isinstance(manifest["files"], list):
        raise ValueError("Incompatible model manifest")
    entries = manifest["files"]
    if len(entries) != len(EXPECTED):
        raise ValueError("Expected exactly six source-label models")
    files: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Malformed manifest entry")
        name, digest = item["path"], item["sha256"]
        if not isinstance(name, str) or name not in EXPECTED or name in files:
            raise ValueError("Unknown or duplicate model artifact")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Malformed manifest digest")
        files[name] = digest
    if set(files) != set(EXPECTED):
        raise ValueError("Model manifest does not match six source schemas")
    return files


def bootstrap_models(
    *,
    repo_id: str,
    revision: str,
    manifest_sha256: str,
    token: str,
    output_root: Path,
    downloader: Callable[..., str] | None = None,
) -> tuple[str, ...]:
    """Fetch and verify only owner-pinned assets, before any pickle load.

    The caller explicitly trusts the files via POSE_TRUST_LOCAL_ARTIFACTS=1.
    SHA256 verification prevents accidental/malicious remote drift, but not
    arbitrary execution from an untrusted original pickle.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("HF repo ID must be owner/private-model-repo")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HF revision must be a pinned 40-character commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("Must pin the exact owner-manifest SHA256")
    if not token:
        raise ValueError("Missing private Hugging Face read token")
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download

    def fetch(filename: str) -> Path:
        return Path(downloader(
            repo_id=repo_id, filename=filename, repo_type="model",
            revision=revision, token=token,
        ))

    manifest_path = fetch(MANIFEST_NAME)
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds safe size")
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Pinned manifest digest mismatch")
    files = validate_manifest(json.loads(manifest_path.read_text("utf-8")))

    output_root = Path(output_root)
    # Download and stage all six files BEFORE placing any trusted files in
    # the scoring root; a partially downloaded batch must not be served.
    with tempfile.TemporaryDirectory(prefix="gym-pose-verified-") as tempdir:
        stage_root = Path(tempdir)
        for rel, expected_digest in files.items():
            source = fetch(rel)
            if not source.is_file() or sha256_file(source) != expected_digest:
                raise ValueError("Trusted model digest mismatch")
            target = stage_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for rel in EXPECTED:
            dest = output_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_root / rel, dest)
    return EXPECTED


def main() -> int:
    """Fail closed on invalid trusted files; return immediately if not opted in."""
    repo = os.environ.get("POSE_HF_REPO_ID", "").strip()
    if not repo:
        # Retain existing local owner trust workflow; Render Free must set
        # POSE_HF_REPO_ID and explicitly opt in after secure private upload.
        return 0
    if os.environ.get("POSE_TRUST_LOCAL_ARTIFACTS") != "1":
        return 0
    try:
        bootstrap_models(
            repo_id=repo,
            revision=os.environ.get("POSE_HF_REVISION", ""),
            manifest_sha256=os.environ.get("POSE_HF_MANIFEST_SHA256", ""),
            token=os.environ.get("HF_TOKEN", ""),
            output_root=Path(os.environ.get("POSE_MODEL_ROOT", "/tmp/gym-pose-models")),
        )
    except Exception as error:
        # Hugging Face transport errors may contain confidential request URLs
        # and tokens. Report ONLY the exception's type, never its message.
        raise RuntimeError(
            "Owner-private source model bootstrap failed (check model repo, "
            "40-char revision, manifest SHA256 and secret token): "
            + type(error).__name__
        ) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
