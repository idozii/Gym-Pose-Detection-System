"""Manifest-driven data provenance and pose-feature extraction.

No images, biometric identifiers or real subject identities are committed.
Manifest rows explicitly identify their real capture subject/session and camera view.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import pandas as pd

from .features import FEATURE_NAMES, SCHEMA_VERSION, pose_to_features
from .training import LABELS

MANIFEST_COLUMNS = (
    "sample_id", "relative_path", "exercise", "label",
    "subject_id", "session_id", "camera_view",
)
VIEWS = ("front", "side", "diagonal", "other")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def _safe_image(root: Path, relative: str) -> Path:
    """Reject traversal and links that resolve outside the approved image root."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Unsafe image path: {relative!r}")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(p in ("", ".", "..") for p in posix.parts):
        raise ValueError(f"Unsafe image path: {relative!r}")
    if posix.suffix.lower() not in IMAGE_TYPES:
        raise ValueError(f"Unsupported image type: {relative!r}")
    base = root.resolve(strict=True)
    candidate = (base / Path(*posix.parts)).resolve(strict=True)
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"Image outside dataset root: {relative!r}")
    return candidate


def _hash_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_template(destination: str | Path) -> Path:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(output, index=False)
    return output


def read_manifest(manifest_path: str | Path, image_root: str | Path) -> pd.DataFrame:
    """Return validated rows with hashes; fail closed on unknown provenance."""
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if len(manifest) == 0:
        raise ValueError("Manifest is empty: populate with actual consented image metadata")
    manifest = manifest.loc[:, list(MANIFEST_COLUMNS)].copy()
    for field in MANIFEST_COLUMNS:
        if not manifest[field].map(lambda s: bool(s) and s == s.strip()).all():
            raise ValueError(f"Missing or whitespace values in {field}")
    for field in ("sample_id", "subject_id", "session_id"):
        if not manifest[field].map(lambda s: bool(SAFE_ID.fullmatch(s))).all():
            raise ValueError(f"Use pseudonymous alphanumeric IDs in {field}")
    if manifest["sample_id"].duplicated().any() or manifest["relative_path"].duplicated().any():
        raise ValueError("Duplicate sample_id or image path in manifest")
    if not manifest["exercise"].eq("squat").all():
        raise ValueError("Only exercise=squat is supported in Milestone 2")
    if not manifest["label"].isin(LABELS).all():
        raise ValueError("Unrecognized label: only correct/incorrect accepted")
    if not manifest["camera_view"].isin(VIEWS).all():
        raise ValueError(f"Unrecognized camera_view: use {VIEWS}")
    if (manifest.groupby("session_id")["subject_id"].nunique() > 1).any():
        raise ValueError("A session_id may not belong to multiple subjects")
    root = Path(image_root)
    paths = [_safe_image(root, p) for p in manifest["relative_path"]]
    hashes = [_hash_image(p) for p in paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate image bytes across manifest rows (sha256 collision)")
    manifest["image_sha256"] = hashes
    return manifest.sort_values("sample_id").reset_index(drop=True)


def summarize_manifest(manifest_path: str | Path, image_root: str | Path) -> dict:
    records = read_manifest(manifest_path, image_root)
    return {
        "status": "valid", "samples": int(len(records)),
        "subjects": int(records.subject_id.nunique()),
        "sessions": int(records.session_id.nunique()),
        "labels": {str(k): int(v) for k, v in records.label.value_counts().items()},
        "views": {str(k): int(v) for k, v in records.camera_view.value_counts().items()},
        "note": "No model performance measured; images remain on local storage",
    }


def extract_manifest(manifest_path: str | Path, image_root: str | Path,
                     output_csv: str | Path, *, min_visibility: float = .5) -> dict:
    """Build labeled 103-feature table using one auditable manifest per image."""
    import cv2
    from .camera import _camera_libraries
    _, mp = _camera_libraries()

    metadata = read_manifest(manifest_path, image_root)
    root = Path(image_root)
    rows, skipped = [], []
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for row in metadata.to_dict(orient="records"):
            # Paths already checked and hashed; revalidate during extraction.
            path = _safe_image(root, row["relative_path"])
            if _hash_image(path) != row["image_sha256"]:
                raise ValueError("Image mutated after manifest validation: " + row["sample_id"])
            img = cv2.imread(str(path))
            if img is None:
                skipped.append(row["sample_id"])
                continue
            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            features = (pose_to_features(result.pose_landmarks.landmark,
                                         min_visibility=min_visibility)
                        if result.pose_landmarks else None)
            if features is None:
                skipped.append(row["sample_id"])
                continue
            rows.append({**row, **features})
    if not rows:
        raise ValueError("No valid pose observations: inspect camera angle/visibility")
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "total_images": int(len(metadata)), "extracted": len(rows),
        "skipped_no_reliable_pose": len(skipped),
        "skipped_sample_ids": skipped, "output": str(out),
        "warning": "Excluded images can bias class/subject distributions; inspect before evaluation",
    }
