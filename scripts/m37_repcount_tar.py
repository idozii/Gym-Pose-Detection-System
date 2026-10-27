"""M37 original RepCount-A .tar.gz streaming adapter.

A user already has the author's ~9 GB .tar.gz archive. Streaming TAR
support reads member headers without extracting the entire archive, and
extracts only up to 20 already-annotated Squat videos.

For embedded CSV labels the TAR stream is scanned twice (first pass to
find publisher labels; second to copy selected clips), which can be slow
but does not require 9 GB of extra output disk. With an explicit
--annotations-csv the archive needs only one streaming pass.

No new video labels, unsupported video classes, network calls, or
unverified per-repetition accuracy are created by this adapter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from scripts.m37_repcount_archive import (
    MATCH_SPLITS, MAX_CSV_BYTES, MAX_MEMBER_COUNT,
    MAX_NAME_LENGTH, MAX_SINGLE_VIDEO_BYTES, VIDEO_EXTENSIONS,
    parse_official_csv
)


def _require_tar(archive: Path) -> None:
    if not archive.is_file() or not (
        archive.name.lower().endswith(".tar.gz") or
        archive.name.lower().endswith(".tgz")
    ):
        raise ValueError(
            "Expected a downloaded RepCountA.tar.gz or .tgz file. "
            "Do not rename an archive to .zip."
        )


def _checked_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name
    if not name or len(name) > MAX_NAME_LENGTH or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe TAR member filename in RepCount source")
    # TAR may use safe, relative leading ./ in original author releases.
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe TAR traversal/absolute path in RepCount source")
    if member.issym() or member.islnk():
        raise ValueError("TAR symlink/hardlink is unsupported for safe RepCount extraction")
    if not member.isfile() and not member.isdir():
        raise ValueError("TAR contains unsupported device/special member")
    if (member.isfile() and path.suffix.lower() in VIDEO_EXTENSIONS and
        member.size > MAX_SINGLE_VIDEO_BYTES):
        # Large excluded videos are allowed in the archive; only
        # selected videos will be copied with bounded size.
        pass
    return path


def _split_csv_candidate(path: PurePosixPath, split: str) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    stem = path.stem.lower()
    return any(stem == name or stem.endswith("_" + name)
               for name in MATCH_SPLITS[split])


def _member_matches_split(path: PurePosixPath, split: str) -> bool:
    folder_names = {section.lower() for section in path.parts[:-1]}
    all_named_splits = set(sum((list(x) for x in MATCH_SPLITS.values()), []))
    explicit = folder_names.intersection(all_named_splits)
    return not explicit or bool(explicit.intersection(MATCH_SPLITS[split]))


def _members(archive: Path):
    """Stream members without materializing archive on disk."""
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                yield tar, member, _checked_member(member)
    except tarfile.TarError as error:
        raise ValueError(
            "Invalid/truncated RepCount TAR.GZ. Check 'file' and "
            "'gzip -t' on your downloaded archive."
        ) from error


def inspect_tar_archive(archive: Path, *, split: str = "validation") -> dict:
    _require_tar(archive)
    if split not in MATCH_SPLITS:
        raise ValueError("Use validation, train or test source split")
    csvs, video_examples = [], []
    count, videos, nested, dirs = 0, 0, 0, 0
    for _, info, path in _members(archive):
        count += 1
        if count > MAX_MEMBER_COUNT:
            raise ValueError("RepCount TAR member-count limit exceeded")
        if info.isdir():
            dirs += 1
            continue
        if path.suffix.lower() == ".csv":
            if len(csvs) < 25:
                csvs.append(str(path))
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos += 1
            if len(video_examples) < 5:
                video_examples.append(str(path))
        if path.name.lower().endswith((".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tgz")):
            nested += 1
    return {
        "source": "user-provided local RepCount-A TAR.GZ; origin not independently verified",
        "archive_format": "tar.gz",
        "archive_members": count,
        "directory_members": dirs,
        "video_members": videos,
        "csv_members_visible_first_25": csvs,
        "nested_archives": nested,
        "example_video_member_paths": video_examples,
        "split_requested": split,
        "all_video_extraction_performed": False,
        "video_bytes_extracted": False,
        "full_tar_gz_stream_scanned": True,
        "note": "Listing a compressed 8.9 GB TAR.GZ requires sequential decompression even though no videos are extracted. M37 source ZIP mode is separate."
    }


def _embedded_csv(archive: Path, split: str) -> bytes:
    found = None
    count = 0
    for tar, member, path in _members(archive):
        count += 1
        if count > MAX_MEMBER_COUNT:
            raise ValueError("RepCount TAR member-count limit exceeded")
        if not member.isfile() or not _split_csv_candidate(path, split):
            continue
        if found is not None:
            raise ValueError(
                "Multiple potential original split CSVs in TAR.GZ; "
                "supply --annotations-csv to choose the correct one"
            )
        if member.size > MAX_CSV_BYTES:
            raise ValueError("RepCount original annotation CSV too large")
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError("Source TAR CSV member cannot be read")
        found = stream.read(MAX_CSV_BYTES + 1)
    if found is None:
        raise ValueError(
            "RepCountA.tar.gz has no embedded split annotation CSV. "
            "Supply --annotations-csv pointing to original "
            "TransRAC open_set/new_valid.csv (source file, no manual labeling)."
        )
    return found


def _copy_selected(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    if not 10000 <= member.size <= MAX_SINGLE_VIDEO_BYTES:
        raise ValueError("Selected RepCount TAR video absent or >250 MiB size cap")
    if destination.exists():
        raise FileExistsError(
            "Selected RepCount video already extracted; no implicit overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError("Source TAR video cannot be read")
    length = 0
    try:
        with destination.open("xb") as target:
            while True:
                chunk = stream.read(min(1 << 20, MAX_SINGLE_VIDEO_BYTES - length + 1))
                if not chunk:
                    break
                length += len(chunk)
                if length > MAX_SINGLE_VIDEO_BYTES:
                    raise ValueError("Source video expands beyond safe size cap")
                target.write(chunk)
        if length != member.size:
            raise ValueError("Source TAR video content shorter than archive header")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def selected_from_tar_archive(
    archive: Path, root: Path, *, split: str = "validation",
    limit: int = 3, annotations_csv: Path | None = None,
    require_explicit_split: bool = False,
    selection_policy: str = "archive_order", selection_seed: int = 2026
) -> tuple[list[dict], str]:
    _require_tar(archive)
    if split not in MATCH_SPLITS or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Use a valid RepCount split and 1..20 selected videos")
    if selection_policy not in ("archive_order", "stable_hash"):
        raise ValueError("Unsupported RepCount video selection policy")
    if type(selection_seed) is not int or not 0 <= selection_seed <= 2**32 - 1:
        raise ValueError("Selection seed must be a nonnegative 32-bit integer")
    if selection_policy == "stable_hash" and not require_explicit_split:
        raise ValueError("Stable-hash selection requires an explicit source split")
    if annotations_csv is not None:
        if not annotations_csv.is_file() or annotations_csv.suffix.lower() != ".csv":
            raise ValueError("Expected original authors' annotation CSV")
        if annotations_csv.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("Annotation CSV too large")
        source = annotations_csv.read_bytes()
    else:
        source = _embedded_csv(archive, split)
    labels = {row["video"].lower(): row for row in parse_official_csv(source)}
    # Choose from publisher annotations BEFORE decoding or evaluating any clip.
    # Original archive_order remains untouched for M37/M39 reproducibility.
    eligible_names = None
    if selection_policy == "stable_hash":
        salt = f"repcount-m40-train-v1:{selection_seed}:".encode("ascii")
        ranked = sorted(labels, key=lambda name: (
            hashlib.sha256(salt + name.encode("ascii")).digest(), name
        ))
        eligible_names = set(ranked[:limit])
    extracted = []
    copied = set()
    member_count = 0
    # One streaming pass when an external official annotation file is given.
    # Selection is deterministic source archive order, not a benchmark-
    # performance-chosen subset. The reader stops once limit is reached.
    for tar, member, path in _members(archive):
        member_count += 1
        if member_count > MAX_MEMBER_COUNT:
            raise ValueError("RepCount TAR member-count limit exceeded")
        if not member.isfile() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = path.name.lower()
        if key not in labels or not _member_matches_split(path, split):
            continue
        if eligible_names is not None and key not in eligible_names:
            continue
        # Development-only M39 requires proof from the TAR member path
        # itself that the video belongs to the requested publisher split.
        if require_explicit_split:
            folders = {folder.lower() for folder in path.parts[:-1]}
            known_split_names = set().union(*(
                set(names) for names in MATCH_SPLITS.values()
            ))
            # A path with BOTH train and validation/test segments is not
            # demonstrably TRAIN-only; reject it, never guess its split.
            if (not folders.intersection(MATCH_SPLITS[split]) or
                folders.intersection(known_split_names).difference(
                    MATCH_SPLITS[split]
                )):
                continue
        if key in copied:
            raise ValueError("Ambiguous duplicate selected Squat video name in TAR.GZ")
        record = labels[key]
        target = root / "official_tar_selected" / split / record["video"]
        _copy_selected(tar, member, target)
        extracted.append({**record, "video_path": target})
        copied.add(key)
        if len(extracted) == limit:
            break
    if not extracted:
        raise ValueError(
            "No pre-annotated Squat videos in TAR.GZ matched the "
            "provided publisher CSV for the selected split. "
            "Inspect archive paths/CSV names, do not invent labels."
        )
    return extracted, hashlib.sha256(source).hexdigest()
