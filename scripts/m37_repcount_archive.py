"""M37 original-author RepCount-A ZIP adapter, no new labels or full extraction.

Accepts an archive downloaded legitimately from the RepCount CVPR 2022
authors' OneDrive, not a direct OneDrive page URL. Reads original-style
"name,type,count,L1,L2,..." CSV annotations either inside the archive
or a separately downloaded official CSV. Only selected, pre-annotated
Squat video members are extracted; reject ambiguous source matches.

The adapter cannot independently attest byte-level source provenance,
source-frame synchronization or suitability for technique classification.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

ACTION_ALIASES = frozenset({"squat", "squatting", "squant"})
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_SINGLE_VIDEO_BYTES = 250 * 1024 * 1024
MAX_MEMBER_COUNT = 10000
MAX_NAME_LENGTH = 220
MATCH_SPLITS = {
    "validation": ("validation", "valid", "val"),
    "train": ("train", "training"),
    "test": ("test", "testing"),
}
VIDEO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,120}\.(?:mp4|avi|mov|mkv)$", re.I)


def _safe_member(member: zipfile.ZipInfo) -> PurePosixPath:
    raw = member.filename
    if not raw or len(raw) > MAX_NAME_LENGTH or "\\" in raw or "\x00" in raw:
        raise ValueError("Unsafe RepCount ZIP member path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("RepCount ZIP contains unsafe traversal path")
    if member.file_size > MAX_SINGLE_VIDEO_BYTES and path.suffix.lower() in VIDEO_EXTENSIONS:
        raise ValueError("RepCount ZIP video exceeds 250 MiB per-clip safety limit")
    unix_mode = member.external_attr >> 16
    if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
        raise ValueError("RepCount ZIP symlink members are disallowed")
    if member.flag_bits & 0x1:
        raise ValueError("Encrypted source ZIP member must be legitimately extracted by owner")
    return path


def _archive_index(zf: zipfile.ZipFile):
    records = zf.infolist()
    if len(records) > MAX_MEMBER_COUNT:
        raise ValueError("RepCount ZIP member-count cap exceeded")
    safe = []
    for member in records:
        path = _safe_member(member)
        if not member.is_dir():
            safe.append((member, path))
    return safe


def inspect_archive(archive: Path, *, split: str = "validation") -> dict:
    if split not in MATCH_SPLITS:
        raise ValueError("Use supported RepCount split validation/train/test")
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise ValueError("Point --source-zip to a downloaded RepCountA .zip file, not a OneDrive folder URL")
    with zipfile.ZipFile(archive) as zf:
        index = _archive_index(zf)
    csvs = [str(path) for _, path in index if path.suffix.lower() == ".csv"]
    videos = [str(path) for _, path in index if path.suffix.lower() in VIDEO_EXTENSIONS]
    nested_archives = sum(path.suffix.lower() in {".zip", ".7z", ".rar"}
                          for _, path in index)
    return {
        "source": "user-supplied local RepCount-A ZIP; authenticity not independently verified",
        "zip_members": len(index),
        "video_members": len(videos),
        "csv_members": len(csvs),
        "nested_archives": nested_archives,
        "annotation_csv_members": csvs[:25],
        "example_video_member_paths": videos[:5],
        "split_requested": split,
        "all_video_extraction_performed": False,
        "video_bytes_extracted": False,
        "next_step": (
            "Use --source-zip with existing original CSV labels; "
            "supply --annotations-csv if annotations are outside the ZIP. "
            "Nested ZIPs are not recursively extracted."
        )
    }


def _parse_count(text: str) -> int:
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError("RepCount source annotation has an invalid count") from error
    if not value.is_integer() or not 0 <= value <= 200:
        raise ValueError("RepCount count must be integer between 0 and 200")
    return int(value)


def _parse_bound(text: str) -> int:
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid source cycle boundary frame") from error
    if not value.is_integer() or not 0 <= value <= 10_000_000:
        raise ValueError("Source cycle boundaries must be nonnegative integer frames")
    return int(value)


def parse_official_csv(source: bytes) -> list[dict]:
    if not source or len(source) > MAX_CSV_BYTES:
        raise ValueError("Original annotation CSV missing or exceeds 20 MiB")
    try:
        stream = io.StringIO(source.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise ValueError("Original source CSV requires UTF-8") from error
    reader = csv.DictReader(stream)
    header = reader.fieldnames or []
    if not {"name", "type", "count", "L1", "L2"}.issubset(set(header)):
        raise ValueError("Expected official RepCount-A CSV columns name,type,count,L1,L2,...")
    labels = sorted((key for key in header if re.fullmatch(r"L[1-9]\d*", key)),
                    key=lambda name: int(name[1:]))
    if labels != [f"L{i}" for i in range(1, len(labels) + 1)]:
        raise ValueError("Non-consecutive RepCount original L* annotation column schema")
    output = []
    names = set()
    for row in reader:
        if None in row:
            raise ValueError("Unexpected extra values in publisher annotation CSV")
        if str(row.get("type", "")).strip().lower() not in ACTION_ALIASES:
            continue
        name = str(row.get("name", "")).strip()
        if not VIDEO_NAME_RE.fullmatch(name):
            raise ValueError("Source Squat filename must be an unambiguous video basename")
        if name in names:
            raise ValueError("Duplicate publisher Squat video name in selected annotations")
        names.add(name)
        count = _parse_count(row["count"])
        nonempty = [str(row[key]).strip() for key in labels
                    if row[key] is not None and str(row[key]).strip()]
        # RepCount-A includes valid Squat examples with count=0 and
        # all L* frame boundaries blank. Keep these negative controls:
        # counting false cycles on a zero-rep source matters.
        # Conversely, zero count WITH frame boundaries is malformed.
        if len(nonempty) != 2 * count:
            raise ValueError("Publisher count and ordered L1..L2n cycle endpoints differ")
        bounds = [_parse_bound(value) for value in nonempty]
        last_end = -1
        for i in range(0, len(bounds), 2):
            a, b = bounds[i:i + 2]
            if a >= b or a < last_end:
                raise ValueError("Publisher source cycle frame intervals overlap or reverse")
            last_end = b
        output.append({
            "video": name, "action": "squat",
            "annotated_count": count, "cycle_bounds_frames": bounds
        })
    if not output:
        raise ValueError("Original CSV has no labeled Squat/squant records")
    return sorted(output, key=lambda entry: entry["video"])


def _select_embedded_csv(index, split):
    names = MATCH_SPLITS[split]
    candidates = []
    for info, path in index:
        if path.suffix.lower() != ".csv":
            continue
        stem = path.stem.lower()
        if stem in names or any(stem.endswith("_" + name) for name in names):
            candidates.append(info)
    if len(candidates) > 1:
        raise ValueError("Ambiguous original source split annotation CSVs; use --annotations-csv")
    return candidates[0] if candidates else None


def _locate_video(index, filename, split):
    matches = [(info, path) for info, path in index
               if path.suffix.lower() in VIDEO_EXTENSIONS
               and path.name.lower() == filename.lower()]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    # Identically named videos across multiple splits cannot silently
    # be disambiguated using a user-selected arbitrary archive order.
    stems = MATCH_SPLITS[split]
    selected = []
    for info, path in matches:
        segments = {segment.lower() for segment in path.parts[:-1]}
        if segments.intersection(stems):
            selected.append(info)
    if len(selected) != 1:
        raise ValueError("Duplicate video name across archive paths; split mapping ambiguous")
    return selected[0]


def _copy_bounded(zf: zipfile.ZipFile, item: zipfile.ZipInfo, target: Path):
    if target.exists():
        # A previous run may have used the same selected video.
        # It must never be overwritten or treated as verified source.
        raise FileExistsError("Selected local RepCount video already exists; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with zf.open(item) as src, target.open("xb") as dest:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_SINGLE_VIDEO_BYTES:
                    raise ValueError("RepCount member exceeds decompressed video byte limit")
                dest.write(block)
        if total < 10000:
            raise ValueError("RepCount selected video too small to be valid")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def selected_from_zip(archive: Path, root: Path, *, split: str = "validation",
                      limit: int = 3, annotations_csv: Path | None = None
                      ) -> tuple[list[dict], str]:
    """Safely extract ONLY selected exact-name annotated Squat clips."""
    if split not in MATCH_SPLITS or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Archive split must be validation/train/test; limit 1..20")
    if archive.suffix.lower() != ".zip" or not archive.is_file():
        raise ValueError("M37 --source-zip requires a downloaded .zip file")
    with zipfile.ZipFile(archive) as zf:
        index = _archive_index(zf)
        if annotations_csv is not None:
            if not annotations_csv.is_file() or annotations_csv.suffix.lower() != ".csv":
                raise ValueError("Expected locally saved official RepCount annotation .csv")
            raw_labels = annotations_csv.read_bytes()
        else:
            csv_info = _select_embedded_csv(index, split)
            if csv_info is None:
                raise ValueError(
                    "RepCount ZIP contains no split annotation CSV. "
                    "Use --annotations-csv with the authors' original "
                    "split CSV; no human labeling needed."
                )
            if csv_info.file_size > MAX_CSV_BYTES:
                raise ValueError("Original RepCount annotation CSV exceeds size limit")
            raw_labels = zf.read(csv_info)
        labels = parse_official_csv(raw_labels)
        matched = []
        for entry in labels:
            video_member = _locate_video(index, entry["video"], split)
            if video_member is None:
                continue
            matched.append((entry, video_member))
        if not matched:
            raise ValueError(
                "No publisher-labeled Squat video in source archive matches "
                "source split CSV names. Verify original ZIP layout and split."
            )
        # Do not silently pick a different split or depend on ZIP entry order.
        selected = sorted(matched, key=lambda pair: pair[0]["video"])[:limit]
        root.mkdir(parents=True, exist_ok=True)
        output = []
        for entry, item in selected:
            dest = root / "official_zip_selected" / split / entry["video"]
            _copy_bounded(zf, item, dest)
            output.append({**entry, "video_path": dest})
    return output, hashlib.sha256(raw_labels).hexdigest()
