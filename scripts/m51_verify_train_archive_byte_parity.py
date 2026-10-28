"""M51: byte-for-byte archive-to-extracted-video parity on frozen 40 TRAIN clips.

No counter/pose/threshold updates or TEST/validation annotations are read.
Before touching archive, M49 verifies the owner M46/M47/M48 reports and
17 frozen code files. M50 report must affirm original 40-video TRAIN cohort
and publisher TRAIN CSV hash.

The compressed TAR.GZ is streamed twice because publisher TRAIN CSV might
appear after its video entries. The streaming reader necessarily traverses
the archive headers across splits, but ONLY original selected 40 TRAIN
video member contents are hashed; no other clip is extracted or decoded.
This does not verify external dataset provenance, license or identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.m37_repcount_archive import (
    MATCH_SPLITS, MAX_MEMBER_COUNT, MAX_SINGLE_VIDEO_BYTES
)
from scripts.m37_repcount_tar import _embedded_csv, _members
from scripts.m49_pretest_git_provenance import run as verify_m49
from scripts.m50_train_label_video_integrity import (
    _publisher_cohorts, _verify_source_files
)

PROTOCOL = "m51-train-selected-archive-byte-parity-v1"
DEFAULT_DEV = Path("data/external/m45_train_relative_motion")
DEFAULT_REPL = Path("data/external/m46_train_rank21_40")
DEFAULT_M50 = Path("data/processed/m50_train_label_video_integrity.json")
DEFAULT_REPORT = Path("data/processed/m51_train_archive_byte_parity.json")
MIN_MP4_BYTES = 10000
CHUNK_SIZE = 1 << 20


def _hash_stream(stream, *, expected_size: int) -> str:
    if (type(expected_size) is not int or not
        MIN_MP4_BYTES <= expected_size <= MAX_SINGLE_VIDEO_BYTES):
        raise ValueError("M51 selected publisher TRAIN video size is invalid")
    sha = hashlib.sha256()
    size = 0
    while size <= expected_size:
        block = stream.read(min(CHUNK_SIZE, expected_size - size + 1))
        if not block:
            break
        size += len(block)
        if size > expected_size:
            raise ValueError("M51 source video stream exceeds declared bytes")
        sha.update(block)
    if size != expected_size:
        raise ValueError("M51 selected TRAIN video stream is truncated")
    return sha.hexdigest()


def _hash_local(path: Path, *, expected_size: int) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("M51 local selected TRAIN MP4 absent or symlinked")
    if path.stat().st_size != expected_size:
        raise ValueError("M51 local MP4 byte length differs from archive")
    with path.open("rb") as f:
        return _hash_stream(f, expected_size=expected_size)


def _verify_existing_m50(path: Path, *, publisher_sha: str,
                         original_lock_sha: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError("M51 requires original owner's M50 TRAIN integrity JSON")
    report = json.loads(path.read_bytes())
    if not isinstance(report, dict):
        raise ValueError("M51 original M50 report must be a JSON object")
    cohorts = report.get("cohort_video_timing")
    if (
        report.get("protocol") != "m50-squat-train-annotation-video-timing-audit-v1" or
        report.get("source_split") != "train" or
        report.get("publisher_train_annotation_sha256") != publisher_sha or
        report.get("original_m48_lock_sha256") != original_lock_sha or
        report.get("videos_total") != 40 or
        report.get("annotated_gt_reps_total") != 521 or
        report.get("critical_timing_or_decode_integrity_flags") != 0 or
        report.get("validation_or_test_scanned") is not False or
        report.get("detector_or_matcher_modified") is not False or
        not isinstance(cohorts, list) or len(cohorts) != 2 or
        [(x.get("cohort"),x.get("videos"),x.get("annotated_reps"))
         for x in cohorts if isinstance(x, dict)] != [
            ("original_development_20", 20, 312),
            ("frozen_replication_20", 20, 209)
        ]
    ):
        raise ValueError("M51 original M50 TRAIN report/source/GT evidence mismatch")
    return report


def _selected_video_bytes(archive: Path, local: dict[str, Path]) -> dict[str, tuple[str, int]]:
    """Hash only explicitly TRAIN-labeled members from the fixed source cohort.

    Other TAR members must still be traversed to reach selected videos,
    but their payload bytes are not exposed to this function.
    """
    known = set().union(*(set(xs) for xs in MATCH_SPLITS.values()))
    train = set(MATCH_SPLITS["train"])
    selected = {}
    member_count = 0
    for tar, member, path in _members(archive):
        member_count += 1
        if member_count > MAX_MEMBER_COUNT:
            raise ValueError("M51 TAR member count exceeds source reader limit")
        if not member.isfile():
            continue
        name = path.name.lower()
        if name not in local:
            continue
        folders = {component.lower() for component in path.parts[:-1]}
        # Never hash a source video with ambiguous or non-TRAIN split path.
        if (not folders.intersection(train) or
            folders.intersection(known).difference(train)):
            continue
        if path.suffix.lower() != ".mp4":
            raise ValueError("M51 selected publisher source is not MP4")
        if name in selected:
            raise ValueError("M51 duplicate selected publisher TRAIN video member")
        file_stream = tar.extractfile(member)
        if file_stream is None:
            raise ValueError("M51 selected TRAIN member is unreadable")
        with file_stream:
            archive_digest = _hash_stream(
                file_stream, expected_size=member.size
            )
        local_digest = _hash_local(local[name], expected_size=member.size)
        if archive_digest != local_digest:
            raise ValueError(
                "M51 selected local TRAIN video content does not match "
                "original archive source bytes"
            )
        selected[name] = (archive_digest, member.size)
    if set(selected) != set(local):
        raise ValueError("M51 not all forty frozen publisher TRAIN videos were found")
    return selected


def run(*, archive: Path, development_root: Path, replication_root: Path,
        m46_path: Path, m47_path: Path, m48_lock: Path, m50_path: Path,
        report: Path, source_csv_fn=None, member_hash_fn=None,
        freeze_gate=None) -> dict:
    if report.exists():
        raise FileExistsError("M51 report exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M51 requires owner-authorized local RepCountA.tar.gz")
    gate = freeze_gate or verify_m49
    checked = gate(
        m46_path=m46_path, m47_path=m47_path, lock_path=m48_lock,
        report_path=report, check_only=True
    )
    if checked.get("research_source_provenance_check_passed") is not True:
        raise ValueError("M51 needs verified original M49/M48 Git source freeze")
    lock_bytes = m48_lock.read_bytes()
    lock = json.loads(lock_bytes)
    reader = source_csv_fn or _embedded_csv
    raw = reader(archive, "train")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != lock.get("publisher_train_annotation_sha256"):
        raise ValueError("M51 publisher TRAIN CSV SHA256 differs from frozen source")
    _verify_existing_m50(
        m50_path, publisher_sha=sha,
        original_lock_sha=hashlib.sha256(lock_bytes).hexdigest()
    )
    cohorts = _publisher_cohorts(raw)
    if ([sum(v["annotated_count"] for v in cohort) for cohort in cohorts] !=
        [312, 209]):
        raise ValueError("M51 original publisher 312/209 TRAIN GT counts mismatch")
    dev = _verify_source_files(cohorts[0], development_root)
    repl = _verify_source_files(cohorts[1], replication_root)
    local = {}
    for row, source_path in dev + repl:
        name = row["video"].lower()
        if name in local:
            raise ValueError("M51 frozen cohorts unexpectedly overlap by filename")
        local[name] = source_path
    if len(local) != 40:
        raise ValueError("M51 must verify exactly 40 distinct local source clips")
    checker = member_hash_fn or _selected_video_bytes
    source_video_hashes = checker(archive, local)
    if not isinstance(source_video_hashes, dict) or set(source_video_hashes) != set(local):
        raise ValueError("M51 incomplete source archive SHA256 comparison")
    # Store only an ordered digest of *individual video content digests*,
    # never names or per-video hashes. Hashes are checked before recording.
    summaries = []
    for name, cohort in (
        ("original_development_20", cohorts[0]),
        ("frozen_replication_20", cohorts[1])
    ):
        digest = hashlib.sha256()
        bytes_total = 0
        for row in cohort:
            source_name = row["video"].lower()
            sha_video, length = source_video_hashes[source_name]
            if (not isinstance(sha_video, str) or len(sha_video) != 64 or
                any(c not in "0123456789abcdef" for c in sha_video) or
                type(length) is not int or length < MIN_MP4_BYTES):
                raise ValueError("M51 malformed digest/size for frozen source clip")
            digest.update(bytes.fromhex(sha_video))
            bytes_total += length
        summaries.append({
            "cohort": name, "videos_byte_identical_to_source_tar": 20,
            "annotated_gt_reps": sum(row["annotated_count"] for row in cohort),
            "source_video_bytes_total": bytes_total,
            "ordered_sha256_digest_root": digest.hexdigest()
        })
    report_data = {
        "protocol": PROTOCOL,
        "study_type": "selected_existing_train_tar_vs_local_byte_parity",
        "publisher_source_origin_verified_independently": False,
        "source_split_video_contents_hashed": "train_only",
        "tar_archive_headers_across_splits_traversed": True,
        "validation_or_test_video_payload_hashed_or_decoded": False,
        "validation_or_test_annotation_csv_loaded": False,
        "publisher_train_annotation_sha256": sha,
        "original_m48_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "original_m50_report_sha256": hashlib.sha256(
            m50_path.read_bytes()
        ).hexdigest(),
        "frozen_source_rank_groups": "ranks_1_20_and_21_40_seed_2026",
        "cohorts": summaries,
        "videos_byte_identical_total": 40,
        "annotated_gt_reps_total": 521,
        "verification_status": "all_40_train_local_mp4_bytes_equal_tar_members",
        "limits": [
            "Only SHA256 byte parity of original author-provided locally owned TRAIN MP4s and their existing extracted copies is verified.",
            "Origin, license, authenticity, publisher GT timing and participant/session disjointness remain unverified.",
            "TAR.GZ reading traverses member headers across all splits; no validation/TEST payload is extracted, hashed or decoded.",
            "This is not an external accuracy, independent user or TEST study.",
            "No original M34, M39, M45 detector, M37 matcher, frozen M48 17-source set or evidence report is modified.",
            "No per-video filenames, byte checksums, frame content or GT event times are persisted."
        ],
        "research_model_modified": False,
        "test_evaluation_performed": False
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as stream:
        json.dump(report_data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report_data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--replication-root", type=Path, default=DEFAULT_REPL)
    parser.add_argument("--m46-report", type=Path, default=Path(
        "data/processed/m46_train_frozen_replication.json"))
    parser.add_argument("--m47-report", type=Path, default=Path(
        "data/processed/m47_train_replication_bootstrap.json"))
    parser.add_argument("--m48-lock", type=Path, default=Path(
        "data/processed/m48_train_evidence_freeze.json"))
    parser.add_argument("--m50-report", type=Path, default=DEFAULT_M50)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    a = parser.parse_args(argv)
    try:
        data = run(
            archive=a.source_archive, development_root=a.development_root,
            replication_root=a.replication_root,
            m46_path=a.m46_report, m47_path=a.m47_report,
            m48_lock=a.m48_lock, m50_path=a.m50_report,
            report=a.report
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        print("M51 stopped: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol": data["protocol"],
        "videos_byte_identical_total": data["videos_byte_identical_total"],
        "annotated_gt_reps_total": data["annotated_gt_reps_total"],
        "cohorts": data["cohorts"],
        "validation_or_test_video_payload_hashed_or_decoded": False,
        "test_evaluation_performed": False
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
