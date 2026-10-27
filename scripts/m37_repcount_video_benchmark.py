"""M37 automated *real-video* RepCount-A Squat evaluation (no manual labeling).

- Fetch a tiny deterministic subset of already-annotated public research
  clips from an explicitly identified third-party Hugging Face mirror.
- Decode the locally downloaded MP4s using OpenCV and estimate landmarks
  with MediaPipe Solutions Python; never upload source frames.
- Replay JSON pose traces through the ORIGINAL M34 JavaScript counter
  (Node subprocess) and match counted-event times against published
  original cycle-boundary labels without optimizing thresholds.
- Keep videos/poses local and Git-ignored; report aggregate metrics and
  excluded-video counts. No M20 Gym7 original VAL/TEST touched.

RepCount-A mirrored-source integrity and browser/Python pose-estimator
parity are NOT independently verified. No claim of form correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO = "momo321654/RepCountA-videos"
REPO_TYPE = "dataset"
MIRROR_CARD = "https://huggingface.co/datasets/momo321654/RepCountA-videos"
ORIGINAL = "https://svip-lab.github.io/dataset/RepCount_dataset.html"
PROTOCOL = "m37-repcount-squat-event-v1"
VALID_SPLITS = ("validation", "test", "train")
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]{2,100}\.(?:mp4|mov|avi|mkv)$")
DEFAULT_ROOT = Path("data/external/m37_repcount")
DEFAULT_REPORT = Path("data/processed/m37_repcount_squat_event_results.json")
DEFAULT_LIMIT = 3
MAX_DURATION_S = 120
MAX_VIDEO_BYTES = 250 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mirror_annotations(path: Path) -> list[dict]:
    """Strictly parse mirrored RepCount-A records and original paired bounds.

    No guessing from text-only video-level labels. We require exact
    source per-cycle frame annotation arrays to assess event precision.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("annotations"), list):
        raw = raw["annotations"]
    elif isinstance(raw, dict):
        # Support keyed archive dictionaries only if their entries are
        # explicitly well-formed annotation records.
        raw = list(raw.values())
    if not isinstance(raw, list):
        raise ValueError("Unexpected RepCount-A mirror annotations: must be list")
    records = []
    names = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Invalid RepCount-A source annotation item")
        action = item.get("action_type")
        if action not in ("squat", "squatting"):
            continue
        name = item.get("video") or item.get("source_name")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError("Squat video source name missing or unsafe")
        if name in names:
            raise ValueError("Duplicate source video in RepCount annotations")
        names.add(name)
        count = item.get("count")
        bounds = item.get("cycle_bounds")
        if (type(count) is not int or count <= 0 or count > 200 or
            not isinstance(bounds, list) or len(bounds) != 2 * count or
            not all(type(value) is int and value >= 0 for value in bounds)):
            raise ValueError("Squat source requires exactly count frame-bound pairs")
        last_end = -1
        for i in range(0, len(bounds), 2):
            start, end = bounds[i:i + 2]
            if not start < end or start < last_end:
                raise ValueError("Source cycle frame bounds overlap or are reversed")
            last_end = end
        records.append({
            "video": name, "action": "squat",
            "annotated_count": count,
            "cycle_bounds_frames": list(bounds)
        })
    if not records:
        raise ValueError("No paired, already-annotated Squat videos in mirror JSON")
    return sorted(records, key=lambda entry: entry["video"])


def _lazy_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Please install huggingface_hub in local research venv: "
            "python -m pip install 'huggingface_hub>=0.27,<2'"
        ) from error
    return hf_hub_download




def probe_huggingface_source(*, split: str = "validation",
                            revision: str = "main", api=None) -> dict:
    """Metadata-only check: does the repo AND required exact file exist now?

    Old search engine snippets are not evidence a Hugging Face mirror is
    still public. Never download a video until this check succeeds.
    """
    if split not in VALID_SPLITS:
        raise ValueError("M37 source probe split is not valid")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError(
                "Source verification requires huggingface_hub; install "
                "'huggingface_hub>=0.27,<2'"
            ) from error
        api = HfApi()
    try:
        # Both repo metadata and exact filenames must be available.
        info = api.dataset_info(REPO, revision=revision)
        files = set(api.list_repo_files(
            REPO, repo_type=REPO_TYPE, revision=revision
        ))
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403, 404) or type(error).__name__ in (
            "RepositoryNotFoundError", "GatedRepoError",
            "RevisionNotFoundError", "EntryNotFoundError"
        ):
            raise RuntimeError(
                "The configured third-party RepCount-A Hugging Face "
                "mirror cannot be accessed now (repo, revision, or "
                "permissions unavailable; HTTP status "
                f"{status if status is not None else 'unknown'}). "
                "A previously indexed dataset card does not guarantee "
                "live access. Inspect the official RepCount dataset "
                "distribution; authenticate ONLY if you have legitimate "
                "access to a private/gated repo. Do not substitute "
                "unverified annotations or invent accuracy results."
            ) from error
        raise RuntimeError(
            "Unable to verify the RepCount-A mirror metadata. "
            "Check network/API access; no videos were downloaded."
        ) from error
    annotation = f"{split}_annotations.json"
    videos = [p for p in files if p.startswith(split + "/")
              and p.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    if annotation not in files or not videos:
        raise RuntimeError(
            f"RepCount-A mirror is visible but does not expose the "
            f"expected {annotation} and {split}/*.mp4 source files. "
            "Its layout may have changed; do not treat count-only "
            "metadata as event-time annotations. No video was downloaded."
        )
    return {
        "repo_id": REPO,
        "revision": getattr(info, "sha", revision),
        "annotation_file": annotation,
        "split_video_files_visible": len(videos),
        "annotation_and_video_files_visible": True,
        "no_video_download_performed": True
    }


def fetch_local_squat_cohort(root: Path, source_dir: Path, *,
                             split: str = "validation",
                             limit: int = DEFAULT_LIMIT) -> tuple[list[dict], str]:
    """Use a previously *authorized* RepCount-A mirrored-layout source.

    No labels are entered by hand. The caller supplies an existing directory
    with publisher-compatible JSON and split MP4s. No network calls.
    """
    if split not in VALID_SPLITS or not 1 <= limit <= 20:
        raise ValueError("M37 split must be train/validation/test and limit 1..20")
    root_dir = source_dir.resolve(strict=True)
    if not root_dir.is_dir():
        raise ValueError("M37 local source must be a directory")
    annotation = root_dir / f"{split}_annotations.json"
    if not annotation.is_file():
        raise FileNotFoundError(
            f"RepCount-A local annotation file absent: {annotation}. "
            "Use the officially obtained source's original event labels."
        )
    selected = read_mirror_annotations(annotation)[:limit]
    matched = []
    for item in selected:
        path = root_dir / split / item["video"]
        try:
            path.resolve(strict=True).relative_to(root_dir)
        except (ValueError, OSError) as error:
            raise ValueError("Source video escapes local dataset root") from error
        if not path.is_file() or not 10000 <= path.stat().st_size <= MAX_VIDEO_BYTES:
            raise ValueError(
                "Local RepCount-A video absent, empty or exceeds 250 MiB per clip"
            )
        matched.append({**item, "video_path": path})
    return matched, _sha256(annotation)


def fetch_small_squat_cohort(root: Path, *, split: str = "validation",
                             limit: int = DEFAULT_LIMIT,
                             revision: str = "main",
                             download_fn=None) -> tuple[list[dict], str]:
    if split not in VALID_SPLITS or not 1 <= limit <= 20:
        raise ValueError("M37 split must be train/validation/test and limit 1..20")
    if not isinstance(revision, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,100}", revision
    ):
        raise ValueError("Invalid Hugging Face mirror revision")
    downloader = download_fn or _lazy_hub()
    root.mkdir(parents=True, exist_ok=True)
    annotations = Path(downloader(
        repo_id=REPO, filename=f"{split}_annotations.json",
        repo_type=REPO_TYPE, revision=revision,
        local_dir=str(root)
    ))
    source_hash = _sha256(annotations)
    selected = read_mirror_annotations(annotations)[:limit]
    matched = []
    for record in selected:
        relative = f"{split}/{record['video']}"
        path = Path(downloader(
            repo_id=REPO, filename=relative,
            repo_type=REPO_TYPE, revision=revision,
            local_dir=str(root)
        ))
        if not path.is_file() or path.stat().st_size < 10000 or (
            path.stat().st_size > MAX_VIDEO_BYTES
        ):
            raise ValueError(
                "RepCount source video missing, too small or larger than "
                "per-clip 250MiB local research limit"
            )
        matched.append({**record, "video_path": path})
    return matched, source_hash


def extract_pose_trace(video: Path, annotation: dict, *,
                       max_duration_s: int = MAX_DURATION_S) -> dict:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "M37 video extraction requires the existing "
            ".venv-m20 camera extras (OpenCV + mediapipe 0.10.14)"
        ) from error
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Source video cannot be decoded with OpenCV")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or not (5 <= fps <= 120):
            raise ValueError("Invalid decoded source frames-per-second")
        claimed_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if claimed_frames <= 0 or claimed_frames > fps * max_duration_s + fps:
            raise ValueError("Source video frame count out of frozen 120s limit")
        stride = max(1, round(fps * .4))
        frames = []
        total = 0
        with mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=.5, min_tracking_confidence=.5
        ) as detector:
            while total < claimed_frames + 3:
                success, rgb = cap.read()
                if not success:
                    break
                if total % stride == 0:
                    result = detector.process(
                        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    )
                    pose = result.pose_landmarks
                    points = None if pose is None else [
                        {
                            "x": float(p.x), "y": float(p.y),
                            "z": float(p.z), "visibility": float(p.visibility)
                        } for p in pose.landmark
                    ]
                    if points is not None and (
                        len(points) != 33 or any(
                            not all(math.isfinite(x) for x in point.values())
                            for point in points
                        )
                    ):
                        points = None
                    frames.append({
                        "time_ms": total / fps * 1000,
                        "landmarks": points
                    })
                total += 1
        if not (8 <= len(frames) <= 3000):
            raise ValueError("Insufficient valid chronological source frame samples")
        if max(annotation["cycle_bounds_frames"]) > total:
            raise ValueError(
                "RepCount annotated cycle frame index exceeds actual "
                "decoded source video: annotation/clip alignment mismatch"
            )
        if total < claimed_frames * .9:
            raise ValueError("Decoded video frame count shorter than declared source")
        return {
            "fps": fps, "total_frames": total,
            "action": annotation["action"],
            "annotated_count": annotation["annotated_count"],
            "cycle_bounds_frames": annotation["cycle_bounds_frames"],
            "frames": frames
        }
    finally:
        cap.release()


def _node_replay(trace_path: Path, output_path: Path,
                 *, script: Path = Path("web/scripts/m37_eval_repcount_trace.js")) -> dict:
    try:
        result = subprocess.run(
            ["node", str(script), str(trace_path), str(output_path)],
            check=False, capture_output=True, text=True, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M37 requires Node 22 and accessible local replay script") from error
    if result.returncode:
        raise RuntimeError("M34 JS counter replay failed: " + result.stderr[:500])
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected M37 M34 JavaScript evaluator output")
    return report


def summarise(results: list[dict], exclusions: Counter,
              *, split: str, revision: str, source_hash: str,
              candidates: int) -> dict:
    if not results:
        raise ValueError(
            "No RepCount Squat video was successfully decoded and evaluated; "
            f"excluded={dict(exclusions)}. No accuracy report created"
        )
    n = len(results)
    tp = sum(r["event_tp"] for r in results)
    fp = sum(r["event_fp"] for r in results)
    fn = sum(r["event_fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {
        "study": "M37 pre-registered heuristic M34 Squat real video evaluation",
        "evaluation": PROTOCOL,
        "source": REPO,
        "upstream_official_dataset": ORIGINAL,
        "mirror_card": MIRROR_CARD,
        "source_split": split,
        "mirror_revision_requested": revision,
        "annotation_sha256": source_hash,
        "squat_candidate_videos": candidates,
        "squat_evaluated_videos": n,
        "excluded_videos": candidates - n,
        "exclusion_reasons": dict(exclusions),
        "source_total_annotated_reps": sum(r["annotated_rep_count"] for r in results),
        "m34_total_observed_cycles": sum(r["website_observed_cycles"] for r in results),
        "session_mean_absolute_count_error": sum(
            r["absolute_count_error"] for r in results
        ) / n,
        "session_exact_count_fraction": sum(
            r["absolute_count_error"] == 0 for r in results
        ) / n,
        "session_within_one_fraction": sum(
            r["absolute_count_error"] <= 1 for r in results
        ) / n,
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "event_precision": precision, "event_recall": recall,
        "event_f1": 2 * precision * recall / (precision + recall)
                    if precision + recall else 0.,
        "per_video_pose_sample_coverage_mean": sum(
            r["pose_sample_coverage"] for r in results
        ) / n,
        "event_matching_tolerance": (
            "max(750ms, 35% ground-truth duration), cap 2500ms"
        ),
        "real_video_measured": True,
        "media_pipe_python_solutions_used": True,
        "browser_python_pose_estimator_parity_validated": False,
        "source_clip_annotation_sync_independently_verified": False,
        "third_party_mirror_provenance_independently_verified": False,
        "technique_quality_or_injury_safety_evaluated": False,
        "original_m20_val_or_test_used": False,
        "m34_algorithm_or_thresholds_modified": False,
        "scope": (
            "Only a selected, already-annotated third-party Squat subset, "
            "not original OHP/Barbell Row; no video-level manual labeling. "
            "Skipped/non-decodable videos explicitly excluded from denominator. "
            "Python Solutions and browser Tasks estimator parity unvalidated. "
            "Original authors' video frame synchronization is assumed only "
            "within decoded bounds; source files must be independently checked."
        )
    }


def run(*, split: str, limit: int, root: Path,
        report_path: Path, revision: str = "main",
        download_fn=None, trace_fn=None, replay_fn=None,
        local_source: Path | None = None) -> dict:
    if report_path.exists():
        raise FileExistsError("M37 output report exists; refusing overwrite")
    if not 1 <= limit <= 20:
        raise ValueError("Only 1..20 clips allowed for first M37 pilot")
    if local_source is not None:
        selected, source_hash = fetch_local_squat_cohort(
            root, local_source, split=split, limit=limit
        )
    else:
        if download_fn is None:
            probe_huggingface_source(split=split, revision=revision)
        selected, source_hash = fetch_small_squat_cohort(
            root, split=split, limit=limit, revision=revision,
            download_fn=download_fn
        )
    extractor = trace_fn or extract_pose_trace
    replay = replay_fn or _node_replay
    exclusions = Counter()
    results = []
    # Pose traces can contain sensitive motion data. Keep them in
    # per-run ephemeral private files; retain only aggregate results.
    with tempfile.TemporaryDirectory(prefix="m37-traces-", dir=root) as temp:
        trace_root = Path(temp)
        for entry in selected:
            name = Path(entry["video"]).stem
            trace_file = trace_root / f"{name}.json"
            result_file = trace_root / f"{name}.m34_eval.json"
            try:
                trace = extractor(entry["video_path"], entry)
                with trace_file.open("x", encoding="utf-8") as handle:
                    json.dump(trace, handle, allow_nan=False)
                output = replay(trace_file, result_file)
                if output["annotated_rep_count"] != entry["annotated_count"]:
                    raise ValueError("Ground-truth repeat count mismatch")
                results.append(output)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                # A failed clip cannot silently be counted as a model win.
                code = type(error).__name__
                exclusions[code] += 1
    report = summarise(
        results, exclusions, split=split, revision=revision,
        source_hash=source_hash, candidates=len(selected)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=VALID_SPLITS, default="validation")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--check-source", action="store_true",
                        help="Verify the HF repository and exact annotation/video paths; download nothing")
    parser.add_argument("--local-source", type=Path,
                        help="Existing legally obtained RepCount-A mirrored-layout files, no Hugging Face API")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        if args.check_source:
            if args.local_source is not None:
                raise ValueError("Choose only one of --check-source or --local-source")
            result = probe_huggingface_source(
                split=args.split, revision=args.revision
            )
        else:
            result = run(
                split=args.split, limit=args.max_videos,
                root=args.root, report_path=args.report,
                revision=args.revision, local_source=args.local_source
            )
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"M37 stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
