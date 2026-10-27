"""M20 Gym7 video-level research features. No source video or model is committed.

A clip's publisher good/bad label is NOT a per-frame error annotation.
A single real subject must remain in exactly ONE data split across views.
"""
from __future__ import annotations
import csv
import hashlib
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import numpy as np

EXERCISES = ("pushup", "bicep_curl")
VIEWS = ("front", "side", "diagonal")
LABELS = ("good", "bad")
META = ("video_id", "archive_member", "subject_id", "exercise", "label", "view")
FEATURES = tuple(f"f{i:03d}" for i in range(264))
CSV_FIELDS = (*META, *FEATURES)
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}
SAFE_SUBJECT = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def infer_metadata(member: str) -> dict | None:
    """Conservative hints only. Ambiguous labels stay blank for owner review."""
    path = PurePosixPath(member)
    if path.suffix.lower() not in VIDEO_TYPES:
        return None
    words = re.findall(r"[a-z0-9]+", member.lower())
    text = member.lower()
    exercise = ("pushup" if re.search(r"push[ _-]?ups?\b", text)
                else "bicep_curl" if re.search(r"bicep[s]?[ _-]?curl", text)
                else "")
    if not exercise:
        return None
    ls = [w for w in LABELS if w in words]
    vs = [w for w in VIEWS if w in words]
    label = ls[0] if len(ls) == 1 else ""
    view = vs[0] if len(vs) == 1 else ""
    subject = re.search(r"(?:^|[/_\- ])(?:subject|participant|person|sub|p|s)[_\- ]?(\d{1,3})(?=$|[/_.\- ])", text)
    subject_id = f"subject_{int(subject.group(1)):03d}" if subject else ""
    video_id = hashlib.sha256(member.encode("utf-8")).hexdigest()[:20]
    return {"video_id": video_id, "archive_member": member,
            "subject_id": subject_id, "exercise": exercise,
            "label": label, "view": view}


def scan_zip(archive: Path, manifest: Path) -> dict:
    """Create editable hints manifest, one row per eligible original video."""
    if manifest.exists():
        raise FileExistsError("Refusing to overwrite existing manually verified manifest")
    rows = []
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir(): continue
            result = infer_metadata(item.filename)
            if result:
                rows.append(result)
    if not rows:
        raise ValueError("No Push-up/Curl video paths found; inspect publisher ZIP naming")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video paths in publisher archive")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x:x["archive_member"]))
    missing = sum(not all(r[c] for c in META) for r in rows)
    return {"manifest": str(manifest), "videos": len(rows),
            "metadata_rows_needing_review": missing,
            "warning": "Verify EVERY subject, label, and view before extraction. Empty fields are never guessed."}


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty manifest")
    seen_ids, seen_paths = set(), set()
    for r in rows:
        if set(r) != set(META):
            raise ValueError("Manifest columns must match M20 video schema exactly")
        if not all(r.get(k) and r[k] == r[k].strip() for k in META):
            raise ValueError("Missing or invalid subject/label/view - review scan CSV first")
        if not SAFE_SUBJECT.fullmatch(r["subject_id"]) or "/" in r["subject_id"]:
            raise ValueError("Use stable pseudonymous subject IDs")
        if r["exercise"] not in EXERCISES or r["label"] not in LABELS or r["view"] not in VIEWS:
            raise ValueError("Unrecognized exercise, label or camera view")
        name = PurePosixPath(r["archive_member"])
        if (name.is_absolute() or ".." in name.parts or
            name.suffix.lower() not in VIDEO_TYPES or "\\" in r["archive_member"]):
            raise ValueError("Unsafe archive video member path")
        if r["video_id"] in seen_ids or r["archive_member"] in seen_paths:
            raise ValueError("Duplicate video/path may leak between splits")
        seen_ids.add(r["video_id"]);seen_paths.add(r["archive_member"])
    return rows


def frame_pose_vector(landmarks, exercise: str, min_visibility: float = .35):
    """132-frame feature schema: 99 normalized XYZ + 33 visibility values.

    Accept a visible SIDE of the body, supporting the publisher's side view.
    Do not mirror data here: front-camera mirror is presentation-only.
    """
    if exercise not in EXERCISES or not 0 <= min_visibility <= 1:
        raise ValueError("Invalid exercise or visibility threshold")
    if landmarks is None or len(landmarks) != 33:
        return None
    points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
    vis = np.array([p.visibility for p in landmarks], dtype=float)
    if not np.isfinite(points).all() or not np.isfinite(vis).all() or np.any((vis < 0)|(vis > 1)):
        return None
    sides = ([11,13,15,23,25,27],[12,14,16,24,26,28]) if exercise == "pushup" else (
        [11,13,15,23], [12,14,16,24]
    )
    if max(min(vis[s]) for s in sides) < min_visibility:
        return None
    hip_center = (points[23] + points[24]) / 2
    shoulder_center = (points[11] + points[12]) / 2
    normalizer = float(np.linalg.norm(shoulder_center[:2]-hip_center[:2]))
    if normalizer < .03:
        return None
    values = np.concatenate([((points - hip_center) / normalizer).reshape(-1), vis])
    if values.shape != (132,) or not np.isfinite(values).all():
        return None
    return values


def clip_aggregate(frames: list[np.ndarray]):
    """Clip statistics, never pseudo-independent per-frame good/bad examples."""
    if len(frames) < 4:
        return None
    data = np.stack(frames)
    result = np.concatenate([np.median(data, axis=0),np.std(data,axis=0)])
    if result.shape != (264,) or not np.isfinite(result).all():
        return None
    return result


def extract_zip(archive: Path, manifest: Path, destination: Path, *,
                sample_fps:float=2., max_frames:int=240, max_video_mb:int=500,
                limit:int=0)->dict:
    if not 0 < sample_fps <= 10 or not 4 <= max_frames <= 2000 or max_video_mb < 5 or limit < 0:
        raise ValueError("Invalid extraction sampling or safety limit")
    if destination.exists():
        raise FileExistsError("Feature CSV exists; refusing overwrite of research split data")
    rows=read_manifest(manifest)
    if limit: rows=rows[:limit]
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Install optional camera dependencies: pip install -e '.[camera]'") from error
    destination.parent.mkdir(parents=True,exist_ok=True)
    accepted, dropped, result_rows=[], [], []
    with zipfile.ZipFile(archive) as zipped:
        archive_members={x.filename:x for x in zipped.infolist()}
        for row in rows:
            name=row["archive_member"]
            entry=archive_members.get(name)
            if entry is None or entry.is_dir() or entry.file_size > max_video_mb*1024*1024:
                dropped.append({"video_id":row["video_id"],"reason":"missing_or_oversize"});continue
            frame_values=[]
            with tempfile.TemporaryDirectory(prefix="gym7_one_video_") as tmp:
                path=Path(tmp)/("clip"+PurePosixPath(name).suffix.lower())
                with zipped.open(entry) as source,path.open("wb") as target:
                    shutil.copyfileobj(source,target,1024*1024)
                reader=cv2.VideoCapture(str(path))
                try:
                    fps=float(reader.get(cv2.CAP_PROP_FPS))
                    if not reader.isOpened() or not math.isfinite(fps) or fps<=0:
                        dropped.append({"video_id":row["video_id"],"reason":"unreadable_clip"});continue
                    stride=max(1,int(round(fps/sample_fps)))
                    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
                        index=0
                        sampled=0
                        while sampled<max_frames:
                            ok,frame=reader.read()
                            if not ok:break
                            index+=1
                            if (index-1)%stride:continue
                            sampled+=1
                            result=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                            vals=frame_pose_vector(
                                result.pose_landmarks.landmark if result.pose_landmarks else None,
                                row["exercise"]
                            )
                            if vals is not None:
                                frame_values.append(vals)
                finally:
                    reader.release()
            aggregation=clip_aggregate(frame_values)
            if aggregation is None:
                dropped.append({"video_id":row["video_id"],"reason":"insufficient_visible_pose"});continue
            result_rows.append({**row,**dict(zip(FEATURES,map(float,aggregation)))})
            accepted.append(row["video_id"])
    if not result_rows:
        raise ValueError("No eligible clips with 4+ visible frames; review extraction coverage")
    with destination.open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        writer.writeheader();writer.writerows(result_rows)
    report={"source":"Mendeley Gym7 v3","doi":"10.17632/kgbb3yn47p.3",
      "license":"CC BY 4.0","feature_schema":"gym7-clip-264-v1",
      "label_semantics":"publisher good/bad VIDEO-level only, NOT named form errors",
      "selected":len(rows),"accepted":len(accepted),"rejected":len(dropped),
      "rejections":dropped,"output":str(destination),
      "warning":"Rejected pose clips may bias quality estimates; review by label and viewpoint"}
    return report
