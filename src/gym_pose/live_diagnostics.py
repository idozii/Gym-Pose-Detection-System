"""Aggregated, privacy-preserving live pose and abstention diagnostics.

Records ONLY frame counters, abstract joint names, and timing summaries.
Never stores joint coordinates, RGB frames, feature vectors, individual
source-label probabilities or people-identifying information.

This diagnostic is supplementary: it does not replace ModelRegistry's
exact per-model pose-quality gates or make models more permissive.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

import numpy as np

from .features import IMPORTANT_INDICES as SQUAT_INDICES, LANDMARK_COUNT
from .ohp_task_features import JOINTS_BY_TASK as OHP_JOINTS
from .barbell_row_torso4_features import REQUIRED_JOINTS as ROW_JOINTS

REQUIRED_BY_TASK = {
    ("squat", "error_knees_forward"): SQUAT_INDICES,
    ("squat", "error_knees_inward"): SQUAT_INDICES,
    ("ohp", "error_elbows"): OHP_JOINTS["error_elbows"],
    ("ohp", "error_knees"): OHP_JOINTS["error_knees"],
    ("barbell_row", "torso_angle"): ROW_JOINTS,
    ("barbell_row", "lumbar"): ROW_JOINTS,
}
JOINT_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


def source_gate_diagnostic(
    landmarks: Any, exercise: str, target: str, *,
    visibility_cutoff: float = .5,
) -> tuple[str, tuple[str, ...]]:
    """Explain a probable gate rejection; NEVER override actual inference.

    Source Squat 103 features require ALL landmark coordinates finite;
    the OHP/Row source feature functions use only their selected joints.
    Shoulder width is needed in all three source feature schemas.
    """
    key = (exercise, target)
    if key not in REQUIRED_BY_TASK:
        raise ValueError(f"Unknown viewer target: {key}")
    if landmarks is None:
        return "no_pose_detected", ()
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return "invalid_landmark_count", ()
        ids = REQUIRED_BY_TASK[key]
        v = np.asarray([float(landmarks[i].visibility) for i in ids], dtype=float)
        ids_for_geometry = range(LANDMARK_COUNT) if exercise == "squat" else ids
        coords = np.asarray([
            (float(landmarks[i].x), float(landmarks[i].y), float(landmarks[i].z))
            for i in ids_for_geometry
        ], dtype=float)
        shoulders = np.asarray([
            (float(landmarks[i].x), float(landmarks[i].y), float(landmarks[i].z))
            for i in (11, 12)
        ], dtype=float)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return "invalid_landmark_values", ()
    if (not np.isfinite(v).all() or not np.isfinite(coords).all()
        or not np.isfinite(shoulders).all()):
        return "nonfinite_landmark_values", ()
    low = tuple(JOINT_NAMES.get(i, str(i)) for i, visibility
                in zip(ids, v) if visibility < visibility_cutoff)
    if low:
        return "required_joint_low_visibility", low
    if np.linalg.norm(shoulders[0] - shoulders[1]) <= 1e-6:
        return "degenerate_shoulder_width", ()
    return "passes_basic_visibility_geometry_checks", ()


def _time_stats(values) -> dict:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(arr.max()), 3),
    }


class LiveSessionDiagnostics:
    """Frame-level counters and coarse timings; no personal/source data saved."""

    def __init__(self) -> None:
        self.frames = 0
        self.pose_detected_frames = 0
        self.by_exercise: Counter[str] = Counter()
        self.abstention_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        self.low_joint_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.score_counts: Counter[str] = Counter()
        # A bounded rolling window avoids unbounded memory in a long webcam
        # session; overall FPS still covers the complete session.
        self.pose_ms = deque(maxlen=3600)
        self.model_ms = deque(maxlen=3600)
        self.frame_ms = deque(maxlen=3600)
        # Per-target calls are timed independently; OHP can score elbows
        # while knees abstains on the very same camera frame.
        self.scored_task_ms = deque(maxlen=3600)
        self.abstained_task_ms = deque(maxlen=3600)
        self.task_scored_ms = defaultdict(lambda: deque(maxlen=3600))
        self.task_abstained_ms = defaultdict(lambda: deque(maxlen=3600))

    def record_task_timing(
        self, exercise: str, target: str, status: str, elapsed_ms: float
    ) -> None:
        """Keep only elapsed milliseconds and named task, never predictions."""
        if (exercise, target) not in REQUIRED_BY_TASK:
            raise ValueError("Unknown source-label task for timing")
        if not isinstance(elapsed_ms, (float, int, np.floating, np.integer)):
            raise ValueError("Task duration must be numeric")
        if not np.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("Task duration must be finite and nonnegative")
        key = f"{exercise}/{target}"
        if status == "experimental_source_label_score":
            self.scored_task_ms.append(float(elapsed_ms))
            self.task_scored_ms[key].append(float(elapsed_ms))
        elif status == "abstain":
            self.abstained_task_ms.append(float(elapsed_ms))
            self.task_abstained_ms[key].append(float(elapsed_ms))
        else:
            raise ValueError("Unknown source-label prediction status for timing")

    def record(self, exercise: str, landmarks: Any, reports: dict[str, dict],
               *, pose_ms: float, model_ms: float, frame_ms: float) -> None:
        if not all(np.isfinite([pose_ms, model_ms, frame_ms])) or min(
                pose_ms, model_ms, frame_ms) < 0:
            raise ValueError("Frame timings must be finite and nonnegative")
        self.frames += 1
        self.by_exercise[exercise] += 1
        if landmarks is not None:
            self.pose_detected_frames += 1
        self.pose_ms.append(float(pose_ms))
        self.model_ms.append(float(model_ms))
        self.frame_ms.append(float(frame_ms))
        for target, report in reports.items():
            path = f"{exercise}/{target}"
            gate, low = source_gate_diagnostic(landmarks, exercise, target)
            for joint in low:
                self.low_joint_counts[path][joint] += 1
            if report.get("status") == "experimental_source_label_score":
                self.score_counts[path] += 1
                continue
            reason = str(report.get("reason", "unknown_abstention"))
            if gate != "passes_basic_visibility_geometry_checks":
                reason = gate
            elif reason == "missing_or_rejected_pose":
                reason = "feature_gate_rejected_after_basic_checks"
            self.abstention_reasons[path][reason] += 1

    def summary(self, *, elapsed_seconds: float) -> dict:
        admission = {}
        for (exercise, target) in REQUIRED_BY_TASK:
            denominator = int(self.by_exercise[exercise])
            if denominator == 0:
                continue
            path = f"{exercise}/{target}"
            scored = int(self.score_counts[path])
            abstained = sum(self.abstention_reasons[path].values())
            admission[path] = {
                "source_frames_seen": denominator,
                "scores_emitted": scored,
                "abstentions": abstained,
                "score_admission_rate": round(scored / denominator, 4),
            }
        return {
            "frames_processed": self.frames,
            "pose_detected_frames": self.pose_detected_frames,
            "no_pose_frames": self.frames - self.pose_detected_frames,
            "pose_detection_rate": (
                round(self.pose_detected_frames / self.frames, 4)
                if self.frames else None),
            "frames_by_exercise": dict(sorted(self.by_exercise.items())),
            "scored_by_target": dict(sorted(self.score_counts.items())),
            "source_label_score_admission_by_target": dict(sorted(admission.items())),
            "abstentions_by_target_and_reason": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(self.abstention_reasons.items())
            },
            "required_low_visibility_joint_occurrences": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(self.low_joint_counts.items())
            },
            "timing_window_frames": len(self.frame_ms),
            "timed_successful_task_calls_in_window": len(self.scored_task_ms),
            "timed_abstained_task_calls_in_window": len(self.abstained_task_ms),
            "timing_ms": {
                "pose_process": _time_stats(self.pose_ms),
                "model_scoring": _time_stats(self.model_ms),
                "successful_model_task_inference": _time_stats(self.scored_task_ms),
                "abstained_model_task_call": _time_stats(self.abstained_task_ms),
                "by_target_successful_task_inference": {
                    task: _time_stats(samples)
                    for task, samples in sorted(self.task_scored_ms.items())
                },
                "by_target_abstained_task_call": {
                    task: _time_stats(samples)
                    for task, samples in sorted(self.task_abstained_ms.items())
                },
                "full_frame_including_display": _time_stats(self.frame_ms),
            },
            "observed_fps_including_display": (
                round(self.frames / elapsed_seconds, 2)
                if elapsed_seconds > 0 else None),
            "no_frame_images_or_per_frame_data_saved": True,
            "limitations": [
                "Joint visibility stats are frame counts, NOT independent samples",
                "Basic gate diagnosis is supplemental; ModelRegistry remains authoritative",
                "Live camera pose tracking/viewpoint differ from publisher train images",
                "Observed frame rate includes webcam capture, rendering and waitKey",
                "Latency distribution covers only the latest 3600 frames or 3600 task calls (bounded memory)",
                "Successful task timing includes feature extraction and RF predict_proba",
                "Score-admission rate is NOT classification accuracy, precision or recall",
                "Displayed-frame model stage may combine scored and abstained tasks",
                "No model recalibration, diagnostic photo collection or publisher TEST access",
            ],
        }
