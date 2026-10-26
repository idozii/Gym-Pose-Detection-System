"""Local-only, experimental webcam viewer for original Fitness-AQA labels.

Milestone 16A is an opt-in RESEARCH visualization, NOT a fitness coach:
* All six pre-trained static error-label models are used without fitting.
* The user explicitly selects Squat, Overhead Press, or Barbell Row.
* Scores refer only to the named source annotations, never 'correct form'.
* Pose absence, unreliable joints, and missing models visibly ABSTAIN.
* No alert sounds, movement directions, workout rep counts, webcam recordings,
  image uploads, or publisher TEST evaluation.
* Model data from the original publishers used static-image detection; live
  MediaPipe tracking and individual camera views are distribution shifts.
"""
from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any

import numpy as np

from .model_registry import ModelRegistry
from .live_diagnostics import LiveSessionDiagnostics, source_gate_diagnostic
from .framing_guide import PANEL_WIDTH, draw_framing_guide

TASKS_BY_EXERCISE = {
    "squat": ("error_knees_forward", "error_knees_inward"),
    "ohp": ("error_elbows", "error_knees"),
    "barbell_row": ("torso_angle", "lumbar"),
}
KEY_TO_EXERCISE = {ord("1"): "squat", ord("2"): "ohp", ord("3"): "barbell_row"}
VIEWER_DISCLAIMER = "RESEARCH ONLY | Source-label scores, NOT form/safety advice"
MAX_TARGET_TEXT = 70


def analyze_landmarks(
    registry: ModelRegistry, exercise: str, landmarks: Any,
    *, on_task_timing=None,
) -> dict[str, dict]:
    """Frame-local static analysis; never carry labels across frames/exercises."""
    if exercise not in TASKS_BY_EXERCISE:
        raise ValueError(f"Unsupported viewer exercise: {exercise}")
    output = {}
    for target in TASKS_BY_EXERCISE[exercise]:
        # Time each source-label call separately; a frame's aggregate model
        # stage mixes scored and abstained tasks in OHP's different pose gates.
        started = perf_counter()
        # Registry is the only source of schema, pose admission and model status.
        try:
            result = registry.score_landmarks(
                exercise, target, landmarks, min_visibility=.5)
        except (ValueError, TypeError, RuntimeError, FloatingPointError):
            # Live UI fails closed on an inference failure: no label/score.
            result = {
                "exercise": exercise, "target": target,
                "status": "abstain", "reason": "model_inference_error",
                "error_probability": None, "threshold": None,
                "source_label_flag": None, "experimental_only": True,
            }
        if on_task_timing is not None:
            on_task_timing(
                exercise, target, result["status"],
                (perf_counter() - started) * 1000.0)
        output[target] = result
    return output


def summary_line(target: str, report: dict) -> str:
    """Display only source-label evidence, never 'correct/incorrect' or alerts."""
    if report.get("status") != "experimental_source_label_score":
        reason = str(report.get("reason", "model_unavailable"))
        return f"{target}: ABSTAIN ({reason})"[:MAX_TARGET_TEXT]
    try:
        p, threshold = float(report["error_probability"]), float(report["threshold"])
    except (ValueError, TypeError, KeyError):
        return f"{target}: ABSTAIN (invalid_score)"
    if (not np.isfinite(p) or not np.isfinite(threshold)
        or not 0 <= p <= 1 or not 0 < threshold < 1):
        return f"{target}: ABSTAIN (invalid_score)"
    # Binary score threshold is a TRAIN-calibrated annotation operating
    # point, not evidence of clinically incorrect/safe movement.
    side = "above" if p >= threshold else "below"
    return f"{target}: src P={p:.3f}, train T={threshold:.2f} ({side})"[:MAX_TARGET_TEXT]


def draw_status(frame, cv2_api, exercise: str, reports: dict,
                diagnostic_hint: str = "") -> None:
    """No color-coded error warnings; explanatory text stays onscreen."""
    h = int(frame.shape[0])
    w = int(frame.shape[1])
    if h < 160 or w < 160:
        # Never draw outside the frame in a low-res/invalid webcam feed.
        return
    tasks = TASKS_BY_EXERCISE[exercise]
    cv2_api.rectangle(frame, (0, 0), (w - 1, 154), (22, 22, 22), -1)
    lines = (
        "Exercises: [1] Squat  [2] OHP  [3] Barbell Row  [Q] Exit",
        "Selected: " + exercise,
        summary_line(tasks[0], reports[tasks[0]]),
        summary_line(tasks[1], reports[tasks[1]]),
        VIEWER_DISCLAIMER,
        diagnostic_hint or "No recording | Static models; live-camera distribution shift",
    )
    for idx, line in enumerate(lines):
        cv2_api.putText(
            frame, line, (9, 18 + idx * 24),
            cv2_api.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)


def run_live_viewer(
    registry: ModelRegistry, *,
    exercise: str = "squat", camera_index: int = 0,
    max_frames: int | None = None,
    show_framing_guide: bool = True,
    cv2_api=None, mp_api=None,
) -> dict:
    """Show local webcam analysis only; return aggregate abstention counts.

    Caller must explicitly load its OWN trusted joblib models before calling.
    Nothing is persisted, including raw images, source labels or predictions.
    """
    if exercise not in TASKS_BY_EXERCISE:
        raise ValueError("Choose squat, ohp or barbell_row")
    if not isinstance(camera_index, int) or camera_index < 0:
        raise ValueError("camera_index must be a nonnegative integer")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be positive or omitted")
    if not isinstance(show_framing_guide, bool):
        raise ValueError("show_framing_guide must be a boolean")
    if cv2_api is None or mp_api is None:
        from .camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()

    camera = cv2_api.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Local webcam unavailable or permission denied")
    counts = Counter()
    diagnostics = LiveSessionDiagnostics()
    frames = 0
    current = exercise
    started_at = perf_counter()
    try:
        with mp_api.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=.5,
            min_tracking_confidence=.5,
        ) as pose:
            while camera.isOpened():
                frame_started = perf_counter()
                ok, frame = camera.read()
                if not ok:
                    break
                if frame is None or getattr(frame, "ndim", -1) != 3:
                    raise RuntimeError("Invalid webcam frame shape")
                pose_started = perf_counter()
                prediction = pose.process(
                    cv2_api.cvtColor(frame, cv2_api.COLOR_BGR2RGB))
                pose_ended = perf_counter()
                landmark_holder = getattr(prediction, "pose_landmarks", None)
                landmarks = (
                    getattr(landmark_holder, "landmark", None)
                    if landmark_holder is not None else None
                )
                if landmark_holder is not None:
                    mp_api.solutions.drawing_utils.draw_landmarks(
                        frame, landmark_holder, mp_api.solutions.pose.POSE_CONNECTIONS)
                reports = analyze_landmarks(
                    registry, current, landmarks,
                    on_task_timing=diagnostics.record_task_timing)
                scored_at = perf_counter()
                for value in reports.values():
                    key = (
                        "scored" if value["status"] == "experimental_source_label_score"
                        else "abstained"
                    )
                    counts[key] += 1
                hint = "No recording | Static models; live-camera distribution shift"
                if landmarks is None:
                    hint = "No MediaPipe pose: show your entire body in frame"
                else:
                    gate, bad = source_gate_diagnostic(
                        landmarks, current, TASKS_BY_EXERCISE[current][0])
                    if gate == "required_joint_low_visibility":
                        hint = "Low visibility: " + ", ".join(bad[:3])
                    elif gate != "passes_basic_visibility_geometry_checks":
                        hint = "Pose geometry: " + gate
                draw_status(frame, cv2_api, current, reports, diagnostic_hint=hint)
                # Only the DISPLAY is enlarged: original camera pixels sent
                # to MediaPipe are never cropped/padded/reused for training.
                if show_framing_guide:
                    height, width = frame.shape[:2]
                    display = np.zeros(
                        (height, width + PANEL_WIDTH, 3), dtype=frame.dtype)
                    display[:, :width] = frame
                    draw_framing_guide(
                        display, cv2_api, original_width=width,
                        exercise=current, landmarks=landmarks)
                else:
                    display = frame
                cv2_api.imshow("Gym Pose - Experimental Source Label Viewer", display)
                frames += 1
                key = cv2_api.waitKey(1) & 0xff
                diagnostics.record(
                    current, landmarks, reports,
                    pose_ms=(pose_ended - pose_started) * 1000.,
                    model_ms=(scored_at - pose_ended) * 1000.,
                    frame_ms=(perf_counter() - frame_started) * 1000.,
                )
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in KEY_TO_EXERCISE:
                    current = KEY_TO_EXERCISE[key]
                if max_frames is not None and frames >= max_frames:
                    break
    finally:
        camera.release()
        cv2_api.destroyAllWindows()
    elapsed_seconds = perf_counter() - started_at
    return {
        "status": "local_research_viewer_closed",
        "frames_processed": frames,
        "task_scores_emitted": counts["scored"],
        "task_abstentions": counts["abstained"],
        "publisher_test_accessed": False,
        "camera_frames_saved": 0,
        "alerts_issued": 0,
        "rep_counting_enabled": False,
        "last_selected_exercise": current,
        "on_screen_framing_guide": show_framing_guide,
        "diagnostics": diagnostics.summary(elapsed_seconds=elapsed_seconds),
        "warning": VIEWER_DISCLAIMER,
    }
