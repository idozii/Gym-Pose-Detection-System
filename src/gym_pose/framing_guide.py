"""On-screen framing guide for the six experimental source-error tasks.

The guide visualizes ONLY per-joint MediaPipe visibility on the LOCAL
webcam display. It is NOT an exercise error assessment. It never logs,
uploads or persists frames, landmarks, joint coordinates or confidences.
ModelRegistry is still solely responsible for pose admission/inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .live_diagnostics import JOINT_NAMES, REQUIRED_BY_TASK

PANEL_WIDTH = 296
MIN_VISIBILITY = .5


def required_joint_ids(exercise: str) -> tuple[int, ...]:
    """Unique required joints across both selected exercise detectors."""
    keys = sorted(key for key in REQUIRED_BY_TASK if key[0] == exercise)
    if not keys:
        raise ValueError(f"Unknown viewer exercise: {exercise}")
    return tuple(sorted(set().union(*(REQUIRED_BY_TASK[key] for key in keys))))


def readiness_rows(
    landmarks: Any, exercise: str, *, cutoff: float = MIN_VISIBILITY
) -> list[tuple[str, str]]:
    """Return transient (anatomical joint, visibility description) rows."""
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("visibility cutoff must be finite and between 0 and 1")
    ids = required_joint_ids(exercise)
    output = []
    for i in ids:
        name = JOINT_NAMES.get(i, f"landmark_{i}")
        if landmarks is None:
            output.append((name, "NO POSE"))
            continue
        try:
            if len(landmarks) != 33:
                output.append((name, "INVALID"))
                continue
            v = float(landmarks[i].visibility)
        except (IndexError, AttributeError, TypeError, ValueError, OverflowError):
            output.append((name, "INVALID"))
            continue
        if not np.isfinite(v) or not 0 <= v <= 1:
            state = "INVALID"
        else:
            state = f"{v:.2f} " + ("VISIBLE" if v >= cutoff else "LOW")
        output.append((name, state))
    return output


def draw_framing_guide(
    display, cv2_api, *, original_width: int, exercise: str,
    landmarks: Any,
) -> None:
    """Paint a right-hand side panel, not on top of user's ankles/body.

    The source camera frame must remain unmodified for MediaPipe/model
    processing. Panels are created only for display AFTER inference.
    """
    h, width = display.shape[:2]
    if original_width < 1 or width - original_width < PANEL_WIDTH:
        raise ValueError("Display canvas lacks the dedicated guide side panel")
    x = original_width
    cv2_api.rectangle(display, (x, 0), (width - 1, h - 1),
                      (24, 24, 24), -1)
    lines = [
        "POSE FRAMING GUIDE",
        exercise.upper().replace("_", " "),
        f"Required joint visibility >= {MIN_VISIBILITY:.2f}",
    ]
    rows = readiness_rows(landmarks, exercise)
    spacing = min(21, max(12, (h - 111) // max(1, len(rows))))
    for idx, line in enumerate(lines):
        cv2_api.putText(display, line[:34], (x + 10, 19 + idx * 22),
                        cv2_api.FONT_HERSHEY_SIMPLEX, .47,
                        (230, 230, 230), 1)
    for idx, (name, state) in enumerate(rows):
        # Cues are visibility status, never 'correct form' traffic lights.
        cv2_api.putText(display, f"{name}: {state}"[:36],
                        (x + 10, 83 + idx * spacing),
                        cv2_api.FONT_HERSHEY_SIMPLEX, .43,
                        (200, 200, 200), 1)
    if not rows:
        raise AssertionError("Each exercise needs source-required joints")
