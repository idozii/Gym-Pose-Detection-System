"""Local camera utilities. Requires the optional [camera] dependencies."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, pose_to_features
from .smoothing import ProbabilitySmoother
from .training import load_model, LABELS


def _camera_libraries():
    import cv2
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe solutions API missing; install the camera extra in a Python 3.10 environment")
    return cv2, mp


def capture(output_root: str | Path, label: str, count: int = 100, camera_index: int = 0) -> int:
    cv2, _ = _camera_libraries()
    if label not in LABELS or count < 1:
        raise ValueError("Label must be correct/incorrect and count must be positive")
    folder = Path(output_root) / "squat" / label
    folder.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Camera unavailable")
    saved = 0
    try:
        while saved < count:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.putText(frame, f"{label}: press C to save ({saved}/{count}), Q to exit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.imshow("Gym Pose: Capture", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            if key == ord("c"):
                # Avoid overwriting previously captured images.
                import uuid
                target = folder / f"{label}_{uuid.uuid4().hex[:12]}.jpg"
                if cv2.imwrite(str(target), frame):
                    saved += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return saved


def extract(input_root: str | Path, output_csv: str | Path, min_visibility: float = .5) -> dict:
    import pandas as pd
    cv2, mp = _camera_libraries()
    root = Path(input_root) / "squat"
    rows, skipped = [], 0
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for label in LABELS:
            for path in sorted((root / label).glob("*.jpg")):
                frame = cv2.imread(str(path))
                if frame is None:
                    skipped += 1
                    continue
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                features = pose_to_features(result.pose_landmarks.landmark, min_visibility=min_visibility) if result.pose_landmarks else None
                if features is None:
                    skipped += 1
                    continue
                rows.append({**features, "label": label, "source_image": str(path.relative_to(root))})
    if not rows:
        raise RuntimeError("No usable images found")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return {"rows": len(rows), "skipped": skipped, "output": str(output)}


def webcam(model_path: str | Path, camera_index: int = 0, min_visibility: float = .5) -> None:
    import pandas as pd
    cv2, mp = _camera_libraries()
    bundle = load_model(model_path)
    model = bundle["model"]
    smoother = ProbabilitySmoother(tuple(bundle["classes"]))
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Camera unavailable")
    try:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    smooth_landmarks=True, min_detection_confidence=.5,
                                    min_tracking_confidence=.5) as pose:
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                features = pose_to_features(result.pose_landmarks.landmark, min_visibility=min_visibility) if result.pose_landmarks else None
                status = "NO RELIABLE POSE"
                if features is None:
                    smoother.reset()
                else:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    X = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=list(FEATURE_NAMES))
                    label, conf = smoother.push(model.predict_proba(X)[0])
                    status = f"{label.upper()} ({conf:.2f})"
                cv2.rectangle(frame, (0, 0), (500, 65), (20, 20, 20), -1)
                cv2.putText(frame, "SQUAT - " + status, (12, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                cv2.imshow("Gym Pose: Baseline", frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
