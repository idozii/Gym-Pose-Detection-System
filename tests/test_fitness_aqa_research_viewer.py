"""Fully mocked headless M16A webcam tests; never access a real camera."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose import research_viewer as viewer


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def score_landmarks(self, exercise, target, landmarks, *, min_visibility=.5):
        self.calls.append((exercise, target, landmarks is None, min_visibility))
        if landmarks is None:
            return {
                "status": "abstain", "reason": "missing_or_rejected_pose",
                "error_probability": None, "source_label_flag": None,
            }
        if exercise == "barbell_row" and target == "lumbar":
            return {
                "status": "abstain", "reason": "model_not_loaded_or_unavailable",
                "error_probability": None, "source_label_flag": None,
            }
        return {
            "status": "experimental_source_label_score",
            "error_probability": .72, "threshold": .3,
            "source_label_flag": True, "experimental_only": True,
        }


class FakeCapture:
    def __init__(self, parent):
        self.parent = parent
        self.released = False
        self.position = 0
    def isOpened(self):
        return self.parent.camera_open and not self.released
    def read(self):
        if self.position >= len(self.parent.frames):
            return False, None
        self.position += 1
        return True, np.zeros((360, 640, 3), dtype=np.uint8)
    def release(self):
        self.released = True


class FakeCV:
    COLOR_BGR2RGB = 1
    FONT_HERSHEY_SIMPLEX = 0
    def __init__(self, keys=(), frames=5, *, camera_open=True):
        self.keys = iter(keys)
        self.frames = range(frames)
        self.camera_open = camera_open
        self.capture = None
        self.puttext = []
        self.images_rendered = 0
        self.destroyed = False
    def VideoCapture(self, index):
        self.capture = FakeCapture(self)
        return self.capture
    def cvtColor(self, frame, code):
        assert code == self.COLOR_BGR2RGB
        return frame
    def rectangle(self, *args):
        pass
    def putText(self, frame, text, *args):
        self.puttext.append(text)
        return frame
    def imshow(self, name, frame):
        assert "Experimental" in name
        self.images_rendered += 1
    def waitKey(self, ms):
        assert ms == 1
        return next(self.keys, -1)
    def destroyAllWindows(self):
        self.destroyed = True


class FakePose:
    def __init__(self, source, **kwargs):
        self.source = source
        assert kwargs["static_image_mode"] is False
        assert kwargs["min_detection_confidence"] == .5
        self.frames = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def process(self, frame):
        self.frames += 1
        return SimpleNamespace(
            pose_landmarks=(
                SimpleNamespace(landmark=[object()] * 33)
                if self.frames != 2 else None
            )
        )


class FakeMP:
    def __init__(self):
        self.drawn = 0
        parent = self
        self.solutions = SimpleNamespace(
            pose=SimpleNamespace(
                Pose=lambda **kw: FakePose(parent, **kw),
                POSE_CONNECTIONS=()),
            drawing_utils=SimpleNamespace(
                draw_landmarks=lambda *a: setattr(
                    parent, "drawn", parent.drawn + 1)),
        )


def test_selected_exercises_switch_without_retaining_stale_scores():
    registry = FakeRegistry()
    cv = FakeCV(keys=[ord("2"), ord("3"), ord("1"), ord("q")], frames=4)
    mp = FakeMP()
    output = viewer.run_live_viewer(
        registry, exercise="squat", max_frames=10,
        cv2_api=cv, mp_api=mp,
    )
    assert output["frames_processed"] == 4
    assert output["task_scores_emitted"] == 5
    assert output["task_abstentions"] == 3
    assert output["publisher_test_accessed"] is False
    assert output["alerts_issued"] == 0
    assert output["camera_frames_saved"] == 0
    assert output["last_selected_exercise"] == "squat"
    assert cv.capture.released and cv.destroyed
    assert cv.images_rendered == 4
    assert mp.drawn == 3
    assert [call[:3] for call in registry.calls] == [
        ("squat", "error_knees_forward", False),
        ("squat", "error_knees_inward", False),
        ("ohp", "error_elbows", True),
        ("ohp", "error_knees", True),
        ("barbell_row", "torso_angle", False),
        ("barbell_row", "lumbar", False),
        ("squat", "error_knees_forward", False),
        ("squat", "error_knees_inward", False),
    ]
    assert all(call[-1] == .5 for call in registry.calls)
    assert sum("ABSTAIN" in text for text in cv.puttext) == 3
    assert all("correct" not in text.lower() for text in cv.puttext)
    assert cv.puttext.count(viewer.VIEWER_DISCLAIMER) == 4


def test_single_frame_snapshot_and_default_no_media_save():
    cv = FakeCV(keys=[], frames=9)
    result = viewer.run_live_viewer(
        FakeRegistry(), exercise="ohp", max_frames=1,
        cv2_api=cv, mp_api=FakeMP())
    assert result["frames_processed"] == 1
    assert result["task_scores_emitted"] == 2
    assert cv.images_rendered == 1
    assert cv.capture.released and cv.destroyed


def test_missing_camera_raises_and_releases():
    cv = FakeCV(camera_open=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        viewer.run_live_viewer(
            FakeRegistry(), cv2_api=cv, mp_api=FakeMP())
    assert cv.capture.released
    assert cv.images_rendered == 0


def test_invalid_input_is_rejected_before_camera_open():
    for opts in ({"exercise":"deadlift"},
                 {"camera_index": -1},
                 {"max_frames": 0}):
        cv = FakeCV()
        with pytest.raises(ValueError):
            viewer.run_live_viewer(
                FakeRegistry(), cv2_api=cv, mp_api=FakeMP(), **opts)
        assert cv.capture is None


def test_summary_abstention_and_bad_proba_are_non_alarm_outputs():
    report = {"status":"abstain", "reason":"missing_or_rejected_pose"}
    assert "ABSTAIN" in viewer.summary_line("error_knees", report)
    assert "error" not in viewer.summary_line("lumbar", report).lower()
    for probability in (float("nan"), -1.1, 1.2):
        result = viewer.summary_line("lumbar", {
            "status": "experimental_source_label_score",
            "error_probability": probability, "threshold": .2})
        assert result.startswith("lumbar: ABSTAIN")
    text = viewer.summary_line("lumbar", {
        "status": "experimental_source_label_score",
        "error_probability": .72, "threshold": .3})
    assert "above" in text
    assert "correct" not in text.lower() and "incorrect" not in text.lower()


def test_viewer_fails_closed_on_model_execution_error():
    class BrokenRegistry(FakeRegistry):
        def score_landmarks(self, *args, **kwargs):
            raise ValueError("Broken model schema")
    out = viewer.analyze_landmarks(BrokenRegistry(), "squat", [])
    assert all(r["status"] == "abstain" for r in out.values())
    assert all(r["source_label_flag"] is None for r in out.values())
    with pytest.raises(ValueError, match="Unsupported"):
        viewer.analyze_landmarks(BrokenRegistry(), "deadlift", None)
