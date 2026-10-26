"""M16C camera-framing guide tests: source pose quality, NOT technique.

All inputs fake MediaPipe landmarks and arrays, no real camera or TEST
dataset, no images are written or uploaded.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose import framing_guide as g


def landmarks(low=(), *, invalid=()):
    return [
        SimpleNamespace(
            x=.1, y=.2, z=0.,
            visibility=(.1 if index in low else .99
                        if index not in invalid else float("nan")),
        ) for index in range(33)
    ]


def test_squat_guide_explains_both_ankles_and_right_knee():
    ids = g.required_joint_ids("squat")
    assert ids == (11, 12, 23, 24, 25, 26, 27, 28)
    rows = dict(g.readiness_rows(landmarks(low=(26, 27, 28)), "squat"))
    assert len(rows) == 8
    assert rows["left_ankle"].endswith("LOW")
    assert rows["right_ankle"].endswith("LOW")
    assert rows["right_knee"].endswith("LOW")
    assert rows["left_knee"].endswith("VISIBLE")
    assert all(float(rows[k].split()[0]) >= .5 for k in ("left_shoulder", "right_hip"))


def test_row_torso4_does_not_require_knee_or_ankle_for_framing():
    ids = g.required_joint_ids("barbell_row")
    assert ids == (11, 12, 23, 24)
    info = dict(g.readiness_rows(
        landmarks(low=(25, 26, 27, 28)), "barbell_row"))
    assert len(info) == 4
    assert all(state.endswith("VISIBLE") for state in info.values())


def test_ohp_guide_lists_union_of_both_target_gates():
    ids = g.required_joint_ids("ohp")
    assert len(ids) == 12
    rows = g.readiness_rows(landmarks(low=(15, 28)), "ohp")
    assert dict(rows)["left_wrist"].endswith("LOW")
    assert dict(rows)["right_ankle"].endswith("LOW")
    assert len({name for name, _ in rows}) == len(rows)


def test_missing_invalid_landmarks_not_disguised_as_visible():
    assert all(value == "NO POSE" for _, value in g.readiness_rows(None, "squat"))
    assert all(value == "INVALID" for _, value in g.readiness_rows([], "squat"))
    info = dict(g.readiness_rows(landmarks(invalid=(11,)), "squat"))
    assert info["left_shoulder"] == "INVALID"
    with pytest.raises(ValueError, match="Unknown"):
        g.required_joint_ids("unknown")
    with pytest.raises(ValueError, match="cutoff"):
        g.readiness_rows(landmarks(), "squat", cutoff=float("nan"))


class FakeCV:
    FONT_HERSHEY_SIMPLEX = 0
    def __init__(self):
        self.messages = []
        self.rectangles = []
    def rectangle(self, img, start, end, color, thickness):
        self.rectangles.append((start, end))
    def putText(self, img, text, pos, font, scale, color, thickness):
        self.messages.append((text, pos))


def test_panel_is_to_right_of_unmodified_camera_pixels():
    image = np.full((360, 640, 3), 77, dtype=np.uint8)
    canvas = np.zeros((360, 640 + g.PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :640] = image
    cv = FakeCV()
    g.draw_framing_guide(
        canvas, cv, original_width=640, exercise="squat",
        landmarks=landmarks(low=(25, 26, 27, 28)))
    assert (canvas[:, :640] == image).all()
    assert cv.rectangles == [((640, 0), (640+g.PANEL_WIDTH-1, 359))]
    assert all(position[0] >= 640 for _, position in cv.messages)
    strings = [s for s, _ in cv.messages]
    assert "POSE FRAMING GUIDE" in strings
    assert any("right_ankle:" in s and "LOW" in s for s in strings)
    assert not any("safe" in s.lower() or "correct" in s.lower() for s in strings)


def test_panel_not_allowed_without_dedicated_extra_width():
    cv = FakeCV()
    with pytest.raises(ValueError, match="dedicated"):
        g.draw_framing_guide(np.zeros((360, 640, 3), dtype=np.uint8),
                             cv, original_width=640,
                             exercise="squat", landmarks=landmarks())
