import pytest

from gym_pose.smoothing import ProbabilitySmoother


def test_prediction_is_class_aligned():
    smoother = ProbabilitySmoother(["correct", "incorrect"], window=2)
    assert smoother.push([.9, .1])[0] == "correct"
    label, confidence = smoother.push([.2, .8])
    assert label == "correct"
    assert confidence == pytest.approx(.55)


def test_lost_pose_clears_stale_history():
    smoother = ProbabilitySmoother(["correct", "incorrect"], window=15)
    for _ in range(10):
        smoother.push([.99, .01])
    smoother.reset()
    label, confidence = smoother.push([.01, .99])
    assert label == "incorrect"
    assert confidence == pytest.approx(.99)


def test_invalid_probabilities_rejected():
    smoother = ProbabilitySmoother(["correct", "incorrect"])
    with pytest.raises(ValueError):
        smoother.push([float("nan"), .5])
    with pytest.raises(ValueError):
        smoother.push([-.1, 1.1])
