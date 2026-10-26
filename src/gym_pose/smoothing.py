"""Probability smoothing with correct label alignment and lost-pose reset."""
from __future__ import annotations

from collections import deque

import numpy as np


class ProbabilitySmoother:
    def __init__(self, classes: list[str] | tuple[str, ...], window: int = 15) -> None:
        self.classes = tuple(str(c) for c in classes)
        if window < 1 or len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("window must be positive and classes must be unique")
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        """Call when no usable pose exists, avoiding stale 'correct' labels."""
        self.history.clear()

    def push(self, probabilities: list[float] | np.ndarray) -> tuple[str, float]:
        arr = np.asarray(probabilities, dtype=float)
        if arr.shape != (len(self.classes),) or not np.isfinite(arr).all():
            raise ValueError("Invalid probability vector")
        if (arr < 0).any() or arr.sum() <= 0:
            raise ValueError("Invalid probability distribution")
        arr = arr / arr.sum()
        self.history.append(arr)
        smooth = np.mean(tuple(self.history), axis=0)
        idx = int(np.argmax(smooth))
        # Confidence is of the SMOOTHED class, not the last raw class.
        return self.classes[idx], float(smooth[idx])
