"""Abstract Detector Protocol interface for Call Shield."""

from typing import Protocol

import numpy as np


class Detector(Protocol):
    """Protocol defining the interface for voice clone detection models."""

    def score(self, x: np.ndarray) -> float:
        """Scores a 4.0-second, 16 kHz, float32, normalized mono audio window.

        Args:
            x: 1D numpy array of exactly 64,000 samples and dtype float32.

        Returns:
            float: > 0 for GENUINE speech, < 0 for SYNTHETIC / cloned speech.
        """
        ...
