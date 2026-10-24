"""Deterministic FakeDetector test double for Call Shield."""

import time
from collections.abc import Sequence

import numpy as np


class FakeDetector:
    """Deterministic test double for speech synthesis detection.

    Heuristic:
      - Computes spectral flatness of the 4.0s window:
          * Multi-frequency / noise-like signals have high spectral flatness -> score > 0 (GENUINE)
          * Pure sinusoidal tones have very low spectral flatness -> score < 0 (SYNTHETIC)
      - Can replay a scripted list of scores.
      - Can simulate inference latency by sleeping for delay_ms.
    """

    EXPECTED_SAMPLES: int = 64000  # 4.0 s at 16 kHz

    def __init__(
        self,
        scripted_scores: Sequence[float] | None = None,
        delay_ms: float = 0.0,
    ) -> None:
        self.scripted_scores: list[float] = list(scripted_scores) if scripted_scores else []
        self.delay_ms: float = max(0.0, float(delay_ms))

    def score(self, x: np.ndarray) -> float:
        """Scores an audio window.

        Args:
            x: 1D float32 numpy array with exactly 64,000 samples.

        Returns:
            float: > 0 for genuine speech stand-in, < 0 for synthetic tone stand-in.

        Raises:
            ValueError: If input shape, length, or dtype is invalid.
        """
        # Contract validation
        if not isinstance(x, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray, got {type(x)}")

        if x.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {x.shape}")

        if len(x) != self.EXPECTED_SAMPLES:
            raise ValueError(
                f"Expected {self.EXPECTED_SAMPLES} samples (4s at 16kHz), got {len(x)}"
            )

        if x.dtype != np.float32:
            raise ValueError(f"Expected float32 dtype, got {x.dtype}")

        # Simulate inference latency
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        # Replay scripted score if available
        if self.scripted_scores:
            return float(self.scripted_scores.pop(0))

        # Spectral flatness heuristic: geometric_mean / arithmetic_mean of power spectrum
        power_spectrum = np.abs(np.fft.rfft(x)) ** 2 + 1e-12
        geom_mean = float(np.exp(np.mean(np.log(power_spectrum))))
        arith_mean = float(np.mean(power_spectrum))
        flatness = geom_mean / arith_mean if arith_mean > 0 else 0.0

        # High flatness (> 0.05) -> Genuine (> 0)
        # Low flatness (<= 0.05) -> Synthetic (< 0)
        raw_score = (flatness - 0.05) * 4.0
        return float(np.clip(raw_score, -2.0, 2.0))
