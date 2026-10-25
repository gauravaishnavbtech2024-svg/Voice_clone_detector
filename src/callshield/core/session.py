"""SessionEngine coordinating real-time audio buffering, scoring, and state emission."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from callshield.core.audio import (
    crop_or_pad,
    is_silent,
    resample_8k_to_16k,
    rms_normalize,
)
from callshield.core.buffer import RollingBuffer
from callshield.core.decider import AlertDecider


class Detector(Protocol):
    """Protocol for synthetic voice detectors."""

    def score(self, x: np.ndarray) -> float:
        """Returns positive float for genuine audio, negative float for synthetic audio."""
        ...


@dataclass(frozen=True)
class SessionState:
    """State emitted for each evaluated time window."""

    t: float
    score: float
    level: str
    flagged: int
    reason: str


class SessionEngine:
    """Orchestrates streaming audio ingestion, windowing, and progressive alert decisions.

    Specifications:
      - Internal evaluation sampling rate: 16,000 Hz
      - Window size: 4.0 s (64,000 samples), repeat-padded if shorter, RMS-normalized to -26 dBFS
      - Hop size: 1.0 s (16,000 samples)
      - First score after >= 2.0 s (32,000 samples) of buffered audio
      - Ingestion: 16 kHz or 8 kHz (automatically resampled to 16 kHz)
      - Silent windows (RMS < -50 dBFS) bypass detector scoring and return level OK
    """

    WINDOW_SAMPLES = 64000  # 4.0 s at 16 kHz
    HOP_SAMPLES = 16000  # 1.0 s at 16 kHz
    MIN_BUFFER_SAMPLES = 32000  # 2.0 s at 16 kHz

    def __init__(
        self,
        detector: Any,
        decider: AlertDecider,
        sample_rate_in: int = 16000,
        silence_threshold_dbfs: float = -42.0,
    ) -> None:
        self.detector = detector
        self.decider = decider
        self.sample_rate_in = sample_rate_in
        self.silence_threshold_dbfs = float(silence_threshold_dbfs)

        self.buffer = RollingBuffer(capacity_samples=self.WINDOW_SAMPLES)
        self.total_samples_16k: int = 0
        self.next_eval_samples: int = self.MIN_BUFFER_SAMPLES

    def feed(self, samples: np.ndarray) -> list[SessionState]:
        """Ingests incoming audio samples and returns any newly triggered evaluation states."""
        samples_f32 = np.asarray(samples, dtype=np.float32)
        if len(samples_f32) == 0:
            return []

        # Convert to 16 kHz if telephony 8 kHz input
        if self.sample_rate_in == 8000:
            samples_16k = resample_8k_to_16k(samples_f32)
        else:
            samples_16k = samples_f32

        chunk_size = 1600  # 100 ms at 16 kHz
        emitted_states: list[SessionState] = []

        for i in range(0, len(samples_16k), chunk_size):
            chunk = samples_16k[i : i + chunk_size]
            self.buffer.append(chunk)
            self.total_samples_16k += len(chunk)

            # Check if one or more evaluation horizons have been reached
            while self.total_samples_16k >= self.next_eval_samples:
                eval_time_s = round(self.next_eval_samples / 16000.0, 3)
                self.next_eval_samples += self.HOP_SAMPLES

                # Extract latest buffered audio (up to 4.0s)
                window = self.buffer.latest(self.WINDOW_SAMPLES)

                # Silence gate: bypass detector if window is silent or has no speech peak
                peak = float(np.max(np.abs(window))) if len(window) > 0 else 0.0
                if is_silent(window, threshold_dbfs=self.silence_threshold_dbfs) or peak < 0.012:
                    current_level = "ALERT" if self.decider.latched else "OK"
                    emitted_states.append(
                        SessionState(
                            t=eval_time_s,
                            score=0.0,
                            level=current_level,
                            flagged=self.decider.flagged,
                            reason="silence",
                        )
                    )
                    continue

                # Remove DC bias to prevent filterbank distortion
                window_clean = window - float(np.mean(window))
                processed_window = crop_or_pad(window_clean, self.WINDOW_SAMPLES)
                normalized_window = rms_normalize(processed_window, target_dbfs=-26.0)

                # Score with detector
                score = float(self.detector.score(normalized_window))

                # Update decider
                level, flagged, _ = self.decider.update(score)

                emitted_states.append(
                    SessionState(
                        t=eval_time_s,
                        score=score,
                        level=level,
                        flagged=flagged,
                        reason="scored",
                    )
                )

        return emitted_states
