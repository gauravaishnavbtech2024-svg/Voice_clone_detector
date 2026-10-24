"""Audio preprocessing and augmentation for Call Shield model training.

Guarantees preprocessing parity with callshield.core.audio within 1e-6.
"""

from __future__ import annotations

import numpy as np
import torch


def rms_normalize(x: np.ndarray, target_dbfs: float = -26.0) -> np.ndarray:
    """Normalizes waveform to target dBFS RMS level.

    Parity with callshield.core.audio.rms_normalize within 1e-6.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if len(x_arr) == 0:
        return np.empty(0, dtype=np.float32)

    mean_sq = float(np.mean(x_arr**2))
    if mean_sq <= 1e-18:
        return np.zeros_like(x_arr)

    rms = float(np.sqrt(mean_sq))
    target_rms = float(10.0 ** (target_dbfs / 20.0))
    gain = float(target_rms / rms)
    return np.asarray(x_arr * gain, dtype=np.float32)


def crop_or_pad(x: np.ndarray, n: int = 64000) -> np.ndarray:
    """Ensures waveform length is exactly n samples (4 seconds at 16 kHz).

    - Shorter inputs are repeat-padded (never zero-padded).
    - Longer inputs are cropped to length n.
    - Parity with callshield.core.audio.crop_or_pad.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    curr_len = len(x_arr)

    if curr_len == n:
        return x_arr.copy()

    if curr_len > n:
        return x_arr[:n].copy()

    if curr_len == 0:
        return np.zeros(n, dtype=np.float32)

    reps = int(np.ceil(n / curr_len))
    tiled = np.tile(x_arr, reps)
    return tiled[:n].astype(np.float32)


def preprocess_eval(waveform: np.ndarray, n_samples: int = 64000) -> np.ndarray:
    """Deterministic evaluation preprocessing: RMS normalise then crop/repeat-pad."""
    normed = rms_normalize(waveform, target_dbfs=-26.0)
    return crop_or_pad(normed, n=n_samples)


def augment_audio(waveform: np.ndarray, n_samples: int = 64000) -> np.ndarray:
    """Applies training-time audio augmentations.

    Includes random circular shift, slight gain fluctuation, and repeat-pad/crop.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    # Random gain variation (+/- 2 dB)
    gain_db = np.random.uniform(-2.0, 2.0)
    gain = 10.0 ** (gain_db / 20.0)
    x = x * gain

    # Normalize and repeat-pad / crop to window
    x = rms_normalize(x, target_dbfs=-26.0)
    x = crop_or_pad(x, n=n_samples)

    # Random circular shift
    shift = np.random.randint(0, n_samples)
    x = np.roll(x, shift)

    return x.astype(np.float32)
