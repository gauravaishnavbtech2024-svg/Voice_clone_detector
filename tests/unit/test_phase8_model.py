"""Phase 8 tests: Preprocessing parity and model integration."""

from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            @staticmethod
            def unit(f):
                return f
    pytest = _MockPytest()

from callshield.core.audio import crop_or_pad, rms_normalize
from training.augment import preprocess_eval


@pytest.mark.unit
def test_preprocessing_parity() -> None:
    """Parity: core inference preprocessing equals training.augment preprocessing within 1e-6."""
    np.random.seed(42)
    # Generate random waveforms of different lengths and amplitudes
    for length in [16000, 32000, 64000, 80000, 128000]:
        waveform = np.random.uniform(-0.8, 0.8, size=length).astype(np.float32)

        # Core preprocessing
        normed = rms_normalize(waveform, target_dbfs=-26.0)
        core_preprocessed = crop_or_pad(normed, n=64000)

        # Training evaluation preprocessing
        train_preprocessed = preprocess_eval(waveform, n_samples=64000)

        assert core_preprocessed.shape == (64000,)
        assert train_preprocessed.shape == (64000,)
        max_diff = np.max(np.abs(core_preprocessed - train_preprocessed))
        assert max_diff < 1e-6, f"Max difference {max_diff} exceeded 1e-6 tolerance"
