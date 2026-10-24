import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from callshield.detectors.base import Detector
from callshield.detectors.fake import FakeDetector

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def genuine_audio_4s() -> np.ndarray:
    path = FIXTURES_DIR / "genuine_like.wav"
    _, data = wavfile.read(path)
    return np.asarray(data[:64000], dtype=np.float32)


@pytest.fixture(scope="module")
def fake_audio_4s() -> np.ndarray:
    path = FIXTURES_DIR / "fake_like.wav"
    _, data = wavfile.read(path)
    return np.asarray(data[:64000], dtype=np.float32)


@pytest.mark.unit
@pytest.mark.parametrize("detector_factory", [lambda: FakeDetector()])
def test_detector_contract_finiteness_and_determinism(
    detector_factory: type, genuine_audio_4s: np.ndarray
) -> None:
    """Contract: detector returns float, is finite, and deterministic for identical input."""
    detector: Detector = detector_factory()
    score1 = detector.score(genuine_audio_4s)
    score2 = detector.score(genuine_audio_4s)

    assert isinstance(score1, float)
    assert np.isfinite(score1)
    assert score1 == score2


@pytest.mark.unit
def test_fake_detector_spectral_flatness_polarity(
    genuine_audio_4s: np.ndarray, fake_audio_4s: np.ndarray
) -> None:
    """Contract: score > 0 for genuine-like (noise-rich) and < 0 for fake-like (pure tone)."""
    detector = FakeDetector()

    genuine_score = detector.score(genuine_audio_4s)
    fake_score = detector.score(fake_audio_4s)

    assert genuine_score > 0.0, f"Expected genuine score > 0, got {genuine_score}"
    assert fake_score < 0.0, f"Expected fake score < 0, got {fake_score}"


@pytest.mark.unit
def test_detector_contract_input_shape_validation(genuine_audio_4s: np.ndarray) -> None:
    """Contract: handles exactly-4s input (64000 samples) and rejects wrong shape."""
    detector = FakeDetector()

    # Exact 4s (64,000 samples) succeeds
    assert len(genuine_audio_4s) == 64000
    score = detector.score(genuine_audio_4s)
    assert isinstance(score, float)

    # Wrong length: shorter
    with pytest.raises(ValueError, match="Expected 64000 samples"):
        detector.score(genuine_audio_4s[:32000])

    # Wrong length: longer
    with pytest.raises(ValueError, match="Expected 64000 samples"):
        detector.score(np.concatenate([genuine_audio_4s, genuine_audio_4s]))

    # Wrong shape: 2D
    with pytest.raises(ValueError, match="1D"):
        detector.score(genuine_audio_4s.reshape(1, -1))


@pytest.mark.unit
def test_detector_contract_input_dtype_validation(genuine_audio_4s: np.ndarray) -> None:
    """Contract: rejects non-float32 dtypes with ValueError."""
    detector = FakeDetector()

    int16_audio = (genuine_audio_4s * 32767).astype(np.int16)
    with pytest.raises(ValueError, match="float32"):
        detector.score(int16_audio)

    float64_audio = genuine_audio_4s.astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        detector.score(float64_audio)


@pytest.mark.unit
def test_fake_detector_scripted_replay(genuine_audio_4s: np.ndarray) -> None:
    """FakeDetector can replay a scripted score list."""
    scripted = [1.5, -0.7, 0.2, -1.8]
    detector = FakeDetector(scripted_scores=scripted)

    for expected in [1.5, -0.7, 0.2, -1.8]:
        assert detector.score(genuine_audio_4s) == expected

    # After scripted queue is exhausted, falls back to spectral flatness
    fallback_score = detector.score(genuine_audio_4s)
    assert fallback_score > 0.0


@pytest.mark.unit
def test_fake_detector_delay_simulation(genuine_audio_4s: np.ndarray) -> None:
    """FakeDetector can sleep N ms to simulate slow inference."""
    delay_ms = 50.0
    detector = FakeDetector(delay_ms=delay_ms)

    start = time.perf_counter()
    detector.score(genuine_audio_4s)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms >= delay_ms * 0.8
