from pathlib import Path

import pytest

from callshield.detectors.factory import DetectorConfig, build_detector
from callshield.detectors.fake import FakeDetector


@pytest.mark.unit
def test_build_detector_fake_default() -> None:
    """Factory creates FakeDetector when name is 'fake'."""
    config = DetectorConfig(name="fake")
    detector = build_detector(config)

    assert isinstance(detector, FakeDetector)


@pytest.mark.unit
def test_build_detector_fake_with_options() -> None:
    """Factory forwards scripted_scores and delay_ms to FakeDetector."""
    config = DetectorConfig(name="fake", scripted_scores=[0.5, -0.5], delay_ms=25.0)
    detector = build_detector(config)

    assert isinstance(detector, FakeDetector)
    assert detector.scripted_scores == [0.5, -0.5]
    assert detector.delay_ms == 25.0


@pytest.mark.unit
def test_build_detector_torch_missing_weights() -> None:
    """Factory raises FileNotFoundError if torch weights are missing."""
    config = DetectorConfig(name="torch", weights_path=Path("non_existent_weights.pt"))

    with pytest.raises(FileNotFoundError, match="not found"):
        build_detector(config)


@pytest.mark.unit
def test_build_detector_unknown_name() -> None:
    """Factory raises ValueError for unknown detector names."""
    config = DetectorConfig(name="unknown_model")

    with pytest.raises(ValueError, match="Unknown detector type"):
        build_detector(config)
