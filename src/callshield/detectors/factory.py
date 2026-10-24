"""Detector factory for instantiating detectors based on configuration."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from callshield.detectors.base import Detector
from callshield.detectors.fake import FakeDetector


@dataclass
class DetectorConfig:
    """Configuration for building voice clone detectors."""

    name: str = "fake"
    weights_path: Path | str | None = None
    scripted_scores: Sequence[float] | None = None
    delay_ms: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


def build_detector(config: DetectorConfig) -> Detector:
    """Builds and returns a Detector instance matching the configuration.

    Args:
        config: DetectorConfig instance.

    Returns:
        Detector: Initialized detector instance.

    Raises:
        FileNotFoundError: If torch checkpoint path does not exist.
        RuntimeError: If torch is requested but PyTorch is not installed.
        ValueError: If detector name is unrecognized.
    """
    detector_type = config.name.lower()

    if detector_type == "fake":
        return FakeDetector(
            scripted_scores=config.scripted_scores,
            delay_ms=config.delay_ms,
        )

    if detector_type == "torch":
        if not config.weights_path or not Path(config.weights_path).is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {config.weights_path}"
            )

        try:
            from callshield.detectors.torch_detector import TorchDetector  # type: ignore[import-untyped]

            return cast(Detector, TorchDetector(weights_path=Path(config.weights_path)))
        except ImportError as err:
            raise RuntimeError(
                "PyTorch is not installed. Install torch to use TorchDetector."
            ) from err

    raise ValueError(f"Unknown detector type: {config.name}")
