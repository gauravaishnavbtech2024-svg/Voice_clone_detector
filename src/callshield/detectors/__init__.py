"""Detector abstraction, test doubles, and factory."""

from callshield.detectors.base import Detector
from callshield.detectors.factory import DetectorConfig, build_detector
from callshield.detectors.fake import FakeDetector

__all__ = [
    "Detector",
    "DetectorConfig",
    "FakeDetector",
    "build_detector",
]
