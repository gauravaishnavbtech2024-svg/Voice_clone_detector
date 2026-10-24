"""PyTorch-based voice clone detector for Call Shield."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from callshield.detectors.base import Detector
from training.model import VoiceCloneDetectorModel


class TorchDetector(Detector):
    """Detector implementation using a trained PyTorch checkpoint.

    Conforms to the Detector Protocol:
      score(x: np.ndarray) -> float
      where x is 4.0s (64,000 samples @ 16 kHz), float32, RMS-normalised.
      Score > 0 indicates GENUINE speech, < 0 indicates SYNTHETIC speech.
    """

    name: str = "torch"

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.weights_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold: float = 0.0
        self.model = VoiceCloneDetectorModel().to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.threshold = float(checkpoint.get("threshold", 0.0))
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            self.threshold = float(checkpoint.get("threshold", 0.0))
        else:
            raise ValueError(f"Unrecognized checkpoint format in {self.weights_path}")

        self.model.load_state_dict(state_dict)

    def score(self, x: np.ndarray) -> float:
        """Scores a 4.0-second, 16 kHz, float32, normalized mono audio window.

        Args:
            x: 1D numpy array of exactly 64,000 samples and dtype float32.

        Returns:
            float: > 0 for GENUINE speech, < 0 for SYNTHETIC speech.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)

        if x.dtype != np.float32:
            x = x.astype(np.float32)

        if x.ndim != 1 or len(x) != 64000:
            raise ValueError(f"Expected 1D audio array of exactly 64,000 samples, got shape {x.shape}")

        with torch.inference_mode():
            tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
            logit = self.model(tensor)
            val = float(logit.squeeze().item())

        if np.isnan(val) or np.isinf(val):
            return 0.0

        # Calibrate against model's optimal decision threshold
        calibrated_score = val - self.threshold
        return calibrated_score
