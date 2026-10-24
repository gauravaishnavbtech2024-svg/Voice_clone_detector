"""PyTorch Model Architecture for Voice Clone / Synthetic Speech Detection.

Conforms to Call Shield convention:
- Input: 1D audio waveform at 16 kHz (64,000 samples for a 4.0s window)
- Output: Scalar logit where score > 0 = GENUINE and score < 0 = SYNTHETIC
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class ResidualBlock(nn.Module):
    """2D Convolutional Residual Block with Batch Normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class VoiceCloneDetectorModel(nn.Module):
    """Deep ResNet-based acoustic detector for synthetic voice / voice clone detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 640,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        # On-device GPU/CPU Log-Mel Spectrogram Front-End
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Convolutional Feature Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1: 32 -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        # Stage 2: 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )

        # Stage 3: 128 -> 256
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )

        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms 1D raw waveform (B, N) or (B, 1, N) to Log-Mel Spectrogram (B, 1, M, T)."""
        if x.dim() == 2:
            pass
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        else:
            raise ValueError(f"Expected waveform of shape (B, N) or (B, 1, N), got {x.shape}")

        spec = self.melspec(x)  # (B, n_mels, time)
        log_spec = torch.log(torch.clamp(spec, min=1e-6))
        return log_spec.unsqueeze(1)  # (B, 1, n_mels, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Raw waveform tensor of shape (B, 64000) or Log-Mel (B, 1, M, T).

        Returns:
            torch.Tensor: Scalar logit of shape (B,) or (B, 1).
                          > 0 indicates GENUINE, < 0 indicates SYNTHETIC.
        """
        if x.dim() in (1, 2) or (x.dim() == 3 and x.size(1) == 1):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            feat = self.extract_features(x)
        else:
            feat = x

        out = self.stem(feat)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logit = self.fc(out).squeeze(-1)
        return logit

    @torch.inference_mode()
    def score_numpy(self, waveform: np.ndarray, device: torch.device | None = None) -> float:
        """Evaluates a single 1D numpy audio array (64,000 samples @ 16 kHz).

        Returns:
            float: score > 0 (Genuine) or < 0 (Synthetic/Spoof).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)
        logit = self.forward(tensor)
        return float(logit.item())
