"""PyTorch Dataset for ASVspoof 2019 Logical Access (LA) voice clone dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from training.augment import augment_audio, preprocess_eval


class ASVspoofLADataset(Dataset):
    """Dataset loader for ASVspoof 2019 Logical Access audio and protocol files."""

    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        is_train: bool = True,
        max_samples: int | None = None,
        target_sample_rate: int = 16000,
        window_samples: int = 64000,
    ) -> None:
        super().__init__()
        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)
        self.is_train = is_train
        self.target_sample_rate = target_sample_rate
        self.window_samples = window_samples

        self.samples: list[tuple[Path, float]] = []
        self._load_protocol(max_samples)

    def _load_protocol(self, max_samples: int | None = None) -> None:
        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"Protocol file not found at: {self.protocol_path}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found at: {self.audio_dir}")

        num_bonafide = 0
        num_spoof = 0

        with open(self.protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Format: SPEAKER_ID AUDIO_ID - SYSTEM_ID KEY
                # e.g., LA_0079 LA_T_1138215 - - bonafide
                audio_id = parts[1]
                key = parts[4].lower()

                # Convention: 1.0 = Genuine / bonafide, 0.0 = Synthetic / spoof
                label = 1.0 if key == "bonafide" else 0.0

                if label == 1.0:
                    num_bonafide += 1
                else:
                    num_spoof += 1

                audio_file = self.audio_dir / f"{audio_id}.flac"
                self.samples.append((audio_file, label))

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        print(
            f"Loaded {len(self.samples)} items from {self.protocol_path.name} "
            f"(Bonafide: {num_bonafide}, Spoof: {num_spoof})"
        )

    @property
    def class_weights(self) -> float:
        """Computes ratio of spoof to bonafide samples for BCEWithLogitsLoss pos_weight."""
        num_bonafide = sum(1 for _, label in self.samples if label == 1.0)
        num_spoof = sum(1 for _, label in self.samples if label == 0.0)
        if num_bonafide == 0:
            return 1.0
        return float(num_spoof / num_bonafide)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, label = self.samples[idx]

        try:
            # Read FLAC audio file
            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
        except Exception as e:
            # Fallback for empty or corrupt audio file
            data = np.zeros(self.window_samples, dtype=np.float32)

        if self.is_train:
            waveform = augment_audio(data, n_samples=self.window_samples)
        else:
            waveform = preprocess_eval(data, n_samples=self.window_samples)

        waveform_tensor = torch.from_numpy(waveform).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return waveform_tensor, label_tensor
