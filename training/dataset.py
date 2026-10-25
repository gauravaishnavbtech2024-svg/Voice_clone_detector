"""PyTorch Datasets for Voice Clone and Deepfake Audio Detection.

Supports:
- DirectoryVoiceCloneDataset: Automatically loads folder-based datasets (e.g. 'new training dataset')
  with subfolders for real samples ('real_samples') and diverse AI clone models ('OpenAI', 'xTTS', etc.).
- ASVspoofLADataset: Standard protocol-based loader for ASVspoof 2019 Logical Access.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from training.augment import augment_audio, preprocess_eval


class BaseVoiceCloneDataset(Dataset):
    """Base dataset class providing safe audio loading, resampling, and augmentation."""

    def __init__(
        self,
        samples: list[tuple[Path, float]],
        is_train: bool = True,
        target_sample_rate: int = 16000,
        window_samples: int = 64000,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.is_train = is_train
        self.target_sample_rate = target_sample_rate
        self.window_samples = window_samples

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
            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample to 16 kHz if necessary
            if sr != self.target_sample_rate:
                g = math.gcd(int(sr), self.target_sample_rate)
                data = resample_poly(
                    data,
                    up=self.target_sample_rate // g,
                    down=int(sr) // g,
                ).astype(np.float32)

        except Exception:
            data = np.zeros(self.window_samples, dtype=np.float32)

        if self.is_train:
            waveform = augment_audio(data, n_samples=self.window_samples)
        else:
            waveform = preprocess_eval(data, n_samples=self.window_samples)

        waveform_tensor = torch.from_numpy(waveform).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return waveform_tensor, label_tensor


class DirectoryVoiceCloneDataset(BaseVoiceCloneDataset):
    """Dataset loader that discovers audio files from directory structures.

    Recognizes:
    - Real / Bonafide human voice folder: 'real_samples', 'real', 'bonafide', 'genuine' (Label: 1.0)
    - Synthetic clone voice folders: All other subfolders, e.g. 'OpenAI', 'xTTS', 'VoiceBox', etc. (Label: 0.0)
    """

    @classmethod
    def create_train_val_split(
        cls,
        data_dir: str | Path,
        val_ratio: float = 0.15,
        seed: int = 42,
        max_samples: int | None = None,
        target_sample_rate: int = 16000,
        window_samples: int = 64000,
    ) -> tuple[DirectoryVoiceCloneDataset, DirectoryVoiceCloneDataset]:
        """Discovers files in data_dir and creates stratified train and validation datasets."""
        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root directory not found at: {root}")

        real_names = {"real_samples", "real", "bonafide", "genuine", "human"}
        real_files: list[Path] = []
        spoof_by_generator: dict[str, list[Path]] = {}

        audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}

        for item in sorted(root.iterdir()):
            if not item.is_dir():
                continue
            folder_name = item.name.lower()
            files = [f for f in item.rglob("*") if f.is_file() and f.suffix.lower() in audio_exts]

            if folder_name in real_names:
                real_files.extend(files)
            else:
                spoof_by_generator[item.name] = files

        rng = random.Random(seed)

        # Stratified split: Real samples
        rng.shuffle(real_files)
        n_val_real = int(len(real_files) * val_ratio)
        val_real = real_files[:n_val_real]
        train_real = real_files[n_val_real:]

        train_samples: list[tuple[Path, float]] = [(p, 1.0) for p in train_real]
        val_samples: list[tuple[Path, float]] = [(p, 1.0) for p in val_real]

        # Stratified split: Each synthetic generator
        for gen_name, g_files in spoof_by_generator.items():
            rng.shuffle(g_files)
            n_val_g = int(len(g_files) * val_ratio)
            val_samples.extend((p, 0.0) for p in g_files[:n_val_g])
            train_samples.extend((p, 0.0) for p in g_files[n_val_g:])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)

        if max_samples is not None:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[: max(1, int(max_samples * val_ratio))]

        print(
            f"Loaded Directory Dataset from '{root.name}':\n"
            f"  Train: {len(train_samples)} items (Bonafide: {sum(1 for _, l in train_samples if l == 1.0)}, "
            f"Spoof: {sum(1 for _, l in train_samples if l == 0.0)})\n"
            f"  Val:   {len(val_samples)} items (Bonafide: {sum(1 for _, l in val_samples if l == 1.0)}, "
            f"Spoof: {sum(1 for _, l in val_samples if l == 0.0)})\n"
            f"  Generators: {list(spoof_by_generator.keys())}"
        )

        train_ds = cls(
            samples=train_samples,
            is_train=True,
            target_sample_rate=target_sample_rate,
            window_samples=window_samples,
        )
        val_ds = cls(
            samples=val_samples,
            is_train=False,
            target_sample_rate=target_sample_rate,
            window_samples=window_samples,
        )
        return train_ds, val_ds


class ASVspoofLADataset(BaseVoiceCloneDataset):
    """Dataset loader for ASVspoof 2019 Logical Access audio and protocol files."""

    def __init__(
        self,
        protocol_path: str | Path | Sequence[tuple[str | Path, str | Path]],
        audio_dir: str | Path | None = None,
        is_train: bool = True,
        max_samples: int | None = None,
        target_sample_rate: int = 16000,
        window_samples: int = 64000,
    ) -> None:
        if isinstance(protocol_path, (str, Path)):
            if audio_dir is None:
                raise ValueError("audio_dir must be provided when protocol_path is a single path")
            pairs = [(Path(protocol_path), Path(audio_dir))]
        else:
            pairs = [(Path(p), Path(a)) for p, a in protocol_path]

        samples = self._load_protocols(pairs, max_samples)
        super().__init__(
            samples=samples,
            is_train=is_train,
            target_sample_rate=target_sample_rate,
            window_samples=window_samples,
        )

    def _load_protocols(
        self,
        pairs: list[tuple[Path, Path]],
        max_samples: int | None = None,
    ) -> list[tuple[Path, float]]:
        samples: list[tuple[Path, float]] = []
        num_bonafide = 0
        num_spoof = 0

        for proto_path, audio_dir in pairs:
            if not proto_path.is_file():
                raise FileNotFoundError(f"Protocol file not found at: {proto_path}")
            if not audio_dir.is_dir():
                raise FileNotFoundError(f"Audio directory not found at: {audio_dir}")

            with open(proto_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    audio_id = parts[1]
                    key = parts[4].lower()
                    label = 1.0 if key == "bonafide" else 0.0

                    if label == 1.0:
                        num_bonafide += 1
                    else:
                        num_spoof += 1

                    audio_file = audio_dir / f"{audio_id}.flac"
                    samples.append((audio_file, label))

                    if max_samples is not None and len(samples) >= max_samples:
                        break

            if max_samples is not None and len(samples) >= max_samples:
                break

        print(
            f"Loaded {len(samples)} total items from {len(pairs)} partition(s) "
            f"(Bonafide: {num_bonafide}, Spoof: {num_spoof})"
        )
        return samples
