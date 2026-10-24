"""Deterministic audio fixture generator for Call Shield tests.

Generates 16 kHz mono WAV fixtures using fixed random seeds:
  - genuine_like.wav: 4.0s multi-frequency noise-like speech stand-in (high spectral flatness)
  - fake_like.wav: 4.0s pure 440 Hz tone (low spectral flatness)
  - silence.wav: 4.0s digital silence
  - mixed.wav: 14.0s total (first 6.0s genuine-like, followed by 8.0s fake-like)
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
FIXTURES_DIR = Path(__file__).resolve().parent


def generate_genuine_like_audio(duration_s: float = 4.0, seed: int = 42) -> np.ndarray:
    """Generates noise-like signal with high spectral flatness acting as speech stand-in."""
    rng = np.random.default_rng(seed)
    num_samples = int(duration_s * SAMPLE_RATE)

    # Multi-frequency component + white noise
    t = np.arange(num_samples) / SAMPLE_RATE
    harmonics = (
        0.2 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 600 * t)
    )
    noise = rng.standard_normal(num_samples) * 0.4
    raw = harmonics + noise

    # Normalize to RMS ≈ -26 dBFS (target_rms ≈ 0.05)
    rms = np.sqrt(np.mean(raw**2))
    target_rms = 10.0 ** (-26.0 / 20.0)
    scaled = raw * (target_rms / rms)
    return np.asarray(np.clip(scaled, -1.0, 1.0), dtype=np.float32)


def generate_fake_like_audio(duration_s: float = 4.0, freq: float = 440.0) -> np.ndarray:
    """Generates a pure single-frequency sinusoidal tone with very low spectral flatness."""
    num_samples = int(duration_s * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    raw = np.sin(2 * np.pi * freq * t)

    # Normalize to RMS ≈ -26 dBFS
    target_rms = 10.0 ** (-26.0 / 20.0)
    scaled = raw * (target_rms / np.sqrt(np.mean(raw**2)))
    return np.asarray(np.clip(scaled, -1.0, 1.0), dtype=np.float32)


def generate_silence_audio(duration_s: float = 4.0) -> np.ndarray:
    """Generates digital silence."""
    num_samples = int(duration_s * SAMPLE_RATE)
    return np.zeros(num_samples, dtype=np.float32)


def generate_mixed_audio(genuine_s: float = 6.0, fake_s: float = 8.0) -> np.ndarray:
    """Generates composite audio: 6.0s genuine-like followed by 8.0s fake-like."""
    part1 = generate_genuine_like_audio(duration_s=genuine_s, seed=100)
    part2 = generate_fake_like_audio(duration_s=fake_s, freq=440.0)
    return np.concatenate([part1, part2]).astype(np.float32)


def generate_all_fixtures(output_dir: Path | None = None) -> dict[str, Path]:
    """Generates and writes all four fixture WAV files."""
    dest = output_dir or FIXTURES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    # 1. Genuine-like (4.0s)
    genuine = generate_genuine_like_audio(4.0)
    path_genuine = dest / "genuine_like.wav"
    wavfile.write(path_genuine, SAMPLE_RATE, genuine)
    files["genuine_like"] = path_genuine

    # 2. Fake-like (4.0s)
    fake = generate_fake_like_audio(4.0)
    path_fake = dest / "fake_like.wav"
    wavfile.write(path_fake, SAMPLE_RATE, fake)
    files["fake_like"] = path_fake

    # 3. Silence (4.0s)
    silence = generate_silence_audio(4.0)
    path_silence = dest / "silence.wav"
    wavfile.write(path_silence, SAMPLE_RATE, silence)
    files["silence"] = path_silence

    # 4. Mixed (14.0s: 6.0s genuine + 8.0s fake)
    mixed = generate_mixed_audio(6.0, 8.0)
    path_mixed = dest / "mixed.wav"
    wavfile.write(path_mixed, SAMPLE_RATE, mixed)
    files["mixed"] = path_mixed

    return files


if __name__ == "__main__":
    generated = generate_all_fixtures()
    for name, path in generated.items():
        print(f"Generated {name}: {path}")
