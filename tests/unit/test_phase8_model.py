"""Phase 8 tests: Preprocessing parity, TorchDetector contract tests, and real fixture validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from callshield.core.audio import crop_or_pad, rms_normalize
from callshield.detectors.torch_detector import TorchDetector
from training.augment import preprocess_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = REPO_ROOT / "models" / "best.pt"
REAL_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "real"


@pytest.mark.unit
def test_preprocessing_parity() -> None:
    """Parity: core inference preprocessing equals training.augment preprocessing within 1e-6."""
    np.random.seed(42)
    # Generate random waveforms of different lengths and amplitudes
    for length in [16000, 32000, 64000, 80000, 128000]:
        waveform = np.random.uniform(-0.8, 0.8, size=length).astype(np.float32)

        # Core preprocessing: rms_normalize to -26 dBFS, then crop_or_pad to 64,000 samples
        normed = rms_normalize(waveform, target_dbfs=-26.0)
        core_preprocessed = crop_or_pad(normed, n=64000)

        # Training evaluation preprocessing
        train_preprocessed = preprocess_eval(waveform, n_samples=64000)

        assert core_preprocessed.shape == (64000,)
        assert train_preprocessed.shape == (64000,)
        max_diff = np.max(np.abs(core_preprocessed - train_preprocessed))
        assert max_diff < 1e-6, f"Max difference {max_diff} exceeded 1e-6 tolerance"


@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="models/best.pt not found")
def test_torch_detector_contract_cpu() -> None:
    """TorchDetector contract test: outputs finite float on CPU for single 64k window."""
    detector = TorchDetector(weights_path=WEIGHTS_PATH, device="cpu")
    assert detector.name == "torch"
    assert detector.device == torch.device("cpu")

    # Generate 4s test window
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(64000).astype(np.float32)
    x = rms_normalize(x, target_dbfs=-26.0)

    score = detector.score(x)
    assert isinstance(score, float)
    assert math.isfinite(score)


@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="models/best.pt not found")
def test_torch_detector_deterministic() -> None:
    """TorchDetector contract test: deterministic (eval mode, no dropout)."""
    detector = TorchDetector(weights_path=WEIGHTS_PATH, device="cpu")

    rng = np.random.default_rng(42)
    x = rng.standard_normal(64000).astype(np.float32)
    x = rms_normalize(x, target_dbfs=-26.0)

    score1 = detector.score(x)
    score2 = detector.score(x)
    assert score1 == score2, f"Expected deterministic output, got {score1} vs {score2}"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    not (WEIGHTS_PATH.is_file() and torch.cuda.is_available()),
    reason="GPU or models/best.pt not available",
)
def test_torch_detector_contract_gpu() -> None:
    """TorchDetector contract test: runs on CUDA GPU if available."""
    detector = TorchDetector(weights_path=WEIGHTS_PATH, device="cuda")
    assert detector.device.type == "cuda"

    rng = np.random.default_rng(999)
    x = rng.standard_normal(64000).astype(np.float32)
    x = rms_normalize(x, target_dbfs=-26.0)

    score = detector.score(x)
    assert isinstance(score, float)
    assert math.isfinite(score)


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_FIXTURES_DIR.is_dir() and (REAL_FIXTURES_DIR / "manifest.json").is_file()),
    reason="tests/fixtures/real/ directory or manifest.json absent (Step 8b)",
)
def test_real_fixtures_classification() -> None:
    """Classifies recorded genuine and synthetic clips against manifest labels."""
    import soundfile as sf

    detector = TorchDetector(weights_path=WEIGHTS_PATH)
    manifest_path = REAL_FIXTURES_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results = []
    for item in manifest:
        audio_file = REAL_FIXTURES_DIR / item["file"]
        expected_label = item["label"]  # 'genuine' or 'synthetic'

        data, sr = sf.read(str(audio_file), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Preprocess
        normed = rms_normalize(data, target_dbfs=-26.0)
        window = crop_or_pad(normed, n=64000)
        score = detector.score(window)

        predicted_label = "genuine" if score > 0 else "synthetic"
        results.append({
            "file": item["file"],
            "expected": expected_label,
            "predicted": predicted_label,
            "score": score,
        })
        assert (
            predicted_label == expected_label
        ), f"File {item['file']} expected {expected_label} but scored {score:.3f} ({predicted_label})"
