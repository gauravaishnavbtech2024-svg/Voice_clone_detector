"""Phase 9 Calibration and Evaluation Tool for Call Shield.

Scores audio recordings listed in an evaluation manifest using sliding windows,
computes calibration metrics, and generates an ROC curve plot.

Reports:
  1. Equal Error Rate (EER) and operating threshold.
  2. ROC curve saved as PNG.
  3. Operating threshold for target False Alarm Rate (FAR, e.g. 1%).
  4. False alerts per hour of genuine audio (via AlertDecider).
  5. Time-to-detect (TTD) on synthetic/clone recordings.

Usage:
  python tools/calibrate.py --manifest tests/fixtures/eval_manifest.json
  python tools/calibrate.py --synthetic  # Runs on self-contained synthetic test suite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from callshield.core.audio import crop_or_pad, rms_normalize
from callshield.core.calibration import (
    calculate_false_alerts_per_hour,
    compute_eer,
    find_threshold_for_target_far,
    simulate_decider_stream,
    write_roc_png,
)
from callshield.core.decider import AlertDecider
from callshield.detectors.factory import DetectorConfig, build_detector


def load_audio_normalized(file_path: Path | str) -> np.ndarray:
    """Loads and standardizes an audio file to 16 kHz mono float32."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    ext = p.suffix.lower()
    if ext in [".wav", ".flac", ".ogg", ".mp3"]:
        try:
            import soundfile as sf
            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr), 16000)
                data = resample_poly(data, up=16000 // g, down=int(sr) // g).astype(np.float32)
            return data
        except Exception:
            pass

    # Windows Media Foundation / audioread fallback
    try:
        from detector_app import load_audio
        return load_audio(str(p))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {p}: {e}") from e


def score_file_sliding_windows(
    audio_data: np.ndarray,
    detector: object,
    window_samples: int = 64000,
    hop_samples: int = 16000,
) -> list[float]:
    """Slices audio into 4s windows with 1s hop and returns detector scores."""
    total_samples = len(audio_data)

    if total_samples < window_samples:
        normed = rms_normalize(audio_data, target_dbfs=-26.0)
        window = crop_or_pad(normed, n=window_samples)
        return [float(detector.score(window))]

    scores: list[float] = []
    for start in range(0, total_samples - window_samples + 1, hop_samples):
        segment = audio_data[start : start + window_samples]
        normed = rms_normalize(segment, target_dbfs=-26.0)
        score = float(detector.score(normed))
        scores.append(score)

    return scores


def generate_synthetic_evaluation_manifest(
    output_dir: Path,
) -> tuple[Path, list[dict[str, object]]]:
    """Generates synthetic manifest and test audio for fast self-contained calibration testing."""
    import soundfile as sf
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    rng = np.random.default_rng(777)
    sr = 16000

    # 4 Genuine voice simulations (6.0s each = 24.0s total)
    for i in range(4):
        dur = 10.0
        n_samples = int(dur * sr)
        t = np.linspace(0, dur, n_samples, endpoint=False)
        # Speech formants + harmonic modulation
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
        noise = rng.standard_normal(n_samples) * 0.3
        harm = np.sin(2 * np.pi * 200 * t) * 0.2 + np.sin(2 * np.pi * 600 * t) * 0.15
        audio = ((noise + harm) * mod).astype(np.float32)

        fname = f"synth_genuine_{i+1}.wav"
        fpath = output_dir / fname
        sf.write(str(fpath), audio, sr)
        manifest_entries.append({"file": str(fpath), "label": "genuine", "duration": dur})

    # 4 Clone voice simulations (10.0s each = 7 windows each)
    for i in range(4):
        dur = 10.0
        n_samples = int(dur * sr)
        t = np.linspace(0, dur, n_samples, endpoint=False)
        # Pure synthetic monotone tone
        audio = (np.sin(2 * np.pi * (400 + i * 50) * t) * 0.4).astype(np.float32)

        fname = f"synth_clone_{i+1}.wav"
        fpath = output_dir / fname
        sf.write(str(fpath), audio, sr)
        manifest_entries.append({"file": str(fpath), "label": "synthetic", "duration": dur})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    return manifest_path, manifest_entries


def run_calibration(
    manifest_path: Path | None,
    detector_name: str = "torch",
    weights_path: Path | None = None,
    device: str = "cuda",
    target_far: float = 0.01,
    output_plot: Path = Path("docs/roc_curve.png"),
    is_synthetic: bool = False,
) -> dict[str, object]:
    """Runs sliding-window calibration on evaluation manifest."""
    # Handle synthetic test run if requested or manifest missing
    if is_synthetic or manifest_path is None or not manifest_path.is_file():
        print("[*] Generating synthetic evaluation suite...")
        synth_dir = BASE_DIR / "tests" / "fixtures" / "synth_eval"
        manifest_path, items = generate_synthetic_evaluation_manifest(synth_dir)
    else:
        raw_text = manifest_path.read_text(encoding="utf-8")
        items = json.loads(raw_text)

    # Initialize Detector
    config = DetectorConfig(
        name=detector_name,
        weights_path=weights_path if weights_path else BASE_DIR / "models" / "best.pt",
        device=device,
    )
    detector = build_detector(config)
    print(f"[*] Detector initialized: {detector.name} on {getattr(detector, 'device', 'default')}")

    genuine_scores: list[float] = []
    fake_scores: list[float] = []
    genuine_file_streams: list[list[float]] = []
    fake_file_streams: list[list[float]] = []
    total_genuine_duration = 0.0
    total_fake_duration = 0.0

    print(f"[*] Scoring {len(items)} audio recordings from manifest...")
    for idx, item in enumerate(items, 1):
        fpath = Path(item["file"])
        if not fpath.is_absolute():
            fpath = manifest_path.parent / fpath

        label = str(item.get("label", "")).lower()
        if label not in ["genuine", "synthetic", "fake", "clone"]:
            print(f"  [!] Skipping {fpath.name}: unknown label '{label}'")
            continue

        audio = load_audio_normalized(fpath)
        dur = len(audio) / 16000.0
        scores = score_file_sliding_windows(audio, detector)

        if label in ["genuine"]:
            genuine_scores.extend(scores)
            genuine_file_streams.append(scores)
            total_genuine_duration += dur
        else:
            fake_scores.extend(scores)
            fake_file_streams.append(scores)
            total_fake_duration += dur

    if not genuine_scores or not fake_scores:
        raise ValueError("Evaluation set must contain both genuine and synthetic audio recordings")

    # 1. Compute EER and curves
    eer, eer_thr, thresholds, far_curve, frr_curve = compute_eer(genuine_scores, fake_scores)

    # 2. Compute operating threshold for target FAR
    target_thr = find_threshold_for_target_far(genuine_scores, target_far)

    # 3. Simulate AlertDecider on Genuine Audio (False Alerts Per Hour)
    total_false_alerts = 0
    for stream in genuine_file_streams:
        sim = simulate_decider_stream(stream, thr=target_thr)
        total_false_alerts += int(sim["alert_count"])

    false_alerts_per_hour = calculate_false_alerts_per_hour(
        total_false_alerts, max(1.0, total_genuine_duration)
    )

    # 4. Simulate AlertDecider on Fake Audio (Time-To-Detect)
    ttd_list: list[float] = []
    for stream in fake_file_streams:
        sim = simulate_decider_stream(stream, thr=target_thr)
        if sim["time_to_first_alert"] is not None:
            ttd_list.append(float(sim["time_to_first_alert"]))

    mean_ttd = float(np.mean(ttd_list)) if ttd_list else float("nan")
    median_ttd = float(np.median(ttd_list)) if ttd_list else float("nan")

    # 5. Render and save ROC Curve PNG
    png_path = write_roc_png(
        far_curve=far_curve,
        frr_curve=frr_curve,
        eer=eer,
        eer_thr=eer_thr,
        output_path=output_plot,
    )

    results = {
        "num_files": len(items),
        "total_genuine_duration_sec": round(total_genuine_duration, 1),
        "total_fake_duration_sec": round(total_fake_duration, 1),
        "eer": round(eer, 4),
        "eer_percent": round(eer * 100.0, 2),
        "eer_threshold": round(eer_thr, 4),
        "target_far": target_far,
        "target_far_threshold": round(target_thr, 4),
        "false_alerts_per_hour": round(false_alerts_per_hour, 2),
        "mean_time_to_detect_sec": round(mean_ttd, 2),
        "median_time_to_detect_sec": round(median_ttd, 2),
        "roc_plot_saved": str(png_path),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Shield Phase 9 Calibration Tool")
    parser.add_argument("--manifest", type=str, default=None, help="Path to evaluation manifest.json")
    parser.add_argument("--synthetic", action="store_true", help="Run calibration on synthetic evaluation set")
    parser.add_argument("--detector", type=str, default="torch", choices=["torch", "fake"])
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-far", type=float, default=0.01, help="Target False Alarm Rate (e.g. 0.01 for 1%)")
    parser.add_argument("--output-plot", type=str, default="docs/roc_curve.png")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    weights_path = Path(args.weights)
    output_plot = Path(args.output_plot)

    print("=" * 65)
    print("  CALL SHIELD PHASE 9: MODEL CALIBRATION & EVALUATION")
    print("=" * 65)

    try:
        results = run_calibration(
            manifest_path=manifest_path,
            detector_name=args.detector,
            weights_path=weights_path,
            device=args.device,
            target_far=args.target_far,
            output_plot=output_plot,
            is_synthetic=args.synthetic or (manifest_path is None),
        )

        print("\n" + "=" * 65)
        print("  CALIBRATION REPORT SUMMARY")
        print("=" * 65)
        print(f"  Files Evaluated:          {results['num_files']}")
        print(f"  Genuine Audio Duration:   {results['total_genuine_duration_sec']}s")
        print(f"  Synthetic Audio Duration: {results['total_fake_duration_sec']}s")
        print("-" * 65)
        print(f"  Equal Error Rate (EER):   {results['eer_percent']}%  (EER Threshold: {results['eer_threshold']:+.3f})")
        print(f"  Target FAR:               {results['target_far'] * 100:.1f}%")
        print(f"  Calibrated Threshold:     {results['target_far_threshold']:+.3f}")
        print(f"  False Alerts / Hour:      {results['false_alerts_per_hour']} alerts/hr (Genuine speech)")
        print(f"  Time-to-Detect (Fakes):   {results['median_time_to_detect_sec']}s (median) / {results['mean_time_to_detect_sec']}s (mean)")
        print(f"  ROC Curve Saved:          {results['roc_plot_saved']}")
        print("=" * 65)
        print("\n[*] Recommended Environment Configuration:")
        print(f"    $env:CALLSHIELD_THRESHOLD = \"{results['target_far_threshold']:.3f}\"")
        print("-" * 65)

    except Exception as e:
        print(f"\n[!] Calibration Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
