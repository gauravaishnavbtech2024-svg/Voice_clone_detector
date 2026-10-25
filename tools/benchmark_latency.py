"""Phase 8 Latency and Concurrency Benchmark.

Measures:
  1. Median and p95 inference time for one 4.0-second window (64,000 samples @ 16 kHz).
  2. Real-Time Factor (RTF = latency / 4.0s).
  3. Maximum concurrent streams sustained before stream processing lag exceeds 1.0 second.

Usage:
  python tools/benchmark_latency.py [--device cuda|cpu] [--iterations 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from callshield.detectors.torch_detector import TorchDetector


def benchmark_single_window_latency(
    detector: TorchDetector,
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Measures single-window inference latency statistics over N iterations."""
    rng = np.random.default_rng(42)
    # 4s window @ 16 kHz, float32, normalized
    test_window = rng.standard_normal(64000).astype(np.float32) * 0.1

    # Warmup runs
    for _ in range(warmup):
        _ = detector.score(test_window)

    if detector.device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = detector.score(test_window)
        if detector.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    arr = np.array(times)
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    mean_ms = float(np.mean(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    rtf = (median_ms / 1000.0) / 4.0

    return {
        "iterations": float(iterations),
        "min_ms": min_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": max_ms,
        "rtf": rtf,
    }


def benchmark_concurrent_streams(
    detector: TorchDetector,
    stream_levels: list[int] | None = None,
    max_lag_seconds: float = 1.0,
) -> tuple[int, list[dict[str, object]]]:
    """Finds maximum concurrent streams sustained before inference lag exceeds max_lag_seconds."""
    if stream_levels is None:
        stream_levels = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]

    rng = np.random.default_rng(101)
    results: list[dict[str, object]] = []
    max_sustained = 0

    for streams in stream_levels:
        windows = [
            (rng.standard_normal(64000).astype(np.float32) * 0.1)
            for _ in range(streams)
        ]

        # Execute concurrent stream batch scoring
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=streams) as pool:
            futures = [pool.submit(detector.score, w) for w in windows]
            _ = [f.result() for f in futures]
            if detector.device.type == "cuda":
                torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        per_stream_lag = total_time  # in 1s real-world interval, all streams must be scored

        passed = per_stream_lag <= max_lag_seconds
        if passed:
            max_sustained = max(max_sustained, streams)

        results.append({
            "streams": streams,
            "total_time_ms": round(total_time * 1000.0, 1),
            "lag_seconds": round(per_stream_lag, 3),
            "sustained": passed,
        })

        if not passed and streams >= 8:
            break

    return max_sustained, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Shield Phase 8 Latency & Concurrency Benchmark")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(BASE_DIR / "models" / "best.pt"),
        help="Path to trained PyTorch weights checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to benchmark ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for single-window latency",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"ERROR: Model weights not found at {weights_path}")
        sys.exit(1)

    print("=" * 65)
    print("  CALL SHIELD PHASE 8: LATENCY & CONCURRENCY BENCHMARK")
    print("=" * 65)
    print(f"Checkpoint: {weights_path.name}")
    print(f"Device:     {args.device}")
    if args.device == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Iterations: {args.iterations}")
    print("-" * 65)

    print("[*] Initializing TorchDetector...")
    detector = TorchDetector(weights_path=weights_path, device=args.device)

    print("[*] Benchmarking single-window (4.0s) latency...")
    lat = benchmark_single_window_latency(detector, iterations=args.iterations)

    print("\n[+] Single Window (4.0s / 64,000 samples) Latency Results:")
    print(f"    - Median Latency:     {lat['median_ms']:.2f} ms")
    print(f"    - p95 Latency:        {lat['p95_ms']:.2f} ms")
    print(f"    - p99 Latency:        {lat['p99_ms']:.2f} ms")
    print(f"    - Mean Latency:       {lat['mean_ms']:.2f} ms")
    print(f"    - Min / Max:          {lat['min_ms']:.2f} ms / {lat['max_ms']:.2f} ms")
    print(f"    - Real-Time Factor:   {lat['rtf']:.4f}x (lower is faster; < 0.05 is exceptional)")

    print("\n[*] Benchmarking concurrent streams (max lag <= 1.0s)...")
    max_sustained, stream_results = benchmark_concurrent_streams(detector)

    print("\n[+] Concurrent Stream Scalability:")
    print(f"    {'Streams':<10} {'Batch Time (ms)':<18} {'Lag (s)':<12} {'Status'}")
    print("    " + "-" * 50)
    for r in stream_results:
        status_str = "SUSTAINED" if r["sustained"] else "EXCEEDED LAG (> 1.0s)"
        print(f"    {r['streams']:<10} {r['total_time_ms']:<18} {r['lag_seconds']:<12} {status_str}")

    print("\n" + "=" * 65)
    print(f"  MAX SUSTAINED CONCURRENT STREAMS: {max_sustained} streams")
    print(f"  4s WINDOW INFERENCE MEDIAN:       {lat['median_ms']:.2f} ms")
    print(f"  4s WINDOW INFERENCE p95:          {lat['p95_ms']:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
