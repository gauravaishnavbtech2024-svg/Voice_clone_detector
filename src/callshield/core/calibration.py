"""Calibration and evaluation metrics for Call Shield (Phase 9).

Provides:
  - Equal Error Rate (EER) and ROC curve computation
  - Target False Alarm Rate (FAR) threshold search
  - AlertDecider session simulation (false alerts per hour & time-to-detect)
  - Pure-Python PNG ROC curve visualization generator
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

from callshield.core.decider import AlertDecider


def compute_roc(
    genuine_scores: np.ndarray | list[float],
    fake_scores: np.ndarray | list[float],
    num_thresholds: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes False Alarm Rate (FAR) and False Rejection Rate (FRR) curves across thresholds.

    Conventions:
      - score >= threshold -> classified as GENUINE
      - score < threshold  -> classified as FAKE / SYNTHETIC
      - FAR (False Alarm Rate): Fraction of genuine speech scored < threshold
      - FRR (False Rejection Rate / Miss Rate): Fraction of fake speech scored >= threshold

    Returns:
      thresholds: 1D array of evaluation thresholds (sorted ascending)
      far_curve: FAR values at each threshold
      frr_curve: FRR values at each threshold
    """
    gen = np.asarray(genuine_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)

    if len(gen) == 0 or len(fake) == 0:
        raise ValueError("genuine_scores and fake_scores must both contain at least 1 sample")

    min_val = min(float(np.min(gen)), float(np.min(fake))) - 0.1
    max_val = max(float(np.max(gen)), float(np.max(fake))) + 0.1
    thresholds = np.linspace(min_val, max_val, num_thresholds)

    # gen < thr is false alarm (genuine incorrectly flagged)
    far_curve = np.array([np.mean(gen < thr) for thr in thresholds], dtype=np.float64)
    # fake >= thr is false rejection (fake incorrectly accepted as genuine)
    frr_curve = np.array([np.mean(fake >= thr) for thr in thresholds], dtype=np.float64)

    return thresholds, far_curve, frr_curve


def compute_eer(
    genuine_scores: np.ndarray | list[float],
    fake_scores: np.ndarray | list[float],
    num_thresholds: int = 1000,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Equal Error Rate (EER) and operating threshold.

    Returns:
      (eer, eer_threshold, thresholds, far_curve, frr_curve)
    """
    thresholds, far_curve, frr_curve = compute_roc(
        genuine_scores, fake_scores, num_thresholds=num_thresholds
    )

    diff = np.abs(far_curve - frr_curve)
    idx = int(np.argmin(diff))
    eer = float((far_curve[idx] + frr_curve[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold, thresholds, far_curve, frr_curve


def find_threshold_for_target_far(
    genuine_scores: np.ndarray | list[float],
    target_far: float,
) -> float:
    """Finds the maximum threshold that guarantees FAR <= target_far on genuine audio.

    As target_far decreases, the returned threshold monotonically decreases (or stays equal),
    flagging fewer genuine windows.
    """
    if not (0.0 <= target_far <= 1.0):
        raise ValueError(f"target_far must be in [0.0, 1.0], got {target_far}")

    gen = np.asarray(genuine_scores, dtype=np.float64)
    if len(gen) == 0:
        raise ValueError("genuine_scores cannot be empty")

    sorted_gen = np.sort(gen)
    # The target_far percentile gives the score below which exactly target_far fraction falls
    thr = float(np.percentile(sorted_gen, target_far * 100.0))
    return thr


def calculate_false_alerts_per_hour(
    alert_count: int,
    duration_seconds: float,
) -> float:
    """Calculates normalized false alerts per hour of genuine speech."""
    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
    hours = duration_seconds / 3600.0
    return float(alert_count / hours)


def simulate_decider_stream(
    window_scores: list[float] | np.ndarray,
    thr: float = 0.0,
    k: int = 4,
    n: int = 6,
    hop_seconds: float = 1.0,
) -> dict[str, object]:
    """Simulates an AlertDecider operating over sequential window scores."""
    decider = AlertDecider(thr=thr, k=k, n=n)
    time_to_first_alert: float | None = None
    time_to_first_warning: float | None = None
    alert_count = 0
    warning_count = 0

    was_alert = False
    for i, s in enumerate(window_scores):
        elapsed_sec = (i + 1) * hop_seconds
        level, _, _ = decider.update(s)

        if level == "WARNING" and time_to_first_warning is None:
            time_to_first_warning = elapsed_sec
            warning_count += 1

        if level == "ALERT":
            if time_to_first_alert is None:
                time_to_first_alert = elapsed_sec
            if not was_alert:
                alert_count += 1
                was_alert = True
        else:
            was_alert = False

    return {
        "final_level": "ALERT" if decider.latched else ("WARNING" if decider.flagged >= k - 2 else "OK"),
        "latched": decider.latched,
        "alert_count": alert_count,
        "warning_count": warning_count,
        "time_to_first_alert": time_to_first_alert,
        "time_to_first_warning": time_to_first_warning,
    }


def write_roc_png(
    far_curve: np.ndarray,
    frr_curve: np.ndarray,
    eer: float,
    eer_thr: float,
    output_path: str | Path,
    width: int = 600,
    height: int = 600,
) -> Path:
    """Generates an ROC / DET curve PNG file without third-party graphics dependencies."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # RGB Canvas: 600x600 pixels
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # Margins and plot area
    pad_left, pad_right, pad_top, pad_bottom = 60, 40, 50, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Background grid (light gray)
    for x_pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        px = pad_left + int(x_pct * plot_w)
        canvas[pad_top : pad_top + plot_h, max(0, px - 1) : min(width, px + 1)] = [230, 230, 230]
    for y_pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        py = pad_top + int(y_pct * plot_h)
        canvas[max(0, py - 1) : min(height, py + 1), pad_left : pad_left + plot_w] = [230, 230, 230]

    # Axes borders (dark slate)
    canvas[pad_top : pad_top + plot_h, pad_left : pad_left + 2] = [60, 60, 60]
    canvas[pad_top + plot_h - 2 : pad_top + plot_h, pad_left : pad_left + plot_w] = [60, 60, 60]

    # Diagonal reference line (EER diagonal: FRR = FAR)
    for i in range(min(plot_w, plot_h)):
        px = pad_left + i
        py = (pad_top + plot_h) - int((i / plot_w) * plot_h)
        if 0 <= px < width and 0 <= py < height:
            canvas[py, px] = [180, 180, 180]

    # Plot ROC curve points: x = FAR, y = 1 - FRR (True Positive Rate)
    # or FAR vs FRR trade-off curve
    points = []
    for far, frr in zip(far_curve, frr_curve):
        clamped_far = max(0.0, min(1.0, float(far)))
        clamped_tpr = max(0.0, min(1.0, float(1.0 - frr)))
        px = pad_left + int(clamped_far * plot_w)
        py = (pad_top + plot_h) - int(clamped_tpr * plot_h)
        points.append((px, py))

    # Draw smooth line connecting points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for s in range(steps + 1):
            cx = int(x1 + (x2 - x1) * (s / steps))
            cy = int(y1 + (y2 - y1) * (s / steps))
            if pad_left <= cx < pad_left + plot_w and pad_top <= cy < pad_top + plot_h:
                canvas[max(0, cy - 1) : min(height, cy + 2), max(0, cx - 1) : min(width, cx + 2)] = [
                    20,
                    120,
                    230,
                ]

    # Mark EER operating point with crimson crosshair
    eer_px = pad_left + int(eer * plot_w)
    eer_py = (pad_top + plot_h) - int((1.0 - eer) * plot_h)
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            if abs(dx) <= 1 or abs(dy) <= 1:
                cx, cy = eer_px + dx, eer_py + dy
                if 0 <= cx < width and 0 <= cy < height:
                    canvas[cy, cx] = [220, 20, 40]

    # Encode to standard uncompressed PNG using zlib
    raw_data = bytearray()
    for row in canvas:
        raw_data.append(0)  # filter type 0: None
        raw_data.extend(row.tobytes())

    compressed_idat = zlib.compress(bytes(raw_data), level=6)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    png_bytes = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes.extend(_png_chunk(b"IHDR", ihdr))
    png_bytes.extend(_png_chunk(b"IDAT", compressed_idat))
    png_bytes.extend(_png_chunk(b"IEND", b""))

    out_p.write_bytes(bytes(png_bytes))
    return out_p
