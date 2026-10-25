"""Phase 9 unit tests: Calibration, EER, and AlertDecider evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from callshield.core.calibration import (
    calculate_false_alerts_per_hour,
    compute_eer,
    compute_roc,
    find_threshold_for_target_far,
    simulate_decider_stream,
    write_roc_png,
)
from callshield.detectors.fake import FakeDetector


@pytest.mark.unit
def test_eer_perfect_separation() -> None:
    """EER = 0 for perfectly separable genuine and fake score distributions."""
    # Genuine scores all well above zero; fake scores all well below zero
    gen_scores = np.array([2.0, 2.5, 3.1, 4.0, 5.2, 1.8, 3.7])
    fake_scores = np.array([-1.5, -2.2, -3.0, -4.5, -5.0, -1.8, -2.9])

    eer, eer_thr, thresholds, far_curve, frr_curve = compute_eer(gen_scores, fake_scores)

    assert eer == 0.0, f"Expected EER 0.0 for perfectly separable data, got {eer}"
    # Optimal threshold must lie between the highest fake score and lowest genuine score
    assert -1.5 <= eer_thr <= 1.8


@pytest.mark.unit
def test_eer_random_scores() -> None:
    """EER ~ 0.5 (random guess) when genuine and fake scores share the same distribution."""
    rng = np.random.default_rng(42)
    # Both sampled from identical standard normal distributions (no discriminative signal)
    gen_scores = rng.standard_normal(2000)
    fake_scores = rng.standard_normal(2000)

    eer, eer_thr, _, _, _ = compute_eer(gen_scores, fake_scores)

    # Allow reasonable statistical bounds around 0.50
    assert 0.45 <= eer <= 0.55, f"Expected random EER ~ 0.5, got {eer:.3f}"


@pytest.mark.unit
def test_threshold_search_monotonic() -> None:
    """Threshold search is strictly monotonic: lower target FAR requires equal or lower threshold."""
    rng = np.random.default_rng(123)
    gen_scores = rng.normal(loc=2.0, scale=1.0, size=500)

    # Test decreasing target FARs: 20%, 10%, 5%, 1%, 0.1%
    target_fars = [0.20, 0.10, 0.05, 0.01, 0.001]
    thresholds = [find_threshold_for_target_far(gen_scores, far) for far in target_fars]

    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1], (
            f"Monotonicity violated: FAR {target_fars[i]} -> thr {thresholds[i]} "
            f"must be >= FAR {target_fars[i+1]} -> thr {thresholds[i+1]}"
        )


@pytest.mark.unit
def test_false_alerts_per_hour_arithmetic() -> None:
    """Verifies false-alerts-per-hour rate scaling arithmetic."""
    # 2 alerts in 30 minutes (1800s) -> 4 alerts per hour
    rate = calculate_false_alerts_per_hour(alert_count=2, duration_seconds=1800.0)
    assert rate == 4.0

    # 1 alert in 7200 seconds (2 hours) -> 0.5 alerts per hour
    rate_2h = calculate_false_alerts_per_hour(alert_count=1, duration_seconds=7200.0)
    assert rate_2h == 0.5

    # 0 alerts in 1 hour -> 0.0
    rate_zero = calculate_false_alerts_per_hour(alert_count=0, duration_seconds=3600.0)
    assert rate_zero == 0.0

    # Validation errors on non-positive duration
    with pytest.raises(ValueError):
        calculate_false_alerts_per_hour(alert_count=1, duration_seconds=0.0)


@pytest.mark.unit
def test_simulate_decider_stream_time_to_detect() -> None:
    """Verifies that an AlertDecider stream correctly tracks latency and alerts."""
    # 3 genuine windows (score +2.0) followed by 5 fake windows (score -2.0)
    # With k=4, threshold=0.0, hop=1.0s:
    # Windows:
    # 1: +2.0 (OK)
    # 2: +2.0 (OK)
    # 3: +2.0 (OK)
    # 4: -2.0 (flagged=1, OK)
    # 5: -2.0 (flagged=2, WARNING, elapsed=5s)
    # 6: -2.0 (flagged=3, WARNING)
    # 7: -2.0 (flagged=4 >= k, ALERT, elapsed=7s)
    stream_scores = [2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0]

    sim = simulate_decider_stream(stream_scores, thr=0.0, k=4, n=6, hop_seconds=1.0)

    assert sim["latched"] is True
    assert sim["final_level"] == "ALERT"
    assert sim["alert_count"] == 1
    assert sim["time_to_first_warning"] == 5.0
    assert sim["time_to_first_alert"] == 7.0


@pytest.mark.unit
def test_write_roc_png(tmp_path: Path) -> None:
    """Verifies that write_roc_png produces a valid standard PNG image file."""
    far_curve = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    frr_curve = np.array([1.0, 0.4, 0.2, 0.1, 0.05, 0.0])
    out_file = tmp_path / "roc_curve.png"

    result_path = write_roc_png(
        far_curve=far_curve,
        frr_curve=frr_curve,
        eer=0.08,
        eer_thr=0.45,
        output_path=out_file,
    )

    assert result_path.is_file()
    data = result_path.read_bytes()
    assert len(data) > 100
    # Standard PNG magic number: \x89PNG\r\n\x1a\n
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
