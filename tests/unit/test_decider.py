import math

import pytest

from callshield.core.decider import AlertDecider


@pytest.mark.unit
def test_decider_invalid_params() -> None:
    """Case 7: n < k raises ValueError, and k <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        AlertDecider(thr=0.0, k=4, n=3)

    with pytest.raises(ValueError):
        AlertDecider(thr=0.0, k=0, n=6)


@pytest.mark.unit
def test_decider_six_genuine_scores() -> None:
    """Case 1: Six genuine scores -> OK."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    for score in [0.8, 1.2, 0.5, 0.9, 1.5, 0.4]:
        level, flagged, _ = decider.update(score)
        assert level == "OK"
        assert flagged == 0


@pytest.mark.unit
def test_decider_one_outlier_among_genuine() -> None:
    """Case 2: One outlier among genuine -> OK."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    scores = [0.8, 0.5, -0.2, 0.9, 1.1, 0.4]
    final_level = "OK"
    final_flagged = 0
    for s in scores:
        final_level, final_flagged, _ = decider.update(s)

    assert final_level == "OK"
    assert final_flagged == 1


@pytest.mark.unit
def test_decider_two_suspicious_warning() -> None:
    """Case 3: Two suspicious in a row -> WARNING (flagged >= k-2, where k=4)."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    decider.update(0.5)
    decider.update(0.8)
    # First suspicious
    level1, flagged1, _ = decider.update(-0.4)
    assert level1 == "OK"
    assert flagged1 == 1

    # Second suspicious in a row -> reaches k-2 (2) -> WARNING
    level2, flagged2, _ = decider.update(-0.7)
    assert level2 == "WARNING"
    assert flagged2 == 2


@pytest.mark.unit
def test_decider_four_suspicious_alert_latches() -> None:
    """Case 4: Four suspicious within the last six -> ALERT."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    sequence = [-0.5, -0.6, 0.5, -0.3, -0.8]
    states = [decider.update(s) for s in sequence]

    assert states[-1][0] == "ALERT"
    assert states[-1][1] == 4
    assert decider.latched is True


@pytest.mark.unit
def test_decider_genuine_after_alert_still_alert() -> None:
    """Case 5: Genuine scores after ALERT still report ALERT (latch persistence)."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    # Trigger ALERT
    for s in [-0.5, -0.6, -0.7, -0.8]:
        decider.update(s)

    assert decider.latched is True

    # Feed genuine scores
    for genuine_score in [1.0, 1.5, 2.0, 0.9, 0.8, 1.2]:
        level, _, _ = decider.update(genuine_score)
        assert level == "ALERT"


@pytest.mark.unit
def test_decider_reset_clears_latch() -> None:
    """Case 6: reset() clears the latch and history."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    for s in [-0.5, -0.6, -0.7, -0.8]:
        decider.update(s)
    assert decider.latched is True

    decider.reset()
    assert decider.latched is False
    assert decider.flagged == 0

    level, flagged, _ = decider.update(0.8)
    assert level == "OK"
    assert flagged == 0


@pytest.mark.unit
def test_decider_nan_score_ignored() -> None:
    """Case 8: NaN score is treated as 'no information' (not appended to history)."""
    decider = AlertDecider(thr=0.0, k=4, n=6)
    decider.update(0.8)
    decider.update(-0.5)
    assert decider.flagged == 1

    # Feed NaN
    level, flagged, reason = decider.update(float("nan"))
    assert decider.flagged == 1
    assert flagged == 1
    assert level == "OK"
    assert reason == "nan_score"

    # Feed math.nan
    level2, flagged2, reason2 = decider.update(math.nan)
    assert decider.flagged == 1
    assert flagged2 == 1
    assert reason2 == "nan_score"
