import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from callshield.core.buffer import RollingBuffer


@pytest.mark.unit
def test_rolling_buffer_invalid_capacity() -> None:
    """Raises ValueError if capacity_samples <= 0."""
    with pytest.raises(ValueError):
        RollingBuffer(capacity_samples=0)
    with pytest.raises(ValueError):
        RollingBuffer(capacity_samples=-10)


@pytest.mark.unit
def test_rolling_buffer_basic_append_and_latest() -> None:
    """Buffer appends samples and returns latest n samples."""
    buf = RollingBuffer(capacity_samples=100)
    assert len(buf) == 0

    samples1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf.append(samples1)
    assert len(buf) == 3

    latest2 = buf.latest(2)
    assert len(latest2) == 2
    np.testing.assert_array_equal(latest2, [2.0, 3.0])

    latest5 = buf.latest(5)
    # Fewer than 5 buffered; returns all currently buffered
    assert len(latest5) == 3
    np.testing.assert_array_equal(latest5, [1.0, 2.0, 3.0])


@pytest.mark.unit
def test_rolling_buffer_capacity_limit() -> None:
    """Buffer never exceeds capacity and drops oldest samples."""
    buf = RollingBuffer(capacity_samples=5)
    buf.append(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert len(buf) == 4

    # Add 3 more samples -> total 7 > capacity 5 -> oldest 2 dropped
    buf.append(np.array([5.0, 6.0, 7.0], dtype=np.float32))
    assert len(buf) == 5
    np.testing.assert_array_equal(buf.latest(5), [3.0, 4.0, 5.0, 6.0, 7.0])

    # Add huge chunk exceeding capacity directly
    buf.append(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32))
    assert len(buf) == 5
    np.testing.assert_array_equal(buf.latest(5), [30.0, 40.0, 50.0, 60.0, 70.0])


@pytest.mark.unit
def test_rolling_buffer_latest_edge_cases() -> None:
    """latest(0) and negative n return empty float32 array."""
    buf = RollingBuffer(capacity_samples=10)
    buf.append(np.array([1.0, 2.0], dtype=np.float32))
    assert len(buf.latest(0)) == 0
    assert len(buf.latest(-5)) == 0


@pytest.mark.unit
@given(
    chunks=st.lists(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=30,
        ),
        min_size=1,
        max_size=20,
    ),
    fetch_n=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_rolling_buffer_hypothesis_chunk_invariance(
    chunks: list[list[float]], fetch_n: int
) -> None:
    """Property test: Feeding random chunks yields identical output to slicing full history."""
    capacity = 100
    buf = RollingBuffer(capacity_samples=capacity)
    all_samples: list[float] = []

    for chunk in chunks:
        arr = np.array(chunk, dtype=np.float32)
        buf.append(arr)
        all_samples.extend(chunk)

    # Reference slicing
    expected_full = np.array(all_samples, dtype=np.float32)
    # Truncate to capacity if exceeded
    if len(expected_full) > capacity:
        expected_buffered = expected_full[-capacity:]
    else:
        expected_buffered = expected_full

    expected_latest = expected_buffered[-fetch_n:] if fetch_n > 0 else np.empty(0, dtype=np.float32)

    actual_latest = buf.latest(fetch_n)
    np.testing.assert_array_equal(actual_latest, expected_latest)
