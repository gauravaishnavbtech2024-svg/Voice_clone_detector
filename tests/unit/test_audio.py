import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from callshield.core.audio import (
    crop_or_pad,
    is_silent,
    mulaw_decode,
    mulaw_encode,
    resample_8k_to_16k,
    rms_normalize,
)


def _compute_itu_g711_reference_table() -> np.ndarray:
    """Computes the 256-element ITU-T Recommendation G.711 Table 2a reference table.

    ITU-T G.711 µ-law defines:
      - 8-bit codeword b is transmitted with all bits inverted: c = b ^ 0xFF
      - Bit 7 is sign bit s (0 = positive, 1 = negative)
      - Bits 6-4 are exponent/segment e (0..7)
      - Bits 3-0 are mantissa/step q (0..15)
      - Reconstruction magnitude: y = ((2 * q + 33) * (2 ** e) - 33) * 4
      - Normalized to 16-bit fullscale [-32768, 32768]: float_val = (+/- y) / 32768.0
    """
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        c = b ^ 0xFF
        s = (c >> 7) & 1
        e = (c >> 4) & 7
        q = c & 0xF
        mag = ((2 * q + 33) * (1 << e) - 33) * 4
        val = -mag if s == 1 else mag
        table[b] = np.float32(val / 32768.0)
    return table


@pytest.mark.unit
def test_mulaw_decode_special_values() -> None:
    """Verify G.711 zeros and maximum magnitude values."""
    data = bytes([0xFF, 0x7F, 0x80, 0x00])
    decoded = mulaw_decode(data)

    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.float32
    assert len(decoded) == 4

    # The two G.711 zeros: 0xFF (+0) and 0x7F (-0)
    assert decoded[0] == 0.0
    assert decoded[1] == 0.0

    # Max magnitudes (+/- 32124 / 32768 ≈ 0.9803467)
    expected_max = 32124.0 / 32768.0
    assert pytest.approx(decoded[2], rel=1e-5) == expected_max
    assert pytest.approx(decoded[3], rel=1e-5) == -expected_max


@pytest.mark.unit
def test_mulaw_decode_itu_g711_reference_table() -> None:
    """Verify all 256 values of mulaw_decode against the standard ITU-T G.711 table."""
    all_bytes = bytes(range(256))
    decoded = mulaw_decode(all_bytes)
    reference = _compute_itu_g711_reference_table()

    assert decoded.shape == (256,)
    assert decoded.dtype == np.float32
    assert np.all(decoded >= -1.0) and np.all(decoded <= 1.0)
    np.testing.assert_allclose(decoded, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
@given(
    samples=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=50)
def test_mulaw_quantization_error_hypothesis(samples: list[float]) -> None:
    """Property test: decode(encode(x)) is within G.711 quantization error bounds."""
    x = np.array(samples, dtype=np.float32)
    encoded = mulaw_encode(x)
    decoded = mulaw_decode(encoded)

    # In G.711, maximum quantization step size occurs in chord 7 (1024 / 32768 ≈ 0.03125)
    # The reconstruction error |x - decoded| should not exceed half of max step + epsilon (~0.02)
    diff = np.abs(x - decoded)
    assert np.all(diff <= 0.035)


@pytest.mark.unit
def test_mulaw_byte_roundtrip() -> None:
    """Property test: encode(decode(b)) == b for all bytes except 0x7F (which maps to 0xFF)."""
    for b in range(256):
        decoded = mulaw_decode(bytes([b]))
        encoded = mulaw_encode(decoded)
        assert len(encoded) == 1
        if b == 0x7F:
            assert encoded[0] == 0xFF
        else:
            assert encoded[0] == b


@pytest.mark.unit
def test_resample_8k_to_16k_length_and_spectrum() -> None:
    """Output length is 2 x input; a 440 Hz sine keeps its dominant frequency within 5 Hz."""
    duration_s = 0.5
    fs_in = 8000
    t = np.arange(int(duration_s * fs_in)) / fs_in
    freq = 440.0
    x = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)

    y = resample_8k_to_16k(x)

    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert len(y) == 2 * len(x)
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()

    # FFT frequency analysis at 16 kHz
    fs_out = 16000
    fft_vals = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs_out)
    peak_freq = freqs[np.argmax(fft_vals)]

    assert abs(peak_freq - freq) <= 5.0

    # Determinism
    y2 = resample_8k_to_16k(x)
    np.testing.assert_array_equal(y, y2)

    # Empty array
    empty_out = resample_8k_to_16k(np.array([], dtype=np.float32))
    assert len(empty_out) == 0
    assert empty_out.dtype == np.float32


@pytest.mark.unit
def test_rms_normalize_target_and_gain_invariance() -> None:
    """RMS is -26 dBFS (+/-0.1 dB); zeros return zeros; gain invariance holds."""
    rng = np.random.default_rng(42)

    # Test several initial amplitudes
    for scale in [0.01, 0.1, 0.5, 0.9]:
        noise = rng.standard_normal(16000).astype(np.float32) * scale
        norm = rms_normalize(noise, target_dbfs=-26.0)

        rms = np.sqrt(np.mean(norm**2))
        dbfs = 20.0 * np.log10(rms)
        assert pytest.approx(dbfs, abs=0.1) == -26.0

    # All-zero array returns all zeros without NaN
    zeros = np.zeros(1000, dtype=np.float32)
    norm_zeros = rms_normalize(zeros)
    assert np.all(norm_zeros == 0.0)
    assert not np.isnan(norm_zeros).any()

    # Gain invariance for g in [0.01, 10.0]
    base_signal = rng.standard_normal(4000).astype(np.float32) * 0.1
    ref_norm = rms_normalize(base_signal, target_dbfs=-26.0)

    for g in [0.01, 0.1, 0.5, 2.0, 10.0]:
        scaled_signal = base_signal * g
        scaled_norm = rms_normalize(scaled_signal, target_dbfs=-26.0)
        np.testing.assert_allclose(scaled_norm, ref_norm, rtol=1e-4, atol=1e-5)


@pytest.mark.unit
def test_crop_or_pad_behavior() -> None:
    """Short inputs repeat-padded (tail repeats start, non-zero);
    long inputs cropped; exact unchanged.
    """
    # Shorter input -> repeat-padded
    short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target_len = 8
    padded = crop_or_pad(short, target_len)

    assert len(padded) == target_len
    # Expected tiling: [1, 2, 3, 1, 2, 3, 1, 2]
    np.testing.assert_array_equal(padded, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
    # Assert tail equals repeat of the start
    assert padded[3] == short[0]
    assert padded[4] == short[1]
    # Assert padded region is not all zeros
    assert not np.all(padded[len(short) :] == 0.0)

    # Longer input -> cropped
    long_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    cropped = crop_or_pad(long_arr, 3)
    assert len(cropped) == 3
    np.testing.assert_array_equal(cropped, [1.0, 2.0, 3.0])

    # Exact-length input -> returned unchanged
    exact = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = crop_or_pad(exact, 3)
    assert len(result) == 3
    np.testing.assert_array_equal(result, exact)


@pytest.mark.unit
def test_is_silent_classification() -> None:
    """True for zeros and very quiet noise; False for normal-level sine."""
    # All zeros -> True
    zeros = np.zeros(16000, dtype=np.float32)
    assert is_silent(zeros, threshold_dbfs=-50.0) is True

    # Quiet noise at -60 dBFS -> True
    rng = np.random.default_rng(123)
    quiet = rng.standard_normal(16000).astype(np.float32)
    # Target -60 dBFS
    target_rms = 10.0 ** (-60.0 / 20.0)
    current_rms = np.sqrt(np.mean(quiet**2))
    quiet = quiet * (target_rms / current_rms)
    assert is_silent(quiet, threshold_dbfs=-50.0) is True

    # Normal sine at -20 dBFS -> False
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    target_rms_sine = 10.0 ** (-20.0 / 20.0)
    sine = sine * (target_rms_sine / np.sqrt(np.mean(sine**2)))
    assert is_silent(sine, threshold_dbfs=-50.0) is False


@pytest.mark.unit
def test_audio_edge_cases() -> None:
    """Cover edge cases: numpy array input to mulaw_decode, empty arrays in rms/crop/silent."""
    # mulaw_decode with numpy array input
    arr_in = np.array([0xFF, 0x00], dtype=np.uint8)
    dec = mulaw_decode(arr_in)
    assert dec[0] == 0.0
    assert dec[1] < -0.9

    # rms_normalize empty array
    empty = np.array([], dtype=np.float32)
    assert len(rms_normalize(empty)) == 0

    # crop_or_pad empty array
    assert len(crop_or_pad(empty, 10)) == 10
    assert np.all(crop_or_pad(empty, 10) == 0.0)

    # is_silent empty array and near-zero array
    assert is_silent(empty) is True
    near_zero = np.full(100, 1e-12, dtype=np.float32)
    assert is_silent(near_zero) is True

