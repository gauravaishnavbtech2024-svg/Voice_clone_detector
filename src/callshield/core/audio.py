"""Pure audio digital signal processing functions for Call Shield.

No I/O or external dependencies beyond numpy and scipy.
Conventions:
  - Waveforms: mono float32 in [-1.0, 1.0]
  - Model sampling rate: 16,000 Hz
  - Telephony sampling rate: 8,000 Hz (G.711 µ-law)
  - Target RMS level: -26 dBFS
"""

import numpy as np
from scipy.signal import resample_poly

# Precompute 256-element G.711 µ-law lookup table (ITU-T Recommendation G.711 Table 2a)
_MULAW_DECODE_TABLE: np.ndarray = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _c = _b ^ 0xFF
    _s = (_c >> 7) & 1
    _e = (_c >> 4) & 7
    _q = _c & 0xF
    _mag = ((2 * _q + 33) * (1 << _e) - 33) * 4
    _val = -_mag if _s == 1 else _mag
    _MULAW_DECODE_TABLE[_b] = np.float32(_val / 32768.0)

# Precompute segment decision thresholds for linear PCM to µ-law mapping
# Based on ITU-T G.711 Table 2a linear decision levels
_SEG_END: tuple[int, ...] = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _linear_to_mulaw_sample(sample: float) -> int:
    """Encode a single float32 sample in [-1.0, 1.0] to an 8-bit G.711 µ-law byte."""
    # Scale to 16-bit linear PCM [-32768, 32767]
    pcm = int(round(sample * 32768.0))
    pcm = max(-32768, min(32767, pcm))

    # Sign bit
    if pcm < 0:
        sign = 0x80
        pcm = -pcm
    else:
        sign = 0x00

    # Clip to maximum magnitude representable in G.711 (32635)
    pcm = min(pcm, 32635)

    # Convert 16-bit to 14-bit scale (divide by 4) and add bias 33
    biased = (pcm >> 2) + 33

    # Find segment (chord)
    seg = 0
    while seg < 7 and biased > _SEG_END[seg]:
        seg += 1

    # Extract 4-bit quantization step
    step = (biased >> (seg + 1)) & 0x0F

    # Assemble codeword and invert all bits per G.711 specification
    codeword = (sign | (seg << 4) | step) ^ 0xFF
    return codeword & 0xFF


def mulaw_decode(data: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decodes 8-bit ITU-T G.711 µ-law bytes into float32 samples in [-1.0, 1.0].

    G.711 properties:
      - 0xFF -> 0.0 (+0)
      - 0x7F -> 0.0 (-0)
      - 0x80 -> +32124 / 32768 ≈ +0.9803467
      - 0x00 -> -32124 / 32768 ≈ -0.9803467
    """
    if isinstance(data, (bytes, bytearray)):
        indices = np.frombuffer(data, dtype=np.uint8)
    else:
        indices = np.asarray(data, dtype=np.uint8)

    return _MULAW_DECODE_TABLE[indices].copy()


def mulaw_encode(x: np.ndarray) -> bytes:
    """Encodes float32 samples in [-1.0, 1.0] into 8-bit ITU-T G.711 µ-law bytes.

    Used by the call simulator to stream standard telephony audio.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    # Clip input to valid range
    clipped = np.clip(x_arr, -1.0, 1.0)
    encoded = [_linear_to_mulaw_sample(float(val)) for val in clipped]
    return bytes(encoded)


def resample_8k_to_16k(x: np.ndarray) -> np.ndarray:
    """Upsamples audio from 8,000 Hz to 16,000 Hz using polyphase FIR filtering.

    Output length is exactly 2 x input length.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if len(x_arr) == 0:
        return np.empty(0, dtype=np.float32)

    resampled = resample_poly(x_arr, up=2, down=1)
    return np.asarray(resampled, dtype=np.float32)


def rms_normalize(x: np.ndarray, target_dbfs: float = -26.0) -> np.ndarray:
    """Normalizes waveform to target dBFS RMS level.

    For all-zero or near-silent inputs, returns zeros without producing NaN.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if len(x_arr) == 0:
        return np.empty(0, dtype=np.float32)

    mean_sq = float(np.mean(x_arr**2))
    if mean_sq <= 1e-18:
        return np.zeros_like(x_arr)

    rms = float(np.sqrt(mean_sq))
    target_rms = float(10.0 ** (target_dbfs / 20.0))
    gain = float(target_rms / rms)
    return np.asarray(x_arr * gain, dtype=np.float32)


def crop_or_pad(x: np.ndarray, n: int) -> np.ndarray:
    """Ensures waveform length is exactly n samples.

    - Shorter inputs are repeat-padded (never zero-padded).
    - Longer inputs are cropped to length n.
    - Exact-length inputs are returned unchanged.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    curr_len = len(x_arr)

    if curr_len == n:
        return x_arr.copy()

    if curr_len > n:
        return x_arr[:n].copy()

    if curr_len == 0:
        return np.zeros(n, dtype=np.float32)

    reps = int(np.ceil(n / curr_len))
    tiled = np.tile(x_arr, reps)
    return tiled[:n].astype(np.float32)


def is_silent(x: np.ndarray, threshold_dbfs: float = -50.0) -> bool:
    """Returns True if the waveform's RMS is below threshold_dbfs."""
    x_arr = np.asarray(x, dtype=np.float32)
    if len(x_arr) == 0:
        return True

    mean_sq = float(np.mean(x_arr**2))
    if mean_sq <= 1e-18:
        return True

    rms = np.sqrt(mean_sq)
    dbfs = 20.0 * np.log10(rms)
    return bool(dbfs < threshold_dbfs)
