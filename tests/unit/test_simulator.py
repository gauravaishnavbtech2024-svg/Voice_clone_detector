"""Unit tests for Call Shield Twilio call simulator framing and audio prep."""

import base64
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from callshield.core.audio import mulaw_decode
from tools.simulate_call import (
    generate_twilio_events,
    load_and_prepare_audio,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.unit
def test_load_and_prepare_audio_16k_to_8k(tmp_path: Path) -> None:
    """16 kHz WAV is converted to mono, downsampled to 8 kHz, and mulaw encoded."""
    sr_16k = 16000
    # 1.0 second of 16 kHz audio (16,000 samples)
    t = np.arange(sr_16k) / sr_16k
    audio_16k = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav_file = tmp_path / "test_16k.wav"
    wavfile.write(wav_file, sr_16k, audio_16k)

    mulaw_bytes = load_and_prepare_audio(wav_file)

    # 1.0 second at 8 kHz is exactly 8,000 bytes of µ-law
    assert len(mulaw_bytes) == 8000
    decoded_samples = mulaw_decode(mulaw_bytes)
    assert len(decoded_samples) == 8000
    assert np.all(np.abs(decoded_samples) <= 1.0)


@pytest.mark.unit
def test_load_and_prepare_audio_stereo_and_int16(tmp_path: Path) -> None:
    """Stereo int16 WAV is converted to mono float32 in [-1, 1] before encoding."""
    sr_8k = 8000
    num_samples = 4000  # 0.5 seconds
    # Stereo int16
    stereo_int16 = np.zeros((num_samples, 2), dtype=np.int16)
    stereo_int16[:, 0] = 16000
    stereo_int16[:, 1] = -16000
    wav_file = tmp_path / "test_stereo.wav"
    wavfile.write(wav_file, sr_8k, stereo_int16)

    mulaw_bytes = load_and_prepare_audio(wav_file)
    assert len(mulaw_bytes) == 4000


@pytest.mark.unit
def test_generate_twilio_events_order_and_structure() -> None:
    """Twilio event generator outputs: connected -> start -> media... -> stop."""
    # 400 ms of audio at 8 kHz = 3200 samples = exactly 20 frames of 160 samples
    mulaw_bytes = bytes([0xFF] * 3200)
    user_id = "test_user_order"
    stream_sid = "MZ_UNIT_TEST"

    events = list(generate_twilio_events(mulaw_bytes, stream_sid=stream_sid, user_id=user_id))

    # Total events: 1 (connected) + 1 (start) + 20 (media) + 1 (stop) = 23 events
    assert len(events) == 23

    # 1. connected
    assert events[0]["event"] == "connected"
    assert events[0]["protocol"] == "Call"

    # 2. start
    assert events[1]["event"] == "start"
    assert events[1]["streamSid"] == stream_sid
    assert events[1]["start"]["customParameters"]["user_id"] == user_id

    # 3. media events
    for idx, event in enumerate(events[2:-1]):
        assert event["event"] == "media"
        assert event["streamSid"] == stream_sid
        assert event["sequenceNumber"] == str(idx + 2)

        media = event["media"]
        assert media["track"] == "inbound"
        assert media["chunk"] == str(idx + 1)
        assert media["timestamp"] == str(idx * 20)

        # Payload is valid base64 decoding to 160 bytes
        payload_bytes = base64.b64decode(media["payload"])
        assert len(payload_bytes) == 160

    # 4. stop
    assert events[-1]["event"] == "stop"
    assert events[-1]["streamSid"] == stream_sid
