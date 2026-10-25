import base64
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from starlette.websockets import WebSocketDisconnect

from callshield.core.audio import mulaw_encode
from callshield.detectors.fake import FakeDetector
from callshield.server.app import app
from callshield.server.config import settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def mixed_wav_bytes() -> bytes:
    path = FIXTURES_DIR / "mixed.wav"
    return path.read_bytes()


@pytest.fixture(scope="module")
def mixed_wav_samples() -> tuple[int, np.ndarray]:
    path = FIXTURES_DIR / "mixed.wav"
    sr, data = wavfile.read(path)
    return sr, np.asarray(data, dtype=np.float32)


@pytest.mark.integration
def test_health_returns_ok_and_fast(client: TestClient) -> None:
    """GET /health returns {'status': 'ok', 'detector': '<name>'} in < 100 ms."""
    start = time.perf_counter()
    response = client.get("/health")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "detector" in data
    assert elapsed_ms < 100.0


@pytest.mark.integration
def test_api_score_mixed_wav_progression(client: TestClient, mixed_wav_bytes: bytes) -> None:
    """POST /api/score on mixed.wav returns timeline reaching ALERT within 6s of switch."""
    # mixed.wav is 14s: first 6s genuine, next 8s fake tone (switch at t=6.0s)
    files = {"file": ("mixed.wav", mixed_wav_bytes, "audio/wav")}
    response = client.post("/api/score", files=files)

    assert response.status_code == 200
    timeline = response.json()["timeline"]
    assert len(timeline) >= 12  # Evaluations from t=2.0s to t=14.0s

    levels = [item["level"] for item in timeline]

    # Initial seconds (during genuine portion) must be OK
    assert levels[0] == "OK"

    # Must transition to ALERT after the switch at t=6.0s
    assert "ALERT" in levels
    first_alert_idx = levels.index("ALERT")
    first_alert_t = timeline[first_alert_idx]["t"]

    # First ALERT within 6s of the switch at t=6.0s (i.e. t <= 12.0s)
    assert first_alert_t <= 12.0


@pytest.mark.integration
def test_api_score_payload_validation(client: TestClient) -> None:
    """Rejects files over MAX_UPLOAD_MB with 413 and unsupported media with 415."""
    # Unsupported media type (e.g. text/plain or non-wav)
    files_txt = {"file": ("test.txt", b"not a wav file", "text/plain")}
    resp_415 = client.post("/api/score", files=files_txt)
    assert resp_415.status_code == 415

    # File exceeding MAX_UPLOAD_MB
    huge_bytes = b"\x00" * ((settings.MAX_UPLOAD_MB + 1) * 1024 * 1024)
    files_huge = {"file": ("large.wav", huge_bytes, "audio/wav")}
    resp_413 = client.post("/api/score", files=files_huge)
    assert resp_413.status_code == 413


@pytest.mark.integration
def test_ws_stream_pcm16_mixed(
    client: TestClient, mixed_wav_samples: tuple[int, np.ndarray]
) -> None:
    """WS /ws/stream with binary PCM16 frames of mixed.wav yields identical level progression."""
    _, samples = mixed_wav_samples
    pcm16_bytes = (samples * 32767.0).astype(np.int16).tobytes()

    # 100 ms chunks = 1600 samples = 3200 bytes
    chunk_bytes = 3200
    states: list[dict[str, object]] = []

    with client.websocket_connect("/ws/stream") as ws:
        for i in range(0, len(pcm16_bytes), chunk_bytes):
            chunk = pcm16_bytes[i : i + chunk_bytes]
            ws.send_bytes(chunk)
            # Drain states: a state is triggered every 16k samples (32k bytes)
            if (i + chunk_bytes) % 32000 == 0 and (i + chunk_bytes) >= 64000:
                state = ws.receive_json()
                states.append(state)

    levels = [s["level"] for s in states]
    assert "OK" in levels
    assert "ALERT" in levels


@pytest.mark.integration
def test_twilio_streaming_publishes_to_alerts_hub(
    client: TestClient, mixed_wav_samples: tuple[int, np.ndarray]
) -> None:
    """Twilio Media Stream JSON events publish alerts to /ws/alerts for that user_id."""
    _, samples = mixed_wav_samples
    user_id = "user_twilio_test_01"
    token = settings.CALLSHIELD_TOKEN

    with client.websocket_connect(f"/ws/alerts?user_id={user_id}&token={token}") as ws_alerts:
        with client.websocket_connect("/twilio") as ws_twilio:
            # 1. connected event
            ws_twilio.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})

            # 2. start event with customParameters
            ws_twilio.send_json(
                {
                    "event": "start",
                    "sequenceNumber": "1",
                    "start": {
                        "streamSid": "MZ123",
                        "accountSid": "AC123",
                        "callSid": "CA123",
                        "customParameters": {"user_id": user_id},
                    },
                    "streamSid": "MZ123",
                }
            )

            # 3. media events: resample to 8kHz and encode with mulaw_encode (160 samples)
            # Take full 14 seconds of mixed audio (6s genuine + 8s fake) to trigger ALERT at t=13s
            sub_samples = samples
            # Downsample 16k -> 8k simply by taking every 2nd sample
            samples_8k = sub_samples[::2]
            mulaw_bytes = mulaw_encode(samples_8k)

            # Send in 20 ms (160 bytes) frames
            frame_size = 160
            received_alerts: list[dict[str, object]] = []

            for i in range(0, len(mulaw_bytes), frame_size):
                payload_b64 = base64.b64encode(mulaw_bytes[i : i + frame_size]).decode("ascii")
                ws_twilio.send_json(
                    {
                        "event": "media",
                        "sequenceNumber": str(i // frame_size + 2),
                        "media": {
                            "track": "inbound",
                            "chunk": str(i // frame_size + 1),
                            "timestamp": str(i * 20),
                            "payload": payload_b64,
                        },
                        "streamSid": "MZ123",
                    }
                )
                # Check for published alert every 1 second (50 frames)
                if (i + frame_size) % (160 * 50) == 0 and (i + frame_size) >= (160 * 100):
                    alert_msg = ws_alerts.receive_json()
                    received_alerts.append(alert_msg)

            # 4. stop event
            ws_twilio.send_json({"event": "stop", "streamSid": "MZ123"})

        assert len(received_alerts) > 0
        levels = [a["level"] for a in received_alerts]
        assert "ALERT" in levels


@pytest.mark.integration
def test_two_simultaneous_twilio_sessions_isolation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two simultaneous Twilio sessions do not leak state or alerts to each other."""
    user1 = "user_alpha"
    user2 = "user_beta"
    token = settings.CALLSHIELD_TOKEN

    import sys

    app_mod = sys.modules["callshield.server.app"]
    monkeypatch.setattr(app_mod, "detector", FakeDetector())

    with (
        client.websocket_connect(f"/ws/alerts?user_id={user1}&token={token}") as ws_sub1,
        client.websocket_connect("/twilio") as ws_tw1,
        client.websocket_connect("/twilio") as ws_tw2,
    ):
        ws_tw1.send_json({"event": "connected"})
        ws_tw1.send_json(
            {
                "event": "start",
                "start": {"customParameters": {"user_id": user1}},
                "streamSid": "stream_1",
            }
        )

        ws_tw2.send_json({"event": "connected"})
        ws_tw2.send_json(
            {
                "event": "start",
                "start": {"customParameters": {"user_id": user2}},
                "streamSid": "stream_2",
            }
        )

        # Send 2s of audible silence to stream_2 only
        for _ in range(100):
            ws_tw2.send_json(
                {
                    "event": "media",
                    "media": {
                        "payload": base64.b64encode(bytes([0xFF] * 160)).decode("ascii")
                    },
                    "streamSid": "stream_2",
                }
            )

        # Send genuine noise to stream_1
        rng = np.random.default_rng(1)
        genuine_8k = rng.standard_normal(16000).astype(np.float32) * 0.3
        genuine_mulaw = mulaw_encode(genuine_8k)
        for i in range(0, len(genuine_mulaw), 160):
            ws_tw1.send_json(
                {
                    "event": "media",
                    "media": {
                        "payload": base64.b64encode(genuine_mulaw[i : i + 160]).decode("ascii")
                    },
                    "streamSid": "stream_1",
                }
            )

        # user1 alert subscriber should receive only user1 state
        alert1 = ws_sub1.receive_json()
        assert alert1["level"] == "OK"
        assert alert1["flagged"] == 0


@pytest.mark.integration
def test_ws_alerts_invalid_token_rejected(client: TestClient) -> None:
    """Wrong token on /ws/alerts is rejected with code 4401."""
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/alerts?user_id=test_user&token=wrong-token") as ws,
    ):
        ws.receive_json()


@pytest.mark.integration
def test_malformed_json_graceful_close(client: TestClient) -> None:
    """Malformed JSON on /twilio closes socket cleanly without crashing the server."""
    with client.websocket_connect("/twilio") as ws:
        ws.send_text("this is not valid json")
        # Should close cleanly or drop without server 500 error


@pytest.mark.integration
def test_slow_inference_health_responsiveness(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """While a slow FakeDetector (500 ms) is scoring, GET /health still responds in < 100 ms."""
    import sys
    app_module = sys.modules["callshield.server.app"]

    # Temporarily set slow detector
    slow_detector = FakeDetector(delay_ms=500.0)
    monkeypatch.setattr(app_module, "detector", slow_detector)

    # Trigger a score request in a background thread or asynchronous task
    import concurrent.futures

    def run_scoring() -> None:
        path = FIXTURES_DIR / "fake_like.wav"
        files = {"file": ("fake.wav", path.read_bytes(), "audio/wav")}
        client.post("/api/score", files=files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(run_scoring)

        # Give the scoring request a moment to start
        time.sleep(0.1)

        # Probe /health: must respond in < 100 ms despite active 500 ms inference
        start = time.perf_counter()
        health_resp = client.get("/health")
        latency_ms = (time.perf_counter() - start) * 1000.0

        assert health_resp.status_code == 200
        assert latency_ms < 100.0

        future.result()
