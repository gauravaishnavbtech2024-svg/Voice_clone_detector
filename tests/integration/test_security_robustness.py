"""Integration tests for Phase 7: Security, Privacy, and Robustness."""

import base64
import logging
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from callshield.server.app import active_sessions, app
from callshield.server.config import ServerSettings, settings


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def _get_directory_snapshot(directory: str | Path) -> set[Path]:
    """Returns a set of relative file paths present in the directory."""
    path = Path(directory)
    if not path.is_dir():
        return set()
    ignored_dirs = {".pytest_cache", ".git", "training_dataset", "dataset", ".venv", "__pycache__"}
    files = set()
    for p in path.glob("*"):
        if p.is_file():
            files.add(p.relative_to(path))
        elif p.is_dir() and p.name not in ignored_dirs:
            for sub in p.glob("*"):
                if sub.is_file():
                    files.add(sub.relative_to(path))
    return files


@pytest.mark.integration
def test_zero_disk_persistence(client: TestClient) -> None:
    """Verifies that processing audio streams never writes audio or state to disk or temp dir."""
    repo_root = Path(__file__).resolve().parents[2]
    temp_dir = Path(tempfile.gettempdir())

    repo_files_before = _get_directory_snapshot(repo_root)
    temp_files_before = {p for p in temp_dir.glob("callshield*")}

    # Execute a full Twilio call session with simulated media
    dummy_audio = b"\xff\x7f" * 160  # 160 bytes of µ-law audio
    payload_b64 = base64.b64encode(dummy_audio).decode("ascii")

    with client.websocket_connect("/twilio") as ws:
        ws.send_json(
            {"event": "start", "start": {"customParameters": {"user_id": "test_privacy_user"}}}
        )
        for _ in range(5):
            ws.send_json({"event": "media", "media": {"payload": payload_b64}})
        ws.send_json({"event": "stop"})

    repo_files_after = _get_directory_snapshot(repo_root)
    temp_files_after = {p for p in temp_dir.glob("callshield*")}

    new_repo_files = repo_files_after - repo_files_before
    new_temp_files = temp_files_after - temp_files_before

    # Filter out any non-callshield temp files that other OS processes might have created
    callshield_temp_files = {p for p in new_temp_files if "callshield" in str(p).lower()}

    assert not new_repo_files, f"Unexpected files persisted in repo: {new_repo_files}"
    assert not callshield_temp_files, f"Audio files persisted in temp dir: {callshield_temp_files}"


@pytest.mark.integration
def test_logging_never_leaks_audio_payloads_or_tokens(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensures logs never contain raw audio waveforms, base64 payload strings, or tokens."""
    caplog.set_level(logging.DEBUG)

    sensitive_payload = b"SUPER_SENSITIVE_AUDIO_STREAM_TEST_MARKER"
    payload_b64 = base64.b64encode(sensitive_payload).decode("ascii")

    # 1. Connect and stream over Twilio
    with client.websocket_connect("/twilio") as ws:
        ws.send_json(
            {"event": "start", "start": {"customParameters": {"user_id": "user_logging_test"}}}
        )
        ws.send_json({"event": "media", "media": {"payload": payload_b64}})
        ws.send_json({"event": "stop"})

    # 2. Connect to alerts with secret token
    token = settings.CALLSHIELD_TOKEN
    with client.websocket_connect(
        f"/ws/alerts?user_id=user_logging_test&token={token}"
    ) as ws_alerts:
        ws_alerts.send_text("ping")

    log_output = caplog.text

    assert payload_b64 not in log_output, "Base64 audio payload was leaked in server logs!"
    assert (
        sensitive_payload.decode("ascii") not in log_output
    ), "Raw audio string was leaked in server logs!"
    assert token not in log_output, "Authentication token was leaked in server logs!"


@pytest.mark.integration
def test_ws_stream_frame_exceeding_max_bytes_closed_with_policy_violation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PCM16 frame larger than MAX_FRAME_BYTES is closed with policy violation (1008)."""
    monkeypatch.setattr(settings, "MAX_FRAME_BYTES", 1024)

    with client.websocket_connect("/ws/stream") as ws:
        oversized_frame = b"\x00" * 2048
        ws.send_bytes(oversized_frame)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()

        assert exc_info.value.code == 1008


@pytest.mark.integration
def test_twilio_frame_exceeding_max_bytes_closed_with_policy_violation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Twilio media payload exceeding MAX_FRAME_BYTES is closed with policy violation (1008)."""
    monkeypatch.setattr(settings, "MAX_FRAME_BYTES", 1024)

    with client.websocket_connect("/twilio") as ws:
        ws.send_json({"event": "start", "start": {"customParameters": {"user_id": "test_user"}}})

        oversized_audio = b"\xff" * 2048
        payload_b64 = base64.b64encode(oversized_audio).decode("ascii")

        ws.send_json({"event": "media", "media": {"payload": payload_b64}})

        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()

        assert exc_info.value.code == 1008


@pytest.mark.integration
def test_session_exceeding_max_duration_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session exceeding MAX_SESSION_SECONDS is closed with policy violation (1008)."""
    monkeypatch.setattr(settings, "MAX_SESSION_SECONDS", 0.1)

    with client.websocket_connect("/ws/stream") as ws:
        time.sleep(0.15)
        # Send a normal frame after session expired
        ws.send_bytes(b"\x00\x00" * 1600)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()

        assert exc_info.value.code == 1008


@pytest.mark.integration
def test_concurrent_sessions_limit_rejects_excess(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When active sessions reach MAX_SESSIONS, excess connection is rejected."""
    monkeypatch.setattr(settings, "MAX_SESSIONS", 2)

    with (
        client.websocket_connect("/ws/stream") as _ws1,
        client.websocket_connect("/ws/stream") as _ws2,
    ):
        # 3rd connection exceeds MAX_SESSIONS=2
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/ws/stream") as ws3,
        ):
            ws3.receive_json()

        assert exc_info.value.code in (1013, 1008)


@pytest.mark.integration
def test_config_environment_overrides_and_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies configuration reads from env vars and provides secure defaults."""
    # Test safe default configuration
    default_cfg = ServerSettings()
    assert default_cfg.MAX_UPLOAD_MB > 0
    assert default_cfg.MAX_FRAME_BYTES >= 1024
    assert default_cfg.MAX_SESSION_SECONDS >= 60.0
    assert default_cfg.MAX_SESSIONS >= 1

    # Test override through environment variables
    monkeypatch.setenv("MAX_FRAME_BYTES", "4096")
    monkeypatch.setenv("MAX_SESSION_SECONDS", "1800")
    monkeypatch.setenv("MAX_SESSIONS", "50")
    monkeypatch.setenv("CALLSHIELD_TOKEN", "custom-secret-token")

    env_cfg = ServerSettings()
    assert env_cfg.MAX_FRAME_BYTES == 4096
    assert env_cfg.MAX_SESSION_SECONDS == 1800.0
    assert env_cfg.MAX_SESSIONS == 50
    assert env_cfg.CALLSHIELD_TOKEN == "custom-secret-token"


@pytest.mark.integration
def test_graceful_shutdown_closes_active_connections() -> None:
    """Verifies that lifespan shutdown closes active connections gracefully."""
    with TestClient(app) as test_client, test_client.websocket_connect("/ws/stream") as ws:
        assert len(active_sessions) >= 1
        ws.send_bytes(b"\x00\x00" * 160)
        # Exiting TestClient triggers lifespan shutdown
    assert len(active_sessions) == 0
