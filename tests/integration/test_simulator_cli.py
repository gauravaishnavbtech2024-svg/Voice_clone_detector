"""Integration tests for Call Shield simulate_call CLI against a live server."""

import socket
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import uvicorn

from callshield.server.app import app

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SIMULATE_SCRIPT = Path(__file__).resolve().parent.parent.parent / "tools" / "simulate_call.py"


def get_free_port() -> int:
    """Finds an unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_server_ws_url() -> Generator[str, None, None]:
    """Starts a live Uvicorn server in a daemon thread for integration testing."""
    import sys
    from callshield.detectors.fake import FakeDetector

    app_module = sys.modules["callshield.server.app"]
    old_det = app_module.detector
    app_module.detector = FakeDetector()

    port = get_free_port()
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server liveness
    time.sleep(0.6)
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)
        app_module.detector = old_det


@pytest.mark.integration
def test_cli_mixed_wav_triggers_alert(live_server_ws_url: str) -> None:
    """Simulating mixed.wav against live server reaches ALERT level and exits with 0."""
    wav_path = FIXTURES_DIR / "mixed.wav"
    cmd = [
        sys.executable,
        str(SIMULATE_SCRIPT),
        "--wav",
        str(wav_path),
        "--url",
        f"{live_server_ws_url}/twilio",
        "--user",
        "cli_user_mixed",
        "--expect",
        "ALERT",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "ALERT" in result.stdout


@pytest.mark.integration
def test_cli_genuine_wav_matches_ok(live_server_ws_url: str) -> None:
    """Simulating genuine_like.wav against live server finishes with OK and exits with 0."""
    wav_path = FIXTURES_DIR / "genuine_like.wav"
    cmd = [
        sys.executable,
        str(SIMULATE_SCRIPT),
        "--wav",
        str(wav_path),
        "--url",
        f"{live_server_ws_url}/twilio",
        "--user",
        "cli_user_genuine",
        "--expect",
        "OK",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    assert "OK" in result.stdout


@pytest.mark.integration
def test_cli_expect_mismatch_returns_exit_code_1(live_server_ws_url: str) -> None:
    """If expected level (ALERT) does not match actual final level (OK), exit code is 1."""
    wav_path = FIXTURES_DIR / "genuine_like.wav"
    cmd = [
        sys.executable,
        str(SIMULATE_SCRIPT),
        "--wav",
        str(wav_path),
        "--url",
        f"{live_server_ws_url}/twilio",
        "--user",
        "cli_user_mismatch",
        "--expect",
        "ALERT",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
