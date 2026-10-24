"""Twilio Media Streams Call Simulator for Call Shield.

Simulates an incoming telephone call by streaming audio frames to the
Call Shield server using the official Twilio Media Streams JSON protocol
over WebSockets (8 kHz, 20 ms frames, G.711 µ-law).
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import websockets
from scipy.io import wavfile
from scipy.signal import resample_poly

from callshield.core.audio import mulaw_encode


def load_and_prepare_audio(wav_path: Path | str) -> bytes:
    """Loads a WAV file, converts to mono float32, downsamples to 8 kHz, and µ-law encodes."""
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sample_rate, raw_audio = wavfile.read(path)
    audio_arr = np.asarray(raw_audio)

    # Convert multichannel to mono (take first channel)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr[:, 0]

    # Convert to float32 in [-1.0, 1.0]
    if np.issubdtype(audio_arr.dtype, np.integer):
        max_int = float(np.iinfo(audio_arr.dtype).max)
        audio_f32 = (audio_arr.astype(np.float32) / max_int).astype(np.float32)
    else:
        audio_f32 = audio_arr.astype(np.float32)

    # Resample to 8000 Hz if needed
    if sample_rate == 16000:
        samples_8k = audio_f32[::2]
    elif sample_rate != 8000:
        samples_8k = resample_poly(audio_f32, up=8000, down=sample_rate).astype(np.float32)
    else:
        samples_8k = audio_f32

    return mulaw_encode(samples_8k)


def generate_twilio_events(
    mulaw_bytes: bytes,
    stream_sid: str = "MZ_SIM",
    user_id: str = "u1",
) -> Generator[dict[str, Any], None, None]:
    """Generates standard Twilio Media Streams JSON events for the audio stream."""
    # 1. Connected event
    yield {"event": "connected", "protocol": "Call", "version": "1.0.0"}

    # 2. Start event with user_id custom parameter
    yield {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": stream_sid,
            "accountSid": "AC_SIM",
            "callSid": "CA_SIM",
            "customParameters": {"user_id": user_id},
        },
        "streamSid": stream_sid,
    }

    # 3. Media events (160 bytes = 20 ms at 8 kHz)
    frame_size = 160
    num_frames = len(mulaw_bytes) // frame_size

    for chunk_idx in range(num_frames):
        offset = chunk_idx * frame_size
        chunk = mulaw_bytes[offset : offset + frame_size]
        payload_b64 = base64.b64encode(chunk).decode("ascii")
        timestamp_ms = chunk_idx * 20
        sequence_num = chunk_idx + 2

        yield {
            "event": "media",
            "sequenceNumber": str(sequence_num),
            "media": {
                "track": "inbound",
                "chunk": str(chunk_idx + 1),
                "timestamp": str(timestamp_ms),
                "payload": payload_b64,
            },
            "streamSid": stream_sid,
        }

    # 4. Stop event
    yield {"event": "stop", "streamSid": stream_sid}


async def run_simulator(
    twilio_url: str,
    alerts_url: str,
    mulaw_bytes: bytes,
    user_id: str = "u1",
    realtime: bool = False,
) -> str:
    """Streams Twilio media frames to twilio_url and receives live updates from alerts_url."""
    final_level = "OK"
    stop_listener = asyncio.Event()

    async def alert_listener(ws_alerts: Any) -> None:
        nonlocal final_level
        try:
            while not stop_listener.is_set():
                try:
                    message_text = await asyncio.wait_for(ws_alerts.recv(), timeout=0.2)
                    data = json.loads(message_text)
                    t = data.get("t", 0.0)
                    score = data.get("score", 0.0)
                    level = data.get("level", "OK")
                    flagged = data.get("flagged", 0)
                    reason = data.get("reason", "")
                    final_level = level
                    print(
                        f"[t={t:4.1f}s] score={score:+.2f} | "
                        f"level={level:<7} | flagged={flagged}/6 | reason={reason}"
                    )
                except TimeoutError:
                    continue
        except Exception:
            pass

    async with websockets.connect(alerts_url) as ws_alerts:
        listener_task = asyncio.create_task(alert_listener(ws_alerts))

        try:
            async with websockets.connect(twilio_url) as ws_twilio:
                events = generate_twilio_events(
                    mulaw_bytes,
                    stream_sid=f"MZ_{int(time.time())}",
                    user_id=user_id,
                )

                for event in events:
                    await ws_twilio.send(json.dumps(event))
                    if realtime and event.get("event") == "media":
                        await asyncio.sleep(0.02)

            # Give a moment for the server to finish processing the last frames
            await asyncio.sleep(1.5)
        finally:
            stop_listener.set()
            await listener_task

    return final_level


def build_alerts_url(twilio_url: str, user_id: str, token: str) -> str:
    """Derives default alerts WebSocket URL from the twilio WebSocket URL."""
    parsed = urlparse(twilio_url)
    query = f"user_id={user_id}&token={token}"
    return urlunparse((parsed.scheme, parsed.netloc, "/ws/alerts", "", query, ""))


def main() -> int:
    """CLI entrypoint for Call Shield call simulator."""
    parser = argparse.ArgumentParser(
        description="Call Shield Twilio Media Streams call simulator.",
    )
    parser.add_argument("--wav", required=True, type=Path, help="Path to input WAV file.")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/twilio",
        help="Twilio WebSocket endpoint URL (default: ws://localhost:8000/twilio).",
    )
    parser.add_argument(
        "--alerts-url",
        default=None,
        help="Alerts WebSocket endpoint URL (default: auto-derived from --url).",
    )
    parser.add_argument(
        "--user",
        default="u1",
        help="User ID to simulate (default: u1).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CALLSHIELD_TOKEN", "secret-token"),
        help="Authentication token for /ws/alerts.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Paces transmission in real time (20 ms per frame).",
    )
    parser.add_argument(
        "--expect",
        choices=["OK", "WARNING", "ALERT"],
        default=None,
        help="Assert expected final alert level. Non-zero exit code if not matched.",
    )

    args = parser.parse_args()

    # Pre-process audio
    try:
        mulaw_bytes = load_and_prepare_audio(args.wav)
    except Exception as err:
        print(f"Error loading audio file: {err}", file=sys.stderr)
        return 1

    alerts_url = args.alerts_url or build_alerts_url(args.url, args.user, args.token)

    total_s = len(mulaw_bytes) / 8000.0
    print(f"Loaded {args.wav.name} ({total_s:.1f}s audio). Streaming to {args.url} ...")

    try:
        final_level = asyncio.run(
            run_simulator(
                twilio_url=args.url,
                alerts_url=alerts_url,
                mulaw_bytes=mulaw_bytes,
                user_id=args.user,
                realtime=args.realtime,
            )
        )
    except Exception as err:
        print(f"Simulator error: {err}", file=sys.stderr)
        return 1

    print(f"\nFinal Alert Level: {final_level}")

    if args.expect and final_level != args.expect:
        print(
            f"ERROR: Expected level {args.expect}, but final level was {final_level}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
