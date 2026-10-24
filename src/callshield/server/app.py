"""FastAPI application for Call Shield synthetic-voice alert server."""

import asyncio
import base64
import io
import json
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from scipy.io import wavfile

from callshield.core.audio import mulaw_decode, resample_8k_to_16k
from callshield.core.decider import AlertDecider
from callshield.core.session import SessionEngine
from callshield.detectors.factory import DetectorConfig, build_detector
from callshield.server.config import settings
from callshield.server.hub import AlertHub
from callshield.server.protocol import (
    HealthResponse,
    ScorePoint,
    ScoreResponse,
)

app = FastAPI(title="Call Shield", version="0.1.0")

# Global detector instance and alert hub
detector = build_detector(
    DetectorConfig(
        name=settings.CALLSHIELD_DETECTOR,
        weights_path=settings.MODEL_WEIGHTS_PATH,
        delay_ms=settings.DETECTOR_DELAY_MS,
    )
)
hub = AlertHub()
inference_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Returns liveness status and active detector name."""
    detector_name = getattr(detector, "name", settings.CALLSHIELD_DETECTOR)
    return HealthResponse(status="ok", detector=detector_name)


@app.post("/api/score", response_model=ScoreResponse)
async def score_audio_file(file: UploadFile) -> ScoreResponse:
    """Evaluates an uploaded audio file and returns the score/level timeline."""
    # Check media type / extension
    filename = file.filename or ""
    content_type = file.content_type or ""
    is_wav_name = filename.lower().endswith(".wav")
    is_wav_type = "wav" in content_type.lower() or "audio" in content_type.lower()
    if not (is_wav_name or is_wav_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio format. Only WAV format is currently supported.",
        )

    # Read bytes and check size limit
    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_MB} MB",
        )

    # Decode WAV in memory (never written to disk)
    try:
        sample_rate, raw_audio = wavfile.read(io.BytesIO(file_bytes))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Failed to decode WAV audio: {err}",
        ) from err

    # Normalize audio to float32 in [-1.0, 1.0]
    audio_arr = np.asarray(raw_audio)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr[:, 0]  # Take first channel if stereo

    if np.issubdtype(audio_arr.dtype, np.integer):
        max_int = np.iinfo(audio_arr.dtype).max
        audio_f32 = (audio_arr.astype(np.float32) / float(max_int)).astype(np.float32)
    else:
        audio_f32 = audio_arr.astype(np.float32)

    # Resample if needed
    samples_16k = resample_8k_to_16k(audio_f32) if sample_rate == 8000 else audio_f32

    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=16000)

    # Run scoring off event loop
    async with inference_semaphore:
        states = await asyncio.to_thread(engine.feed, samples_16k)

    timeline = [ScorePoint(t=s.t, score=s.score, level=s.level) for s in states]
    return ScoreResponse(timeline=timeline)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Ingests binary 16 kHz PCM16LE audio chunks from browser microphone."""
    await websocket.accept()
    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=16000)

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()
            if not raw_bytes:
                continue

            # Convert 16-bit PCM little-endian to float32 [-1.0, 1.0]
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Score off the event loop
            async with inference_semaphore:
                states = await asyncio.to_thread(engine.feed, samples)

            for state in states:
                await websocket.send_json(asdict(state))

    except WebSocketDisconnect:
        pass


@app.websocket("/twilio")
async def websocket_twilio(websocket: WebSocket) -> None:
    """Ingests Twilio Media Streams JSON protocol with 8 kHz base64 µ-law audio."""
    await websocket.accept()
    user_id: str = "default_user"
    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=8000)

    try:
        while True:
            message_text = await websocket.receive_text()
            try:
                event_data: dict[str, Any] = json.loads(message_text)
            except Exception:
                # Malformed JSON closes cleanly without crashing
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            event = event_data.get("event")

            if event == "start":
                start_data = event_data.get("start", {})
                custom_params = start_data.get("customParameters", {})
                if "user_id" in custom_params:
                    user_id = str(custom_params["user_id"])

            elif event == "media":
                media_data = event_data.get("media", {})
                payload_b64 = media_data.get("payload")
                if payload_b64:
                    raw_bytes = base64.b64decode(payload_b64)
                    samples_8k = mulaw_decode(raw_bytes)

                    # Score off event loop
                    async with inference_semaphore:
                        states = await asyncio.to_thread(engine.feed, samples_8k)

                    for state in states:
                        alert_payload = {
                            "t": state.t,
                            "score": state.score,
                            "level": state.level,
                            "flagged": state.flagged,
                            "reason": state.reason,
                            "alert": (state.level == "ALERT"),
                            "message": (
                                (
                                    "This voice may be AI-generated. "
                                    "Hang up and call back on a number you know."
                                )
                                if state.level == "ALERT"
                                else "Voice analysis in progress."
                            ),
                        }
                        await hub.publish(user_id, alert_payload)

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    user_id: str = Query(...),
    token: str = Query(...),
) -> None:
    """Authenticates and streams alerts for user_id."""
    if token != settings.CALLSHIELD_TOKEN:
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    hub.subscribe(user_id, queue, loop)

    async def sender_task() -> None:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)

    send_task = asyncio.create_task(sender_task())
    try:
        while True:
            # Keep socket alive listening for any client pings / disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        hub.unsubscribe(user_id, queue)
