# Call Shield Specification (SPEC.md)

## 1. System Overview
Call Shield is a real-time synthetic-voice alert system designed to protect users during phone calls or browser-based voice communications against voice cloning scams.
The system analyzes incoming audio from the remote party, evaluates synthetic vs genuine likelihood in sliding windows, and raises progressive visual/haptic alerts: `OK`, `WARNING`, or `ALERT`.

## 2. Core Acoustic & Processing Conventions
- **Internal Model Rate**: 16,000 Hz (16 kHz), mono, 32-bit floating point (`float32`), bounded in `[-1.0, 1.0]`.
- **Telephony Ingestion**: 8,000 Hz (8 kHz) G.711 µ-law (`PCMU`), 20 ms frames (160 samples per frame).
- **Browser Ingestion**: 16,000 Hz PCM16 little-endian binary frames.
- **Windowing**:
  - Evaluation Window: 4.0 seconds (64,000 samples at 16 kHz).
  - Hop Size: 1.0 second (16,000 samples at 16 kHz).
  - Minimum buffered duration before first score: 2.0 seconds.
  - Short-window padding: Repeat-pad (never zero-pad).
  - Normalization: RMS-normalize to -26 dBFS (+/- 0.1 dB).
- **Silence Detection**: Windows below -50 dBFS are treated as silence and not scored (level remains `OK`, reason=`silence`).
- **Detector Score Polarity**:
  - Score > 0.0: `GENUINE` (Label 1)
  - Score < 0.0: `SYNTHETIC` (Label 0)

## 3. Decision Logic & Latching
- **AlertDecider Parameters**:
  - Threshold `thr = 0.0`
  - Window history length `n = 6`
  - Alert trigger threshold `k = 4`
- **Rules**:
  - A window is considered suspicious if `score < thr`.
  - `flagged` = number of suspicious windows in the last `n` windows.
  - If `flagged >= k`: state transitions to `ALERT` and **latches** for the entire session duration until explicitly reset.
  - If `flagged >= k - 2` (i.e. >= 2): state is `WARNING` (if not latched to `ALERT`).
  - Otherwise: state is `OK`.
  - `NaN` scores are treated as non-information and ignored in history.

## 4. Server API & Protocol

### HTTP Endpoints
- `GET /health`:
  - Returns `{"status": "ok", "detector": "<name>"}`.
  - Must respond in < 100 ms even when detector inference is running.
- `POST /api/score`:
  - Multipart audio upload (`.wav`, etc.).
  - Rejects files exceeding `MAX_UPLOAD_MB` (HTTP 413) or unsupported formats (HTTP 415).
  - Returns JSON timeline: `[{"t": float, "score": float, "level": "OK"|"WARNING"|"ALERT"}]`.

### WebSockets
- `WS /ws/stream`:
  - Direct microphone stream from browser.
  - Input: Binary PCM16LE 16 kHz frames.
  - Output: JSON state objects `{"t": float, "score": float, "level": "OK"|"WARNING"|"ALERT", "flagged": int, "reason": str}`.
- `WS /twilio`:
  - Ingestion for Twilio Media Streams.
  - Inbound JSON events: `connected`, `start`, `media`, `stop`.
  - Media payload: Base64-encoded 8 kHz µ-law audio.
  - Dispatches state updates to the `AlertHub` keyed by `user_id`.
- `WS /ws/alerts?user_id=<user_id>&token=<token>`:
  - Publishes alert events to the user's browser client.
  - Closes with code `4401` if authentication token is invalid.

## 5. Security & Privacy Guardrails
1. **Zero Audio Persistence**: Raw call audio, waveforms, and base64 audio frames are NEVER persisted to disk, database, or logging systems.
2. **Alert Phrasing**: Warnings and alerts must explicitly advise caution without asserting absolute certainty:
   - Alert Banner: *"This voice may be AI-generated. Hang up and call back on a number you know."*
3. **Configuration**: All secrets, tokens, and thresholds are passed via environment variables (e.g. `CALLSHIELD_TOKEN`, `CALLSHIELD_THRESHOLD`).
