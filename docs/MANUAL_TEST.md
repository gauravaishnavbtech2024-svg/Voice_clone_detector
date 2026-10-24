# Call Shield: Manual Testing Checklist (MANUAL_TEST.md)

This checklist provides step-by-step procedures to verify Call Shield across different operating modes.

## Checklist Overview
- [ ] 1. Local Environment & Dependencies
- [ ] 2. Audio Processing Core Verification
- [ ] 3. Call Simulator Streaming
- [ ] 4. Browser UI & Live Microphone Capture
- [ ] 5. Twilio Media Stream Ingestion (Optional Telephony)

---

## 1. Local Environment & Dependencies
1. Confirm Python 3.11: `python --version` -> `Python 3.11.x`
2. Confirm Node 20: `node --version` -> `v20.x` or `v24.x`
3. Confirm Git: `git --version`
4. Confirm FFmpeg: `ffmpeg -version`
5. Run automated test suite: `pytest -m "unit or integration"`
6. Verify code style & typing: `ruff check .` and `mypy src tests`

---

## 2. Audio Processing Core Verification
1. Run G.711 µ-law verification tests: `pytest tests/unit/test_audio.py`
2. Verify RMS normalization (-26 dBFS target) and repeat-padding.
3. Verify `AlertDecider` table tests latch on synthetic detection: `pytest tests/unit/test_decider.py`

---

## 3. Call Simulator Streaming (Phase 5)
1. Start Call Shield server in one terminal:
   ```bash
   uvicorn callshield.server.app:app --port 8000
   ```
2. In a second terminal, stream `tests/fixtures/mixed.wav`:
   ```bash
   python -m callshield.tools.simulate_call --wav tests/fixtures/mixed.wav --url ws://localhost:8000/twilio --user user_test --realtime
   ```
3. Observe terminal output:
   - Initial 6 seconds (genuine): Level `OK`
   - Following seconds (synthetic tone/clip): Level transitions `OK` -> `WARNING` -> `ALERT`
   - Confirm first `ALERT` occurs within 6 seconds of transition.

---

## 4. Browser UI & Microphone Capture (Phase 6)
1. Open `http://localhost:8000` (or client server).
2. Click **Start Monitoring**. Grant microphone permissions.
3. Speak normally for 10 seconds.
   - Status chip should show `OK` (green).
   - Score sparkline should remain in positive territory (> 0.0).
4. Play synthetic/cloned audio sample through speakers.
   - Status chip transitions to `WARNING` (amber) then `ALERT` (red).
   - Alert banner displays: *"This voice may be AI-generated. Hang up and call back on a number you know."*
   - Supported devices trigger haptic vibration.

---

## 5. Telephony Ingestion via Twilio (Phase 11 Optional)
1. Launch ngrok: `ngrok http 8000`
2. Configure Twilio voice webhook:
   ```xml
   <Response>
     <Start>
       <Stream track="inbound_track" url="wss://<your-ngrok-domain>/twilio" />
     </Start>
     <Dial>+1234567890</Dial>
   </Response>
   ```
3. Place a test call to the Twilio number.
4. Verify inbound base64 µ-law audio is scored without audio persistence.
