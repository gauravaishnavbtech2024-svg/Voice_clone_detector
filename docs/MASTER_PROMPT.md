# MASTER PROMPT: Call Shield (real-time synthetic-voice alert prototype)

## 1. Product brief
Build a prototype that listens to the REMOTE party's audio during a phone call
(or a browser "call"), scores it in near real time for synthetic/cloned speech,
and shows an alert: OK, WARNING or ALERT. Audio arrives either as
(a) PCM16 16 kHz frames from a browser microphone, or
(b) Twilio Media Streams JSON events (8 kHz mu-law, 20 ms frames, base64).
A pretrained detector (PyTorch checkpoint) is supplied later by the user.
Until then everything must work with a deterministic FakeDetector test double.

Success criteria for the prototype:
1. Streaming a recording through the simulator produces correct level transitions.
2. The browser page shows the live score and level and reaches ALERT on a fake clip.
3. All unit and integration tests pass; coverage on `core` >= 85%.
4. No audio is ever persisted.

## 2. Working protocol (applies to every phase)
For each phase, in order:
1. Write an implementation plan artifact (files, tests, risks). Wait for approval.
2. RED: write the tests listed for the phase. Run them. Show the failing output.
3. GREEN: write the minimum code to pass. Run the tests. Show the passing output.
4. REFACTOR: clean up with tests green. Run ruff, mypy and the FULL suite.
5. Produce a phase report artifact: tests added, pass/fail counts, coverage, open risks.
6. Commit (test commits before feature commits).
7. STOP at the gate and wait for "APPROVED PHASE <n>".
Hackathon critical path: phases 0-6, 8 and 11. Phases 7, 9 and 10 are stretch.

## 3. Target repository layout
```
callshield/
├── .agent/rules/callshield.md
├── docs/MASTER_PROMPT.md  docs/SPEC.md  docs/MANUAL_TEST.md
├── src/callshield/
│   ├── core/    audio.py  buffer.py  decider.py  session.py  fusion.py
│   ├── detectors/ base.py  fake.py  torch_detector.py  factory.py
│   ├── server/  app.py  hub.py  config.py  protocol.py
│   └── tools/   simulate_call.py  calibrate.py
├── training/    (user-provided: model.py, augment.py, copied unchanged)
├── models/      (user-provided: best.pt, git-ignored)
├── client/      index.html  app.js  logic.js  worklet.js  tests/
├── tests/       unit/  integration/  e2e/  fixtures/
├── Makefile  pyproject.toml  README.md
```

## PHASE 0: Scaffold and tooling
Goal: empty but fully wired project where `make test` works.
Tests first:
- `tests/unit/test_smoke.py`: package imports and `callshield.__version__` exists.
- `tests/unit/test_markers.py`: markers unit/integration/e2e/slow/gpu are registered
  (run pytest with --strict-markers).
Implement: pyproject.toml (pytest, pytest-asyncio, pytest-cov, hypothesis, httpx,
ruff, mypy, fastapi, uvicorn, numpy, scipy, pydantic-settings, websockets),
Makefile targets: test, test-unit, test-int, cov, lint, e2e.
Write docs/SPEC.md from section 1, the conventions in the rules file and the phases below.
Done when: `make test` and `make lint` pass on a clean checkout.
HUMAN STEP 0: install Python 3.11, Node 20, git and ffmpeg; run `make test`
yourself; confirm the Antigravity workspace opens this folder with the rules file loaded.

## PHASE 1: Audio core (pure functions, no I/O)
Module `core/audio.py`. Tests first (write ALL before any code):
- `mulaw_decode`:
  * byte 0xFF -> 0.0 and 0x7F -> 0.0 (the two G.711 zeros);
  * 0x80 -> approx +0.980 and 0x00 -> approx -0.980 (G.711 maximum magnitude 32124/32768);
  * verify against the ITU-T G.711 reference table; show how you verified it;
  * output dtype float32, range within [-1, 1], length equals input length.
- `mulaw_encode` is used only by the simulator: property test (hypothesis) that
  decode(encode(x)) is within the G.711 quantisation error and encode(decode(b)) == b
  for all bytes except 0x7F (which maps to 0xFF).
- `resample_8k_to_16k`: output length is exactly 2 x input; a 440 Hz sine keeps its
  dominant frequency (FFT peak within 5 Hz); no NaN or inf; deterministic.
- `rms_normalize`: result RMS is -26 dBFS (+/-0.1 dB) for random noise at several
  input levels; an all-zero input returns zeros with no NaN (clamped);
  the output of normalising x * g equals the output for x (gain invariance) for g in [0.01, 10].
- `crop_or_pad(x, n)`: shorter inputs are REPEAT-padded (assert the tail equals a
  repeat of the start, and that the padded region is not all zeros); longer inputs
  are cropped to length n; exact-length inputs are returned unchanged.
- `is_silent(x, threshold_dbfs=-50)`: True for zeros and very quiet noise,
  False for a normal-level sine.
Implement with numpy and scipy only (NO torch import in `core`).
Gate: user reviews the report. Check that the G.711 verification is real.

## PHASE 2: Rolling buffer, AlertDecider and SessionEngine
`core/buffer.py` RollingBuffer(capacity_samples):
- append returns nothing; `latest(n)` returns the last n samples (fewer if not yet filled);
  never exceeds capacity; property test with random chunk sizes equals slicing a full array.
`core/decider.py` AlertDecider(thr=0.0, k=4, n=6):
- history = last n booleans, suspicious = score < thr; flagged = count.
- flagged >= k -> ALERT and LATCHES for the session; flagged >= k-2 -> WARNING; else OK.
- Table-driven tests: six genuine scores -> OK; one outlier among genuine -> OK;
  two suspicious in a row -> WARNING; four suspicious within the last six -> ALERT;
  genuine scores after ALERT still report ALERT; reset() clears the latch;
  n < k raises ValueError; NaN score is treated as "no information" (not appended).
`core/session.py` SessionEngine(detector, decider, sample_rate_in):
- feed(samples) accumulates audio; every 1 s of NEW audio, once >= 2 s are buffered,
  it scores the latest 4 s window (repeat-padded, RMS-normalised) and returns a State
  (t_seconds, score, level, flagged, reason).
- Silent windows are NOT scored: level OK, reason="silence".
- Tests with FakeDetector: 1 s of audio -> no state yet; 2 s -> first state;
  feeding in 20 ms chunks gives the same states as feeding all at once (chunking invariance);
  a fake sequence reaches ALERT at the expected second; two engines never share state.
Gate: review the decider test table against section 6.

## PHASE 3: Detector abstraction and test double
`detectors/base.py`: Protocol Detector with `score(x: np.ndarray) -> float`
(x is 4 s, 16 kHz, float32, normalised).
`detectors/fake.py`: FakeDetector, a deterministic TEST DOUBLE, NOT a real detector.
It returns a positive score for noise-like signals (high spectral flatness) and a negative
score for pure tones. It can also replay a scripted score list and can sleep N ms to
simulate slow inference.
`detectors/factory.py`: build_detector(config) -> Fake or Torch (Torch imported lazily).
Tests first: contract tests parametrised over detector implementations
(returns float, finite, deterministic for the same input, > 0 genuine-like and
< 0 fake-like for the fixtures, handles exactly-4s input and rejects wrong dtype or shape
with a clear error). The factory raises a clear error if torch or the weights are missing.
Add `tests/fixtures/generate_fixtures.py` (seeded) producing: genuine_like.wav (noise-ish speech
stand-in), fake_like.wav (tone), silence.wav, mixed.wav (genuine 6 s then fake 8 s).

## PHASE 4: Server and integration tests
`server/protocol.py`: typed message models. `server/hub.py`: AlertHub (per-user asyncio
queues, subscribe/unsubscribe/publish). `server/app.py` endpoints:
- GET /health -> {"status":"ok","detector":"<name>"}
- POST /api/score (multipart audio file) -> JSON timeline [{t, score, level}], rejects
  files over MAX_UPLOAD_MB and unsupported formats with 413/415.
- WS /ws/stream (browser): binary PCM16LE 16 kHz frames in; JSON state messages out.
- WS /twilio: Twilio events connected/start/media/stop; media.payload is base64 mu-law 8 kHz;
  reads start.customParameters.user_id; publishes states to the AlertHub.
- WS /ws/alerts?user_id=..&token=.. : alert subscriber (token checked, 4401 close if wrong).
Inference runs OFF the event loop (asyncio.to_thread) with a bounded concurrency semaphore;
if inference falls behind, drop the oldest pending window instead of queuing without limit.
Integration tests first (Starlette TestClient or httpx + websockets, FakeDetector):
- health returns ok; /api/score on mixed.wav returns a timeline whose level goes
  OK -> ... -> ALERT after the fake section starts, with the first ALERT within 6 s of the switch;
- /ws/stream with PCM16 frames of mixed.wav yields the same levels as /api/score;
- /twilio with real-format JSON events (generated from mixed.wav through mulaw_encode)
  publishes ALERT to a subscriber of /ws/alerts for that user_id;
- two simultaneous /twilio sessions do not leak state into each other;
- wrong token on /ws/alerts is rejected; malformed JSON closes the socket cleanly (no crash);
- while a slow FakeDetector (500 ms) is scoring, GET /health still responds in < 100 ms
  (proves inference is off the event loop).
Gate: user reviews the endpoint list and message schemas in docs/SPEC.md.

## PHASE 5: Call simulator (end-to-end without a phone)
`tools/simulate_call.py --wav file.wav --url ws://localhost:8000/twilio --user u1 [--realtime]`
streams a WAV as 20 ms mu-law Twilio-format events (real time or faster) and prints
(time, score, level) per second.
Tests first: unit tests for framing (160 samples per frame, base64 valid, correct event
order connected -> start -> media... -> stop); integration test running the simulator
against a live app instance in a thread; exit code non-zero if the final level is
not the expected one when `--expect` is given.
HUMAN STEP 5: run the simulator yourself on mixed.wav and confirm the console
output matches your expectation; paste the output into the chat.

## PHASE 6: Web client
Files: client/index.html, app.js, worklet.js, logic.js.
- worklet.js: AudioWorklet capturing mic audio, downsampling device rate -> 16 kHz,
  emitting PCM16 chunks of ~100 ms.
- logic.js (pure, ES module): `toViewModel(state)` (level -> color, text, vibrate flag),
  `downsample(float32, srcRate)` (pure function), `backoff(attempt)` for reconnects.
- app.js: connects to /ws/stream, renders a level chip (OK/WARNING/ALERT), a score
  sparkline, and for ALERT a banner: "This voice may be AI-generated. Hang up and call
  back on a number you know." and navigator.vibrate on ALERT. A mute/test-mode toggle.
Tests first:
- unit (node:test): toViewModel table; downsample(48 kHz->16 kHz) returns length/3,
  no NaN, preserves a 440 Hz tone's peak; backoff is monotonic and capped.
- e2e (pytest-playwright, marker e2e): start the server with FakeDetector, launch Chromium with
  `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream
  --use-file-for-fake-audio-capture=tests/fixtures/fake_like.wav`, open the page, click Start,
  assert the level chip reaches ALERT within 15 s; repeat with genuine_like.wav and
  assert it stays OK for 15 s. Save screenshots as artifacts.
HUMAN STEP 6: run `playwright install chromium`; open the page in your own browser with a real
microphone, speak for 10 s, confirm OK, then play a cloned clip through your speakers.

## PHASE 7 (stretch): Security, privacy and robustness
Tests first:
- no files are created or modified in the working directory or temp dir during a full
  /twilio session (snapshot before and after);
- logs captured with caplog never contain raw audio, base64 payloads or tokens;
- a frame larger than MAX_FRAME_BYTES or a session longer than MAX_SESSION_SECONDS is closed
  with a policy-violation code; more than MAX_SESSIONS rejects politely;
- config is read from environment variables, with safe defaults and no secrets in the repo.
Add graceful shutdown (open sockets closed, in-flight scoring cancelled).

## PHASE 8: Real model integration (needs the user)
HUMAN STEP 8 (STOP and wait): the user copies `model.py` and `augment.py` from the
hackathon training repo UNCHANGED into `training/`, and `best.pt` into `models/`.
The agent must not write or retrain a model.
Tests first:
- preprocessing parity: for a fixed seeded waveform, the inference preprocessing
  (rms_normalize + crop_or_pad) equals `training.augment` evaluation preprocessing within 1e-6
  (if they differ, report the difference to the user instead of "fixing" either side);
- TorchDetector contract tests (marker slow, gpu optional): finite float, deterministic
  (eval mode, no dropout), batch of one, CPU fallback works;
- real-fixture tests (skipped with a clear message when tests/fixtures/real/ is absent):
  every file in manifest.json is classified with the labelled sign;
  report the per-file scores in the artifact.
- a latency benchmark script: median and p95 inference time for one 4 s window, plus the
  max concurrent streams sustained before the lag exceeds 1 s.
HUMAN STEP 8b: record at least 3 genuine clips (including phone-quality ones) and generate
3 fakes with voices of PEOPLE WHO CONSENTED; put them in tests/fixtures/real/ with
manifest.json {"file": "...", "label": "genuine|synthetic", "channel": "mic|phone"}.

## PHASE 9 (stretch): Calibration and evaluation
`tools/calibrate.py --manifest ...` scores every file with sliding windows and reports: EER,
the ROC curve (saved as PNG), the threshold for a target false-alarm rate, false alerts
per hour of genuine audio (using the real AlertDecider), and time-to-detect on fakes.
Tests first (using FakeDetector and synthetic manifests): EER = 0 for perfectly separable
scores and ~0.5 for random scores; false-alerts-per-hour arithmetic; the threshold search
is monotonic.
HUMAN STEP 9: run it on a held-out set that was NOT used for training or tuning, choose the
threshold, set CALLSHIELD_THRESHOLD in your environment, and record the numbers in README.

## PHASE 10 (stretch): Scam-language fusion
`core/fusion.py`: risk = 0.5 * voice_suspicion + 0.4 * scam_language + 0.1 * unknown_caller,
levels WARNING > 0.4 and ALERT > 0.7 (weights configurable). Define an `Asr` and a
`ScamClassifier` interface; use FAKES in tests (a keyword classifier that flags "OTP",
"transfer now", "don't tell anyone", "account will be blocked").
Tests first: table-driven fusion cases (two weak signals together reach WARNING; one
weak signal alone does not), monotonicity (a higher input never lowers the risk),
and integration with SessionEngine. Real ASR integration is optional and behind a flag.

## PHASE 11: Packaging, docs, demo
- README (install, run, test, architecture diagram, results table, limitations, privacy).
- docs/MANUAL_TEST.md: checklist for browser microphone, simulator, and Twilio tests.
- Dockerfile and `make demo` that starts the server and opens the client.
- Demo script: genuine voice -> OK, cloned clip -> ALERT, phone-quality clip -> ALERT.
HUMAN STEP 11 (optional telephony): create a Twilio (or Indian provider) account and number,
run ngrok, set the voice webhook to return TwiML with <Start><Stream track="inbound_track"
url="wss://<host>/twilio"> then <Dial>. Verify attributes in the provider docs.
Then test with a real call and record results in docs/MANUAL_TEST.md.

## 6. Reference semantics (do not change without asking)
Decider table (thr=0, k=4, n=6): see Phase 2. Window 4 s, hop 1 s, first score after 2 s of audio.

## 7. Definition of done
All non-stretch phases approved; `make test` green; coverage >= 85% on core;
no skipped tests without a documented reason; README demo reproduced by a fresh clone.
