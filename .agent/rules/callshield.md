# Call Shield: standing rules

## Process
- Always use TDD: write a failing test first (RED), run it and show the failure,
  write the minimum code to pass (GREEN), then refactor with tests still green.
- Work on ONE phase at a time from docs/MASTER_PROMPT.md. Never start the next
  phase until the user replies exactly "APPROVED PHASE <n>".
- Before coding a phase, produce an implementation plan artifact and wait for approval.
- When a step is marked HUMAN STEP, stop, tell the user exactly what to do,
  and wait. Never invent or fake the thing the human must provide.
- Commit after each phase: first "test:" commits, then "feat:" / "refactor:" commits.

## Testing
- Never weaken, delete or skip a test to make it pass. If a test is wrong,
  explain why and ask the user.
- Unit tests: no network, no GPU, no disk beyond tmp_path, no sleeps, deterministic seeds.
- Integration tests: real FastAPI app and real WebSocket, test doubles only for the detector.
- Markers: unit, integration, e2e, slow, gpu. Default `make test` runs unit + integration.
- Coverage gate: >= 85% on src/callshield/core.

## Conventions (do not change without asking)
- Waveforms: mono float32 in [-1, 1]. Model rate 16000 Hz. Telephony input 8000 Hz µ-law.
- Detector score: > 0 means GENUINE, < 0 means SYNTHETIC. Label 1 = genuine, 0 = spoof.
- Windows: 4 s, hop 1 s, repeat-pad (never zero-pad) short clips, RMS-normalise to -26 dBFS.
- Python 3.11, type hints everywhere, ruff and mypy clean.

## Privacy and safety
- Never write call audio to disk or logs. Log only scores, levels and timestamps.
- Alerts say "may be synthetic, verify", never "this is fake".
- No secrets in code. Configuration comes from environment variables.
