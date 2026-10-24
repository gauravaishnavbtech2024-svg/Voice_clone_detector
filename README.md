# Call Shield: Real-Time Synthetic-Voice Alert Prototype

Call Shield protects users from phone-based voice cloning scams by analyzing incoming speech in real time, detecting synthetic signatures, and raising immediate visual and haptic alerts.

---

## Features
- **Dual Stream Ingestion**:
  - Browser microphone stream (16 kHz PCM16).
  - Telephony stream via Twilio Media Streams (8 kHz G.711 µ-law).
- **Near Real-Time Evaluation**:
  - 4.0-second evaluation window, 1.0-second sliding hop.
  - Latched `ALERT` state on repeated suspicious windows (4 out of 6).
- **Zero Audio Persistence**:
  - Raw audio is processed purely in volatile memory. No audio is written to disk or logs.
- **TDD & High Test Coverage**:
  - Strict Test-Driven Development with >= 85% core coverage requirement.

---

## Target Repository Layout
```
callshield/
├── .agent/rules/callshield.md
├── docs/MASTER_PROMPT.md  docs/SPEC.md  docs/MANUAL_TEST.md
├── src/callshield/
│   ├── core/        # audio.py, buffer.py, decider.py, session.py, fusion.py
│   ├── detectors/   # base.py, fake.py, torch_detector.py, factory.py
│   ├── server/      # app.py, hub.py, config.py, protocol.py
│   └── tools/       # simulate_call.py, calibrate.py
├── training/        # User-provided model definitions
├── models/          # User-provided PyTorch checkpoints (git-ignored)
├── client/          # Web frontend: index.html, app.js, logic.js, worklet.js
├── tests/           # unit, integration, e2e, fixtures
├── Makefile         # Build, test, and lint targets
├── pyproject.toml   # Project configuration, dependencies, and tooling
└── README.md
```

---

## Prerequisites (HUMAN STEP 0)
Ensure the following tools are installed:
1. **Python 3.11**: [python.org](https://www.python.org/) or `winget install Python.Python.3.11`
2. **Node 20+**: [nodejs.org](https://nodejs.org/) (Installed)
3. **Git**: [git-scm.com](https://git-scm.com/) or `winget install Git.Git`
4. **FFmpeg**: [ffmpeg.org](https://ffmpeg.org/) or `winget install Gyan.FFmpeg`

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### 2. Running Tests
```bash
# Default (unit + integration):
make test
# Or with pytest directly:
pytest -m "unit or integration"

# Unit tests only:
make test-unit

# Linting & Type Checking:
make lint
```

---

## Phased Development Progress
- [x] **Phase 0: Scaffold and Tooling**
- [ ] **Phase 1: Audio Core**
- [ ] **Phase 2: Rolling Buffer, AlertDecider, and SessionEngine**
- [ ] **Phase 3: Detector Abstraction and Test Double**
- [ ] **Phase 4: Server and Integration Tests**
- [ ] **Phase 5: Call Simulator**
- [ ] **Phase 6: Web Client**
- [ ] **Phase 7: Security, Privacy, and Robustness (Stretch)**
- [ ] **Phase 8: Real Model Integration**
- [ ] **Phase 9: Calibration and Evaluation (Stretch)**
- [ ] **Phase 10: Scam-Language Fusion (Stretch)**
- [ ] **Phase 11: Packaging, Docs, Demo**
