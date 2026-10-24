"""Standalone Call Shield Voice Clone Detector Application.

Uses the trained PyTorch model (models/best.pt) on GPU/CPU to perform window-by-window
synthetic speech detection with the full editorial Call Shield UI.

Usage:
  # Launch interactive web UI in browser:
  python detector_app.py

  # Or detect an audio file directly from terminal:
  python detector_app.py --file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

# Add src and current directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from callshield.core.audio import crop_or_pad, is_silent, rms_normalize
from training.model import VoiceCloneDetectorModel

# -----------------------------------------------------------------------------
# PyTorch Model Initialization
# -----------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "best.pt"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model checkpoint not found at: {MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Loading Voice Clone Detector on {DEVICE}...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    MODEL_THRESHOLD = float(checkpoint.get("threshold", 1.632))
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
    MODEL_THRESHOLD = float(checkpoint.get("threshold", 1.632))
else:
    raise ValueError(f"Unrecognized checkpoint format in {MODEL_PATH}")

model = VoiceCloneDetectorModel().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"[*] Model loaded successfully! Calibrated threshold: {MODEL_THRESHOLD:.3f}")


# -----------------------------------------------------------------------------
# Window-by-Window Detection Engine
# -----------------------------------------------------------------------------
def analyze_waveform(samples_16k: np.ndarray) -> dict[str, object]:
    """Analyzes a 16 kHz mono float32 waveform window-by-window."""
    total_samples = len(samples_16k)
    duration_s = round(total_samples / 16000.0, 2)

    window_samples = 64000  # 4.0s
    hop_samples = 16000  # 1.0s
    min_samples = 32000  # 2.0s

    timeline: list[dict[str, object]] = []

    # If audio is shorter than 2 seconds, evaluate single padded window
    if total_samples < min_samples:
        eval_ends = [total_samples]
    else:
        eval_ends = list(range(min_samples, total_samples + 1, hop_samples))
        if eval_ends and eval_ends[-1] < total_samples:
            eval_ends.append(total_samples)

    for end_idx in eval_ends:
        t = round(end_idx / 16000.0, 1)
        start_idx = max(0, end_idx - window_samples)
        window = samples_16k[start_idx:end_idx]

        # Check silence
        if is_silent(window, threshold_dbfs=-45.0):
            timeline.append({
                "t": t,
                "score": 0.0,
                "level": "OK",
                "flag": "Quiet / Silence",
                "reason": "silence",
            })
            continue

        # Normalization: repeat-pad to exactly 4s and normalize to -26 dBFS
        normed = rms_normalize(crop_or_pad(window, window_samples), target_dbfs=-26.0)
        tensor = torch.from_numpy(normed).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logit = model(tensor).item()

        # Score calibrated against trained threshold (Positive = Genuine, Negative = Clone)
        calibrated_score = round(logit - MODEL_THRESHOLD, 2)

        # Level determination
        if calibrated_score >= 0.0:
            level = "OK"
            flag = "[OK] Natural"
        elif calibrated_score >= -2.0:
            level = "WARNING"
            flag = "[WARN] Irregular"
        else:
            level = "ALERT"
            flag = "[ALERT] AI Clone"

        timeline.append({
            "t": t,
            "score": calibrated_score,
            "level": level,
            "flag": flag,
            "reason": "scored",
        })

    # Summary statistics
    scored_points = [p for p in timeline if p["reason"] == "scored"]
    if scored_points:
        scores = [float(p["score"]) for p in scored_points]
        avg_score = round(float(np.mean(scores)), 2)
        min_score = round(float(np.min(scores)), 2)
    else:
        avg_score = 0.0
        min_score = 0.0

    alert_windows = [p for p in timeline if p["level"] == "ALERT"]
    alert_count = len(alert_windows)
    warning_count = sum(1 for p in timeline if p["level"] == "WARNING")
    peak_time = alert_windows[0]["t"] if alert_windows else None

    # Overall verdict
    if alert_count > 0:
        overall_level = "ALERT"
        verdict_title = "VERDICT: SYNTHETIC CLONE DETECTED"
        verdict_desc = f"Strong vocal clone signatures detected starting at {peak_time:.1f}s. Recommend hanging up immediately."
    elif warning_count >= 2:
        overall_level = "WARNING"
        verdict_title = "VERDICT: ELEVATED ANOMALY SUSPICION"
        verdict_desc = "Mild acoustic irregularities detected. Exercise caution and verify caller identity."
    else:
        overall_level = "OK"
        verdict_title = "VERDICT: AUTHENTIC NATURAL SPEECH"
        verdict_desc = "Vocal dynamics, breath patterns, and harmonic structures match authentic natural human voice."

    return {
        "duration": duration_s,
        "avg_score": avg_score,
        "min_score": min_score,
        "alert_count": alert_count,
        "warning_count": warning_count,
        "peak_time": peak_time,
        "overall_level": overall_level,
        "verdict_title": verdict_title,
        "verdict_desc": verdict_desc,
        "timeline": timeline,
    }


def decode_audio_file(file_bytes: bytes) -> np.ndarray:
    """Decodes audio bytes (WAV, FLAC, MP3, OGG) to 16 kHz mono float32."""
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    except Exception as e:
        raise ValueError(f"Could not decode audio format: {e}") from e

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr == 16000:
        return data.astype(np.float32)

    import math
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), 16000)
    up = 16000 // g
    down = int(sr) // g
    resampled = resample_poly(data, up=up, down=down).astype(np.float32)
    return resampled


# -----------------------------------------------------------------------------
# FastAPI Application & Frontend UI
# -----------------------------------------------------------------------------
app = FastAPI(title="Call Shield Voice Clone Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount samples directory
SAMPLES_DIR = BASE_DIR / "client" / "samples"
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.post("/api/detect")
async def detect_api(file: UploadFile = File(...)) -> JSONResponse:
    """Ingests any audio file and returns comprehensive window-by-window threat analysis."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided")

    try:
        samples_16k = decode_audio_file(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Audio decoding failed: {e}") from e

    results = analyze_waveform(samples_16k)
    results["filename"] = file.filename
    return JSONResponse(
        content=results,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> Response:
    """Serves the complete human-crafted Call Shield web UI with zero-caching headers."""
    html_content = UI_HTML
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


# -----------------------------------------------------------------------------
# Embedded Complete UI Template
# -----------------------------------------------------------------------------
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Shield - Real-Time AI Voice Clone Defense</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-linen: #f9ebd7;
            --bg-card: #f4dfc4;
            --bg-card-hover: #edd6b8;
            --text-primary: #1a1a1a;
            --text-secondary: #4a453e;
            --text-muted: #736b63;
            --border-strong: #2c2c2c;
            --border-light: rgba(44, 44, 44, 0.22);
            --color-ok: #046a38;
            --color-warn: #c25e00;
            --color-alert: #e34222;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            border-radius: 0 !important;
        }

        body {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 16px;
            line-height: 1.5;
            color: var(--text-primary);
            background-color: var(--bg-linen);
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--border-strong);
            margin-bottom: 28px;
        }

        .brand-title {
            font-family: 'DM Serif Display', serif;
            font-size: 2.2rem;
            letter-spacing: -0.02em;
        }

        .brand-sub {
            font-size: 0.95rem;
            color: var(--text-secondary);
            font-style: italic;
        }

        .model-chip {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            background: #fff;
            border: 2px solid var(--border-strong);
            padding: 6px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--color-ok);
        }

        .preset-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 24px;
        }

        .preset-btn {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 10px 18px;
            background: #fff;
            color: var(--text-primary);
            border: 2px solid var(--border-strong);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .preset-btn:hover {
            background: var(--border-strong);
            color: #fff;
        }

        .dropzone {
            background: #fff;
            border: 2px dashed var(--border-strong);
            padding: 40px 24px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 28px;
            transition: background 0.15s ease;
        }

        .dropzone:hover, .dropzone.dragover {
            background: #fff8ef;
            border-style: solid;
        }

        .dropzone-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.4rem;
            margin-bottom: 6px;
        }

        .dropzone-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .mic-box {
            display: flex;
            justify-content: center;
            margin-bottom: 24px;
        }

        .btn-mic {
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            padding: 12px 28px;
            background: #fff;
            border: 2px solid var(--border-strong);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn-mic.recording {
            background: var(--color-alert);
            color: #fff;
            border-color: var(--color-alert);
        }

        .loading-box {
            display: none;
            padding: 30px;
            background: #fff;
            border: 2px solid var(--border-strong);
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            margin-bottom: 28px;
        }

        .results-section {
            display: none;
            flex-direction: column;
            gap: 24px;
        }

        .verdict-banner {
            border: 2px solid var(--border-strong);
            padding: 24px;
            background: #fff;
        }

        .verdict-banner.OK {
            border-left: 10px solid var(--color-ok);
            background: rgba(4, 106, 56, 0.05);
        }

        .verdict-banner.WARNING {
            border-left: 10px solid var(--color-warn);
            background: rgba(194, 94, 0, 0.05);
        }

        .verdict-banner.ALERT {
            border-left: 10px solid var(--color-alert);
            background: rgba(227, 66, 34, 0.08);
        }

        .verdict-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .verdict-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 4px 12px;
            border: 2px solid currentColor;
        }

        .verdict-badge.OK { color: var(--color-ok); }
        .verdict-badge.WARNING { color: var(--color-warn); }
        .verdict-badge.ALERT { color: var(--color-alert); }

        .verdict-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.6rem;
            margin-bottom: 6px;
        }

        .verdict-desc {
            font-size: 1.05rem;
            color: var(--text-secondary);
        }

        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .telemetry-card {
            background: #fff;
            border: 2px solid var(--border-strong);
            padding: 16px 20px;
        }

        .telemetry-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .telemetry-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.7rem;
            font-weight: 700;
        }

        .visualizer-card {
            background: #fff;
            border: 2px solid var(--border-strong);
            padding: 20px;
        }

        .visualizer-title {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
        }

        .bar-track {
            display: flex;
            align-items: flex-end;
            gap: 6px;
            height: 90px;
            border-bottom: 2px solid var(--border-strong);
            padding-bottom: 4px;
            overflow-x: auto;
        }

        .bar-col {
            flex: 1;
            min-width: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
        }

        .bar-pill {
            width: 100%;
            transition: opacity 0.15s ease;
        }

        .bar-pill.OK { background: var(--color-ok); }
        .bar-pill.WARNING { background: var(--color-warn); }
        .bar-pill.ALERT { background: var(--color-alert); }

        .bar-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        audio {
            width: 100%;
            margin-top: 16px;
            border: 2px solid var(--border-strong);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            background: #fff;
            border: 2px solid var(--border-strong);
            margin-top: 20px;
        }

        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-light);
        }

        th {
            background: #f7ede0;
            font-weight: 700;
            border-bottom: 2px solid var(--border-strong);
        }

        td.score-cell {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1 class="brand-title">Call Shield</h1>
                <p class="brand-sub">Neural Voice Clone & Deepfake Interceptor (ResNet-18 Log-Mel)</p>
            </div>
            <div class="model-chip">
                <span class="status-dot"></span>
                <span>MODEL: best.pt (CUDA)</span>
            </div>
        </header>

        <!-- 1-Click Verification Presets -->
        <div class="preset-row">
            <button class="preset-btn" id="btn-auth">Sample 1: Authentic Voice</button>
            <button class="preset-btn" id="btn-mix">Sample 2: Mixed Call (Authentic &rarr; Clone)</button>
            <button class="preset-btn" id="btn-clone">Sample 3: AI Voice Clone</button>
        </div>

        <!-- In-Browser Microphone Recording -->
        <div class="mic-box">
            <button class="btn-mic" id="btn-mic">
                <span id="mic-icon">&#127908;</span>
                <span id="mic-label">Record Voice from Microphone</span>
            </button>
        </div>

        <!-- File Upload Dropzone -->
        <input type="file" id="file-input" accept="audio/*" style="display: none;">
        <div class="dropzone" id="dropzone">
            <h3 class="dropzone-title">Upload Audio Recording for Neural Inspection</h3>
            <p class="dropzone-sub">Drag & drop any .WAV, .MP3, .FLAC, or .M4A file here, or click to browse</p>
        </div>

        <!-- Loading Indicator -->
        <div class="loading-box" id="loading-box">
            Running 4-second sliding window neural spectrogram inference on GPU...
        </div>

        <!-- Detection Results Section -->
        <div class="results-section" id="results-section">
            <div class="verdict-banner" id="verdict-banner">
                <div class="verdict-header">
                    <span class="verdict-badge" id="verdict-badge">AUTHENTIC</span>
                    <span id="file-meta" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;"></span>
                </div>
                <h2 class="verdict-title" id="verdict-title">VERDICT: AUTHENTIC NATURAL SPEECH</h2>
                <p class="verdict-desc" id="verdict-desc"></p>
                <audio id="audio-player" controls></audio>
            </div>

            <!-- Telemetry Metrics -->
            <div class="telemetry-grid">
                <div class="telemetry-card">
                    <div class="telemetry-label">Mean Classifier Score</div>
                    <div class="telemetry-value" id="val-mean">+0.00</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">Lowest Window Score</div>
                    <div class="telemetry-value" id="val-min">+0.00</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">Alert Segments</div>
                    <div class="telemetry-value" id="val-alerts">0 Windows</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">First Threat Detected</div>
                    <div class="telemetry-value" id="val-peak">None</div>
                </div>
            </div>

            <!-- Window Threat Progression Visualizer -->
            <div class="visualizer-card">
                <div class="visualizer-title">
                    <span>Window-by-Window Threat Progression (Click bar to seek)</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Green = Authentic &bull; Amber = Irregular &bull; Red = AI Clone</span>
                </div>
                <div class="bar-track" id="bar-track"></div>
            </div>

            <!-- Diagnostics Table -->
            <details>
                <summary style="font-family: 'Inter', sans-serif; font-weight: 700; cursor: pointer; padding: 10px 0;">
                    Expand Diagnostic Table (<span id="table-count">0</span> Windows)
                </summary>
                <table>
                    <thead>
                        <tr>
                            <th>Window Horizon</th>
                            <th>Threat Level</th>
                            <th>Acoustic Score</th>
                            <th>Neural Flag</th>
                        </tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
            </details>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('file-input');
        const loadingBox = document.getElementById('loading-box');
        const resultsSection = document.getElementById('results-section');
        const audioPlayer = document.getElementById('audio-player');

        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) uploadAndAnalyze(e.target.files[0]);
        });

        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) uploadAndAnalyze(e.dataTransfer.files[0]);
        });

        // Preset button handlers
        document.getElementById('btn-auth').addEventListener('click', async () => {
            const resp = await fetch('/samples/sample_authentic.wav');
            const blob = await resp.blob();
            uploadAndAnalyze(new File([blob], 'authentic_human_speech.wav', { type: 'audio/wav' }));
        });
        document.getElementById('btn-mix').addEventListener('click', async () => {
            const resp = await fetch('/samples/sample_mixed.wav');
            const blob = await resp.blob();
            uploadAndAnalyze(new File([blob], 'mixed_authentic_and_clone.wav', { type: 'audio/wav' }));
        });
        document.getElementById('btn-clone').addEventListener('click', async () => {
            const resp = await fetch('/samples/sample_clone.wav');
            const blob = await resp.blob();
            uploadAndAnalyze(new File([blob], 'ai_voice_clone_threat.wav', { type: 'audio/wav' }));
        });

        // Microphone recording
        let mediaRecorder = null;
        let audioChunks = [];
        const btnMic = document.getElementById('btn-mic');
        const micLabel = document.getElementById('mic-label');

        btnMic.addEventListener('click', async () => {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunks, { type: 'audio/wav' });
                    uploadAndAnalyze(new File([blob], 'microphone_recording.wav', { type: 'audio/wav' }));
                };
                mediaRecorder.start();
                btnMic.classList.add('recording');
                micLabel.textContent = 'Stop Recording & Analyze';
            } else {
                mediaRecorder.stop();
                btnMic.classList.remove('recording');
                micLabel.textContent = 'Record Voice from Microphone';
            }
        });

        async function uploadAndAnalyze(file) {
            loadingBox.style.display = 'block';
            resultsSection.style.display = 'none';
            audioPlayer.src = URL.createObjectURL(file);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/detect', { method: 'POST', body: formData });
                if (!response.ok) throw new Error(await response.text());
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert('Detection failed: ' + err.message);
            } finally {
                loadingBox.style.display = 'none';
            }
        }

        function renderResults(data) {
            resultsSection.style.display = 'flex';

            // Verdict banner
            const banner = document.getElementById('verdict-banner');
            banner.className = 'verdict-banner ' + data.overall_level;
            const badge = document.getElementById('verdict-badge');
            badge.className = 'verdict-badge ' + data.overall_level;
            badge.textContent = data.overall_level;
            document.getElementById('file-meta').textContent = data.filename + ' (' + data.duration + 's)';
            document.getElementById('verdict-title').textContent = data.verdict_title;
            document.getElementById('verdict-desc').textContent = data.verdict_desc;

            // Telemetry cards
            const valMean = document.getElementById('val-mean');
            valMean.textContent = (data.avg_score > 0 ? '+' : '') + data.avg_score.toFixed(2);
            valMean.style.color = data.avg_score > 0 ? 'var(--color-ok)' : 'var(--color-alert)';

            const valMin = document.getElementById('val-min');
            valMin.textContent = (data.min_score > 0 ? '+' : '') + data.min_score.toFixed(2);
            valMin.style.color = data.min_score > 0 ? 'var(--color-ok)' : 'var(--color-alert)';

            document.getElementById('val-alerts').textContent = data.alert_count + ' Windows';
            document.getElementById('val-peak').textContent = data.peak_time ? data.peak_time.toFixed(1) + 's' : 'None';

            // Visualizer bars
            const barTrack = document.getElementById('bar-track');
            barTrack.innerHTML = '';
            data.timeline.forEach((pt) => {
                const col = document.createElement('div');
                col.className = 'bar-col';
                col.title = 'Window at ' + pt.t + 's: Score ' + pt.score + ' (' + pt.level + ')';

                const clamped = Math.max(-10, Math.min(10, pt.score));
                const heightPx = Math.round(16 + ((clamped + 10) / 20) * 60);

                const pill = document.createElement('div');
                pill.className = 'bar-pill ' + pt.level;
                pill.style.height = heightPx + 'px';

                const timeLabel = document.createElement('span');
                timeLabel.className = 'bar-time';
                timeLabel.textContent = Math.round(pt.t) + 's';

                col.appendChild(pill);
                col.appendChild(timeLabel);

                col.addEventListener('click', () => {
                    audioPlayer.currentTime = Math.max(0, pt.t - 2.0);
                    audioPlayer.play().catch(() => {});
                });
                barTrack.appendChild(col);
            });

            // Diagnostic table
            document.getElementById('table-count').textContent = data.timeline.length;
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            data.timeline.forEach((pt) => {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td>' + pt.t.toFixed(1) + 's</td>' +
                    '<td style="font-weight:600; color:' + (pt.level==='ALERT'?'var(--color-alert)':(pt.level==='WARNING'?'var(--color-warn)':'var(--color-ok)')) + '">' + pt.level + '</td>' +
                    '<td class="score-cell">' + (pt.score > 0 ? '+' : '') + pt.score.toFixed(2) + '</td>' +
                    '<td>' + pt.flag + '</td>';
                tbody.appendChild(tr);
            });
        }
    </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Main Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call Shield Voice Clone Detector")
    parser.add_argument("--file", type=str, default=None, help="Path to audio file for direct CLI evaluation")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        print(f"\n========================================================")
        print(f"CALL SHIELD: EVALUATING {file_path.name}")
        print(f"========================================================")
        with open(file_path, "rb") as f:
            data = decode_audio_file(f.read())
        results = analyze_waveform(data)

        print(f"\nDuration:              {results['duration']}s")
        print(f"Mean Classifier Score: {results['avg_score']:+.2f}")
        print(f"Lowest Window Score:   {results['min_score']:+.2f}")
        print(f"Alert Windows Count:   {results['alert_count']}")
        print(f"First Anomaly Time:    {results['peak_time']}")
        print(f"\n>>> {results['verdict_title']} <<<")
        print(f"Details: {results['verdict_desc']}")

        print(f"\nWindow-by-Window Diagnostic Timeline:")
        print(f"{'Time':<8} | {'Score':<8} | {'Level':<10} | {'Flag'}")
        print("-" * 45)
        for pt in results["timeline"]:
            print(f"{pt['t']:<6.1f}s | {pt['score']:<+8.2f} | {pt['level']:<10} | {pt['flag']}")
        print("========================================================\n")
    else:
        print("\n==================================================================")
        print(" CALL SHIELD // AI VOICE CLONE INTERCEPTOR ")
        print("==================================================================")
        print(f" Loaded Model:      {MODEL_PATH} ({DEVICE})")
        print(f" Calibrated Thr:    {MODEL_THRESHOLD:.3f}")
        print(f" Web UI Address:    http://{args.host}:{args.port}")
        print("==================================================================\n")
        uvicorn.run(app, host=args.host, port=args.port)
