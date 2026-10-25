/**
 * Call Shield Client Application Orchestrator
 *
 * ES Module orchestrating:
 * - WebSocket streaming to /ws/stream with exponential backoff reconnection
 * - Web Audio API & AudioWorklet capture of microphone PCM16 chunks
 * - Real-time synthetic test tone generation (440 Hz)
 * - 60 FPS Canvas sparkline telemetry visualizer
 * - Progressive alerts and accessible UI updates
 */

import { toViewModel, backoff, downsample, encodeWAV, summarizeTimeline } from './logic.js';

// Application State
let ws = null;
let reconnectAttempts = 0;
let reconnectTimeout = null;
let isRecording = false;
let dismissedAlert = false;

// Audio Context & Graph
let audioCtx = null;
let micStream = null;
let micSource = null;
let workletNode = null;
let silentGain = null;

// Telemetry History for Sparkline
const MAX_HISTORY = 40;
const scoreHistory = [];
for (let i = 0; i < MAX_HISTORY; i++) {
    scoreHistory.push({ score: 0.0, level: 'OK' });
}

// DOM Elements
const wsStatusDot = document.getElementById('ws-status-dot');
const wsStatusText = document.getElementById('ws-status-text');
const bannerAlert = document.getElementById('banner-alert');
const alertMessage = document.getElementById('alert-banner-message');
const btnDismissAlert = document.getElementById('btn-dismiss-alert');
const levelChip = document.getElementById('level-chip');
const chipLevel = document.getElementById('chip-level');
const telemetryScore = document.getElementById('telemetry-score');
const telemetryFlagged = document.getElementById('telemetry-flagged');
const telemetryStatus = document.getElementById('telemetry-status');
const monitorHeading = document.getElementById('monitor-heading');
const monitorSubtext = document.getElementById('monitor-subtext');
const btnMicToggle = document.getElementById('btn-mic-toggle');
const btnMicLabel = document.getElementById('btn-mic-label');
const sparklineCanvas = document.getElementById('sparkline-canvas');

// ============================================================================
// WebSocket Connection Management
// ============================================================================

function getWebSocketUrl() {
    if (window.location.protocol === 'file:' || !window.location.host) {
        return 'ws://127.0.0.1:8000/ws/stream';
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/stream`;
}

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }

    updateWsStatus('connecting', 'Connecting...');
    const url = getWebSocketUrl();

    try {
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            reconnectAttempts = 0;
            updateWsStatus('connected', 'Live Shield Active');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerState(data);
            } catch (err) {
                console.error('[CallShield] Failed to parse server message:', err);
            }
        };

        ws.onclose = () => {
            updateWsStatus('disconnected', 'Disconnected');
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.warn('[CallShield] WebSocket error:', err);
            ws.close();
        };
    } catch (err) {
        console.error('[CallShield] WebSocket connection failed:', err);
        scheduleReconnect();
    }
}

function scheduleReconnect() {
    if (reconnectTimeout) return;
    const delay = backoff(reconnectAttempts);
    reconnectAttempts++;
    updateWsStatus('disconnected', `Reconnecting in ${(delay / 1000).toFixed(1)}s...`);
    reconnectTimeout = setTimeout(() => {
        reconnectTimeout = null;
        connectWebSocket();
    }, delay);
}

function updateWsStatus(status, text) {
    if (!wsStatusDot || !wsStatusText) return;
    wsStatusDot.className = 'ws-status-dot';
    if (status === 'connected') {
        wsStatusDot.classList.add('connected');
    } else if (status === 'disconnected') {
        wsStatusDot.classList.add('disconnected');
    }
    wsStatusText.textContent = text;
}

// ============================================================================
// UI & Alert Updates
// ============================================================================

function handleServerState(state) {
    const vm = toViewModel(state);

    // Reset dismiss flag once state recovers to OK
    if (vm.level === 'OK') {
        dismissedAlert = false;
    }

    // Update level chip
    levelChip.className = `level-chip level-${vm.level}`;
    chipLevel.textContent = vm.level === 'OK' ? 'AUTHENTIC' : vm.level;

    // Update telemetry metrics
    const scoreVal = vm.score;
    const sign = scoreVal > 0 ? '+' : '';
    telemetryScore.textContent = `${sign}${scoreVal.toFixed(2)}`;
    telemetryScore.style.color = vm.color;

    telemetryFlagged.textContent = `${vm.flagged} / 6`;
    telemetryStatus.textContent = vm.reason.charAt(0).toUpperCase() + vm.reason.slice(1);

    // Update Card Titles
    monitorHeading.textContent = vm.title;
    monitorSubtext.textContent = vm.description;

    // Handle Alert Banner
    if (vm.showAlertBanner && !dismissedAlert) {
        bannerAlert.classList.add('active');
        alertMessage.textContent = vm.bannerText;
        if (vm.vibrate && 'vibrate' in navigator) {
            navigator.vibrate([250, 100, 250]);
        }
    } else if (!vm.showAlertBanner) {
        bannerAlert.classList.remove('active');
    }

    // Push into timeline history for sparkline
    scoreHistory.push({ score: vm.score, level: vm.level });
    if (scoreHistory.length > MAX_HISTORY) {
        scoreHistory.shift();
    }
}

// Dismiss alert banner handler
if (btnDismissAlert) {
    btnDismissAlert.addEventListener('click', () => {
        dismissedAlert = true;
        bannerAlert.classList.remove('active');
    });
}

// ============================================================================
// Web Audio & AudioWorklet Setup
// ============================================================================

async function ensureAudioContext() {
    if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        try {
            audioCtx = new AudioContextClass({ sampleRate: 16000 });
        } catch (_) {
            audioCtx = new AudioContextClass();
        }
    }
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    if (!workletNode) {
        // Load AudioWorklet processor
        await audioCtx.audioWorklet.addModule('worklet.js');
        workletNode = new AudioWorkletNode(audioCtx, 'callshield-worklet');

        // Stream PCM16 little-endian chunks to WebSocket
        workletNode.port.onmessage = (event) => {
            const pcmBuffer = event.data;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(pcmBuffer);
            }
        };

        // Create zero-gain silent route to keep worklet rendering graph active
        silentGain = audioCtx.createGain();
        silentGain.gain.value = 0.0;
        workletNode.connect(silentGain);
        silentGain.connect(audioCtx.destination);
    }
}

async function startMicrophone() {
    await ensureAudioContext();

    micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
        },
    });

    micSource = audioCtx.createMediaStreamSource(micStream);
    micSource.connect(workletNode);

    isRecording = true;
    btnMicToggle.classList.add('recording');
    btnMicLabel.textContent = 'Stop Voice Monitor';
    telemetryStatus.textContent = 'Monitoring Live';
}

function stopMicrophone() {
    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }
    if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
        micStream = null;
    }

    isRecording = false;
    btnMicToggle.classList.remove('recording');
    btnMicLabel.textContent = 'Start Voice Monitor';
    telemetryStatus.textContent = 'Idle';
}

btnMicToggle.addEventListener('click', async () => {
    try {
        if (!isRecording) {
            await startMicrophone();
        } else {
            stopMicrophone();
        }
    } catch (err) {
        console.error('[CallShield] Failed to access microphone:', err);
        alert('Microphone access denied or audio initialization failed.');
    }
});



// ============================================================================
// 60 FPS Canvas Sparkline Visualizer
// ============================================================================

function initCanvasDPI() {
    const dpr = window.devicePixelRatio || 1;
    const rect = sparklineCanvas.getBoundingClientRect();
    sparklineCanvas.width = rect.width * dpr;
    sparklineCanvas.height = rect.height * dpr;
    const ctx = sparklineCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
}

function drawSparkline() {
    requestAnimationFrame(drawSparkline);

    if (!sparklineCanvas) return;
    const ctx = sparklineCanvas.getContext('2d');
    const width = sparklineCanvas.clientWidth;
    const height = sparklineCanvas.clientHeight;

    ctx.clearRect(0, 0, width, height);

    // Draw baseline (score = 0.0) at vertical middle
    const midY = height / 2;
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(4, 106, 56, 0.25)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw grid labels
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#046a38';
    ctx.fillText('+1.0 [AUTHENTIC]', 8, 14);
    ctx.fillStyle = '#e34222';
    ctx.fillText('-1.0 [SYNTHETIC]', 8, height - 6);

    const len = scoreHistory.length;
    if (len < 2) return;

    // Calculate (x, y) coordinates for each historical score
    // Score range: [-1.0, 1.0] -> y range: [height - 12, 12]
    const points = [];
    const stepX = width / (MAX_HISTORY - 1);

    for (let i = 0; i < len; i++) {
        const item = scoreHistory[i];
        const x = i * stepX;
        // Clamp score [-1.0, 1.0]
        const clamped = Math.max(-1.0, Math.min(1.0, item.score));
        // Normalize: +1 -> 12px, 0 -> midY, -1 -> height - 12px
        const y = midY - (clamped * (midY - 14));
        points.push({ x, y, level: item.level });
    }

    const latest = points[points.length - 1];
    let strokeColor = '#046a38';
    let fillColorTop = 'rgba(4, 106, 56, 0.22)';
    let fillColorBottom = 'rgba(4, 106, 56, 0.01)';

    if (latest.level === 'ALERT') {
        strokeColor = '#e34222';
        fillColorTop = 'rgba(227, 66, 34, 0.25)';
        fillColorBottom = 'rgba(227, 66, 34, 0.01)';
    } else if (latest.level === 'WARNING') {
        strokeColor = '#c25e00';
        fillColorTop = 'rgba(194, 94, 0, 0.22)';
        fillColorBottom = 'rgba(194, 94, 0, 0.01)';
    }

    // Draw gradient area fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, fillColorTop);
    gradient.addColorStop(1, fillColorBottom);

    ctx.beginPath();
    ctx.moveTo(points[0].x, midY);
    for (let i = 0; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.lineTo(points[points.length - 1].x, midY);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line path with sharp miter join
    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'miter';
    ctx.lineCap = 'square';

    for (let i = 0; i < points.length; i++) {
        if (i === 0) {
            ctx.moveTo(points[i].x, points[i].y);
        } else {
            ctx.lineTo(points[i].x, points[i].y);
        }
    }
    ctx.stroke();

    // Draw square marker (no rounding) at the latest point
    ctx.fillStyle = strokeColor;
    ctx.fillRect(latest.x - 3.5, latest.y - 3.5, 7, 7);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(latest.x - 6.5, latest.y - 6.5, 13, 13);
}

window.addEventListener('resize', initCanvasDPI);

// ============================================================================
// Mode Tab Switching (Live vs Audio Upload Inspector)
// ============================================================================

const tabBtnLive = document.getElementById('tab-btn-live');
const tabBtnUpload = document.getElementById('tab-btn-upload');
const panelLive = document.getElementById('panel-live');
const panelUpload = document.getElementById('panel-upload');

function switchTab(mode) {
    if (mode === 'live') {
        tabBtnLive.classList.add('active');
        tabBtnLive.setAttribute('aria-selected', 'true');
        tabBtnUpload.classList.remove('active');
        tabBtnUpload.setAttribute('aria-selected', 'false');
        panelLive.style.display = 'block';
        panelUpload.style.display = 'none';
        initCanvasDPI();
    } else {
        tabBtnUpload.classList.add('active');
        tabBtnUpload.setAttribute('aria-selected', 'true');
        tabBtnLive.classList.remove('active');
        tabBtnLive.setAttribute('aria-selected', 'false');
        panelUpload.style.display = 'block';
        panelLive.style.display = 'none';
    }
}

if (tabBtnLive && tabBtnUpload) {
    tabBtnLive.addEventListener('click', () => switchTab('live'));
    tabBtnUpload.addEventListener('click', () => switchTab('upload'));
}

// ============================================================================
// Audio File Inspector & Upload Forensics
// ============================================================================

const dropzone = document.getElementById('dropzone');
const audioFileInput = document.getElementById('audio-file-input');
const uploadProcessing = document.getElementById('upload-processing');
const processingStep = document.getElementById('processing-step');
const analysisResults = document.getElementById('analysis-results');

const fileNameDisplay = document.getElementById('file-name-display');
const fileSpecsDisplay = document.getElementById('file-specs-display');
const audioPlayer = document.getElementById('audio-player');

const verdictBanner = document.getElementById('verdict-banner');
const verdictBadge = document.getElementById('verdict-badge');
const verdictTitle = document.getElementById('verdict-title');
const verdictDesc = document.getElementById('verdict-desc');

const uploadAvgScore = document.getElementById('upload-avg-score');
const uploadMinScore = document.getElementById('upload-min-score');
const uploadAlertCount = document.getElementById('upload-alert-count');
const uploadPeakTime = document.getElementById('upload-peak-time');
const timelineBarTrack = document.getElementById('timeline-bar-track');
const tableWindowCount = document.getElementById('table-window-count');
const diagnosticsTableBody = document.getElementById('diagnostics-table-body');

let activeAudioTimeline = [];

// Drag & drop handlers
if (dropzone && audioFileInput) {
    dropzone.addEventListener('click', () => {
        audioFileInput.click();
    });

    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            audioFileInput.click();
        }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            processSelectedFile(files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            processSelectedFile(e.target.files[0]);
        }
    });
}

/**
 * Handles audio file ingestion, format conversion (if not WAV), and API scoring.
 */
async function processSelectedFile(file) {
    if (!file) return;

    // Show processing indicator
    uploadProcessing.style.display = 'flex';
    analysisResults.style.display = 'none';
    processingStep.textContent = 'Preparing audio stream for analysis...';

    try {
        let uploadBlob = file;
        let finalDuration = 0;
        let finalSampleRate = 16000;

        const isWav = file.name.toLowerCase().endsWith('.wav') || file.type.includes('wav');

        if (!isWav) {
            processingStep.textContent = 'Transcoding audio to 16 kHz PCM stream in browser memory...';
            const arrayBuffer = await file.arrayBuffer();
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const tempCtx = new AudioContextClass();
            const decoded = await tempCtx.decodeAudioData(arrayBuffer);

            finalDuration = decoded.duration;
            finalSampleRate = 16000;

            const rawChannel = decoded.getChannelData(0);
            const downsampled = downsample(rawChannel, decoded.sampleRate, 16000);
            const wavBuffer = encodeWAV(downsampled, 16000);
            uploadBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            await tempCtx.close();
        }

        // Setup audio player preview
        const audioUrl = URL.createObjectURL(uploadBlob);
        audioPlayer.src = audioUrl;

        // Query server /api/score
        processingStep.textContent = 'Running rolling multi-window acoustic classifier on server...';
        const formData = new FormData();
        formData.append('file', uploadBlob, file.name.endsWith('.wav') ? file.name : `${file.name}.wav`);

        const response = await fetch('/api/score', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        const timeline = data.timeline || [];

        renderAnalysisResults(file, timeline, finalDuration, finalSampleRate);
    } catch (err) {
        console.error('[CallShield] Audio analysis failed:', err);
        alert(`Analysis failed: ${err.message}`);
    } finally {
        uploadProcessing.style.display = 'none';
    }
}

/**
 * Renders verdict, statistics, timeline bar chart, and window details.
 */
function renderAnalysisResults(file, timeline, estimatedDuration, sampleRate) {
    analysisResults.style.display = 'flex';
    activeAudioTimeline = timeline;

    const summary = summarizeTimeline(timeline);

    // Metadata
    fileNameDisplay.textContent = file.name;
    const durSec = estimatedDuration || (timeline.length > 0 ? timeline[timeline.length - 1].t : 0);
    const minStr = Math.floor(durSec / 60);
    const secStr = Math.floor(durSec % 60).toString().padStart(2, '0');
    fileSpecsDisplay.textContent = `${minStr}:${secStr} • ${(file.size / 1024).toFixed(1)} KB • ${sampleRate || 16000} Hz`;

    // Verdict
    verdictBanner.className = `verdict-banner ${summary.overallLevel}`;
    verdictBadge.textContent = summary.overallLevel === 'OK' ? 'AUTHENTIC' : summary.overallLevel;
    verdictTitle.textContent = summary.verdictTitle;
    verdictDesc.textContent = summary.verdictDescription;

    // Metrics
    const avgSign = summary.avgScore > 0 ? '+' : '';
    uploadAvgScore.textContent = `${avgSign}${summary.avgScore.toFixed(2)}`;
    uploadAvgScore.style.color = summary.avgScore > 0 ? 'var(--color-ok)' : 'var(--color-alert)';

    const minSign = summary.minScore > 0 ? '+' : '';
    uploadMinScore.textContent = `${minSign}${summary.minScore.toFixed(2)}`;
    uploadMinScore.style.color = summary.minScore > 0 ? 'var(--color-ok)' : 'var(--color-alert)';

    uploadAlertCount.textContent = `${summary.alertCount} Windows`;
    uploadPeakTime.textContent = summary.peakAlertTimestamp !== null
        ? `${summary.peakAlertTimestamp.toFixed(1)}s`
        : 'None';

    // Interactive Timeline Bars
    timelineBarTrack.innerHTML = '';
    timeline.forEach((pt, index) => {
        const col = document.createElement('div');
        col.className = 'timeline-bar-col';
        col.dataset.t = pt.t;
        col.title = `Window at ${pt.t.toFixed(1)}s: Score ${pt.score > 0 ? '+' : ''}${pt.score.toFixed(2)} (${pt.level})`;

        // Height maps score [-1, 1] to height [14px, 68px]
        const clampedScore = Math.max(-1.0, Math.min(1.0, pt.score));
        const normalizedHeight = Math.round(14 + ((clampedScore + 1.0) / 2.0) * 54);

        const pill = document.createElement('div');
        pill.className = `timeline-bar-pill ${pt.level}`;
        pill.style.height = `${normalizedHeight}px`;

        const timeLabel = document.createElement('span');
        timeLabel.className = 'timeline-bar-time';
        timeLabel.textContent = `${Math.round(pt.t)}s`;

        col.appendChild(pill);
        col.appendChild(timeLabel);

        // Click on bar seeks audio player directly to window start
        col.addEventListener('click', () => {
            if (audioPlayer) {
                audioPlayer.currentTime = Math.max(0, pt.t - 2.0);
                audioPlayer.play().catch(() => {});
            }
        });

        timelineBarTrack.appendChild(col);
    });

    // Window Diagnostics Table
    tableWindowCount.textContent = timeline.length;
    diagnosticsTableBody.innerHTML = '';
    timeline.forEach((pt) => {
        const row = document.createElement('tr');

        const cellTime = document.createElement('td');
        cellTime.textContent = `${pt.t.toFixed(1)}s`;

        const cellLevel = document.createElement('td');
        cellLevel.style.color = pt.level === 'ALERT'
            ? 'var(--color-alert)'
            : pt.level === 'WARNING'
            ? 'var(--color-warn)'
            : 'var(--color-ok)';
        cellLevel.style.fontWeight = '600';
        cellLevel.textContent = pt.level;

        const cellScore = document.createElement('td');
        const sSign = pt.score > 0 ? '+' : '';
        cellScore.textContent = `${sSign}${pt.score.toFixed(3)}`;

        const cellFlag = document.createElement('td');
        cellFlag.textContent = pt.level === 'ALERT'
            ? '🚨 AI Spoof'
            : pt.level === 'WARNING'
            ? '⚠️ Irregular'
            : '🟢 Natural';

        row.appendChild(cellTime);
        row.appendChild(cellLevel);
        row.appendChild(cellScore);
        row.appendChild(cellFlag);
        diagnosticsTableBody.appendChild(row);
    });
}

// Sync audio playback time to highlight the current timeline bar
if (audioPlayer) {
    audioPlayer.addEventListener('timeupdate', () => {
        const curr = audioPlayer.currentTime;
        const cols = timelineBarTrack.querySelectorAll('.timeline-bar-col');
        cols.forEach((col) => {
            const t = parseFloat(col.dataset.t);
            // Highlight window if current time is within [t-2, t]
            if (curr >= t - 2.0 && curr <= t) {
                col.classList.add('playing');
            } else {
                col.classList.remove('playing');
            }
        });
    });
}

// ============================================================================
// Initialization
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initCanvasDPI();
    drawSparkline();
    connectWebSocket();
});

