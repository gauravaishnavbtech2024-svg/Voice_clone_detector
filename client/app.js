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

import { toViewModel, backoff } from './logic.js';

// Application State
let ws = null;
let reconnectAttempts = 0;
let reconnectTimeout = null;
let isRecording = false;
let isInjectingTone = false;
let dismissedAlert = false;

// Audio Context & Graph
let audioCtx = null;
let micStream = null;
let micSource = null;
let workletNode = null;
let silentGain = null;
let testOscillator = null;

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
const btnTestTone = document.getElementById('btn-test-tone');
const btnToneLabel = document.getElementById('btn-tone-label');
const sparklineCanvas = document.getElementById('sparkline-canvas');

// ============================================================================
// WebSocket Connection Management
// ============================================================================

function getWebSocketUrl() {
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
        audioCtx = new AudioContextClass();
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
    telemetryStatus.textContent = isInjectingTone ? 'Testing Tone' : 'Idle';
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
// Synthetic Test Tone Generator (440 Hz Sinusoid)
// ============================================================================

async function startTestTone() {
    await ensureAudioContext();

    testOscillator = audioCtx.createOscillator();
    testOscillator.type = 'sine';
    testOscillator.frequency.setValueAtTime(440, audioCtx.currentTime);

    // Connect oscillator directly into worklet input
    testOscillator.connect(workletNode);
    testOscillator.start();

    isInjectingTone = true;
    btnTestTone.classList.add('active');
    btnToneLabel.textContent = 'Stop Test Tone';
    telemetryStatus.textContent = 'Injecting 440Hz Tone';
}

function stopTestTone() {
    if (testOscillator) {
        try {
            testOscillator.stop();
            testOscillator.disconnect();
        } catch (_) {
            // Ignore already stopped oscillator
        }
        testOscillator = null;
    }

    isInjectingTone = false;
    btnTestTone.classList.remove('active');
    btnToneLabel.textContent = 'Inject Synthetic Tone (440Hz)';
    telemetryStatus.textContent = isRecording ? 'Monitoring Live' : 'Idle';
}

btnTestTone.addEventListener('click', async () => {
    try {
        if (!isInjectingTone) {
            await startTestTone();
        } else {
            stopTestTone();
        }
    } catch (err) {
        console.error('[CallShield] Failed to start test tone:', err);
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
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw grid labels
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(16, 185, 129, 0.5)';
    ctx.fillText('+1.0 (Authentic)', 8, 14);
    ctx.fillStyle = 'rgba(244, 63, 94, 0.5)';
    ctx.fillText('-1.0 (Synthetic)', 8, height - 6);

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
    let strokeColor = '#10b981';
    let fillColorTop = 'rgba(16, 185, 129, 0.25)';
    let fillColorBottom = 'rgba(16, 185, 129, 0.0)';

    if (latest.level === 'ALERT') {
        strokeColor = '#f43f5e';
        fillColorTop = 'rgba(244, 63, 94, 0.35)';
        fillColorBottom = 'rgba(244, 63, 94, 0.02)';
    } else if (latest.level === 'WARNING') {
        strokeColor = '#f59e0b';
        fillColorTop = 'rgba(245, 158, 11, 0.30)';
        fillColorBottom = 'rgba(245, 158, 11, 0.02)';
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

    // Draw line path
    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (let i = 0; i < points.length; i++) {
        if (i === 0) {
            ctx.moveTo(points[i].x, points[i].y);
        } else {
            ctx.lineTo(points[i].x, points[i].y);
        }
    }
    ctx.stroke();

    // Draw pulsing current dot at the latest point
    ctx.beginPath();
    ctx.arc(latest.x, latest.y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = strokeColor;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(latest.x, latest.y, 8, 0, Math.PI * 2);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

window.addEventListener('resize', initCanvasDPI);

// ============================================================================
// Initialization
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initCanvasDPI();
    drawSparkline();
    connectWebSocket();
});
