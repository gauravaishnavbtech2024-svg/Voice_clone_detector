/**
 * Call Shield Client Pure Logic Module
 *
 * ES Module containing pure functions with zero DOM dependencies:
 * - toViewModel: transforms server state into display attributes & alert banners.
 * - downsample: audio rate converter (e.g. 48 kHz -> 16 kHz).
 * - backoff: exponential reconnection backoff calculator.
 * - floatTo16BitPCM: quantizes float32 in [-1, 1] to signed 16-bit PCM buffer.
 */

export function toViewModel(state) {
    const level = state && state.level ? state.level : 'OK';
    const score = state && typeof state.score === 'number' ? state.score : 0.0;
    const flagged = state && typeof state.flagged === 'number' ? state.flagged : 0;
    const reason = state && state.reason ? state.reason : 'initializing';

    if (level === 'ALERT') {
        return {
            level: 'ALERT',
            score,
            flagged,
            reason,
            color: '#f43f5e',
            bgGradient: 'linear-gradient(135deg, rgba(244, 63, 94, 0.25) 0%, rgba(225, 29, 72, 0.08) 100%)',
            title: 'CRITICAL ALERT: Synthetic Voice',
            description: 'Strong acoustic anomalies detected matching known AI voice clone patterns.',
            vibrate: true,
            showAlertBanner: true,
            bannerText: 'This voice may be AI-generated. Hang up and call back on a number you know.',
        };
    }

    if (level === 'WARNING') {
        return {
            level: 'WARNING',
            score,
            flagged,
            reason,
            color: '#f59e0b',
            bgGradient: 'linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.08) 100%)',
            title: 'WARNING: Elevated Suspicion',
            description: 'Unusual vocal periodicity detected. Continue monitoring caller closely.',
            vibrate: false,
            showAlertBanner: false,
            bannerText: '',
        };
    }

    // Default 'OK' state
    return {
        level: 'OK',
        score,
        flagged,
        reason,
        color: '#10b981',
        bgGradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.20) 0%, rgba(5, 150, 105, 0.06) 100%)',
        title: 'AUTHENTIC: Natural Voice',
        description: 'Vocal acoustic dynamics match authentic human speech characteristics.',
        vibrate: false,
        showAlertBanner: false,
        bannerText: '',
    };
}

export function downsample(samples, srcRate, targetRate = 16000) {
    if (!samples || samples.length === 0) {
        return new Float32Array(0);
    }
    if (srcRate === targetRate) {
        return new Float32Array(samples);
    }

    const ratio = srcRate / targetRate;
    const outputLength = Math.floor(samples.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
        const srcPos = i * ratio;
        const idx = Math.floor(srcPos);
        const frac = srcPos - idx;
        const s1 = samples[idx] || 0.0;
        const s2 = (idx + 1 < samples.length) ? samples[idx + 1] : s1;
        output[i] = s1 + frac * (s2 - s1);
    }

    return output;
}

export function backoff(attempt, baseMs = 500, maxMs = 10000) {
    const safeAttempt = Math.max(0, Number.isFinite(attempt) ? attempt : 0);
    const exponential = baseMs * Math.pow(2, safeAttempt);
    return Math.min(maxMs, exponential);
}

export function floatTo16BitPCM(samples) {
    const buffer = new ArrayBuffer(samples.length * 2);
    const view = new DataView(buffer);

    for (let i = 0; i < samples.length; i++) {
        // Hard clip to [-1.0, 1.0]
        const s = Math.max(-1.0, Math.min(1.0, samples[i]));
        // Scale to 16-bit signed integer
        const intSample = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
        // Little-endian
        view.setInt16(i * 2, intSample, true);
    }

    return buffer;
}

export function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF chunk descriptor
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');

    // fmt sub-chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true); // NumChannels (1 mono)
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate
    view.setUint16(32, 2, true); // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample

    // data sub-chunk
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM 16-bit samples
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1.0, Math.min(1.0, samples[i]));
        const intSample = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
        view.setInt16(44 + i * 2, intSample, true);
    }

    return buffer;
}

export function summarizeTimeline(timeline) {
    if (!timeline || !Array.isArray(timeline) || timeline.length === 0) {
        return {
            windowCount: 0,
            avgScore: 0.0,
            minScore: 0.0,
            overallLevel: 'OK',
            peakAlertTimestamp: null,
            alertCount: 0,
            warningCount: 0,
            okCount: 0,
            verdictTitle: 'NO DATA',
            verdictDescription: 'No timeline windows were processed.',
        };
    }

    let sum = 0;
    let minScore = Infinity;
    let alertCount = 0;
    let warningCount = 0;
    let okCount = 0;
    let peakAlertTimestamp = null;

    for (const pt of timeline) {
        const score = typeof pt.score === 'number' ? pt.score : 0.0;
        sum += score;
        if (score < minScore) {
            minScore = score;
        }

        if (pt.level === 'ALERT') {
            alertCount++;
            if (peakAlertTimestamp === null) {
                peakAlertTimestamp = pt.t;
            }
        } else if (pt.level === 'WARNING') {
            warningCount++;
        } else {
            okCount++;
        }
    }

    const avgScore = sum / timeline.length;
    let overallLevel = 'OK';
    let verdictTitle = 'VERDICT: AUTHENTIC NATURAL SPEECH';
    let verdictDescription = 'Acoustic dynamics and harmonic structures match authentic natural human voice.';

    if (alertCount > 0) {
        overallLevel = 'ALERT';
        verdictTitle = 'VERDICT: SYNTHETIC CLONE DETECTED';
        const tStr = peakAlertTimestamp !== null ? peakAlertTimestamp.toFixed(1) : '0';
        verdictDescription = `Strong synthetic voice characteristics identified (first detected at ${tStr}s).`;
    } else if (warningCount > 0) {
        overallLevel = 'WARNING';
        verdictTitle = 'VERDICT: SUSPICIOUS ANOMALIES';
        verdictDescription = 'Elevated periodic or flat acoustic patterns detected. Potential voice modification.';
    }

    return {
        windowCount: timeline.length,
        avgScore,
        minScore: minScore === Infinity ? 0.0 : minScore,
        overallLevel,
        peakAlertTimestamp,
        alertCount,
        warningCount,
        okCount,
        verdictTitle,
        verdictDescription,
    };
}
