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
