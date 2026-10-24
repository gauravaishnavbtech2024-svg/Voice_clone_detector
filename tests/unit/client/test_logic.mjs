import test from 'node:test';
import assert from 'node:assert/strict';
import { toViewModel, downsample, backoff, floatTo16BitPCM, encodeWAV, summarizeTimeline } from '../../../client/logic.js';

test('toViewModel maps OK level properly', () => {
    const vm = toViewModel({ level: 'OK', score: 1.5, flagged: 0, reason: 'scored' });
    assert.equal(vm.level, 'OK');
    assert.equal(vm.vibrate, false);
    assert.equal(vm.showAlertBanner, false);
    assert.ok(vm.color.includes('10b981') || vm.color.includes('green') || vm.color.includes('emerald'));
});

test('toViewModel maps WARNING level properly', () => {
    const vm = toViewModel({ level: 'WARNING', score: -0.2, flagged: 2, reason: 'scored' });
    assert.equal(vm.level, 'WARNING');
    assert.equal(vm.vibrate, false);
    assert.equal(vm.showAlertBanner, false);
    assert.ok(vm.color.includes('f59e0b') || vm.color.includes('amber') || vm.color.includes('yellow'));
});

test('toViewModel maps ALERT level properly with mandatory phrasing and vibration', () => {
    const vm = toViewModel({ level: 'ALERT', score: -0.2, flagged: 4, reason: 'scored' });
    assert.equal(vm.level, 'ALERT');
    assert.equal(vm.vibrate, true);
    assert.equal(vm.showAlertBanner, true);
    assert.ok(vm.color.includes('f43f5e') || vm.color.includes('ef4444') || vm.color.includes('rose') || vm.color.includes('red'));
    assert.ok(vm.bannerText.includes('This voice may be AI-generated'));
    assert.ok(vm.bannerText.includes('Hang up and call back on a number you know'));
});

test('downsample scales 48 kHz to 16 kHz accurately and produces no NaN', () => {
    const srcRate = 48000;
    const targetRate = 16000;
    const numSamples = 4800; // 0.1 s at 48 kHz
    const input = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
        input[i] = Math.sin(2 * Math.PI * 440 * (i / srcRate));
    }

    const output = downsample(input, srcRate, targetRate);
    assert.equal(output.length, 1600); // 4800 / 3

    for (let i = 0; i < output.length; i++) {
        assert.ok(!Number.isNaN(output[i]), `NaN found at index ${i}`);
        assert.ok(Number.isFinite(output[i]), `Non-finite value at index ${i}`);
    }

    // Verify 440 Hz tone peak preservation via discrete Fourier correlation
    const N = output.length;
    let cos440 = 0, sin440 = 0;
    let cos1000 = 0, sin1000 = 0;
    for (let n = 0; n < N; n++) {
        const angle440 = 2 * Math.PI * 440 * (n / targetRate);
        cos440 += output[n] * Math.cos(angle440);
        sin440 += output[n] * Math.sin(angle440);

        const angle1000 = 2 * Math.PI * 1000 * (n / targetRate);
        cos1000 += output[n] * Math.cos(angle1000);
        sin1000 += output[n] * Math.sin(angle1000);
    }
    const power440 = cos440 * cos440 + sin440 * sin440;
    const power1000 = cos1000 * cos1000 + sin1000 * sin1000;
    assert.ok(power440 > power1000 * 50, `440 Hz power (${power440}) should dominate 1000 Hz (${power1000})`);
});

test('backoff increases monotonically and is hard-capped at maxMs', () => {
    const baseMs = 500;
    const maxMs = 10000;
    let prev = 0;

    for (let attempt = 0; attempt < 10; attempt++) {
        const delay = backoff(attempt, baseMs, maxMs);
        assert.ok(delay >= prev, `Backoff not monotonic: attempt ${attempt} returned ${delay} < ${prev}`);
        assert.ok(delay <= maxMs, `Backoff exceeded maxMs: attempt ${attempt} returned ${delay} > ${maxMs}`);
        prev = delay;
    }

    assert.equal(backoff(20, baseMs, maxMs), maxMs);
});

test('floatTo16BitPCM quantizes and bounds samples correctly', () => {
    const input = new Float32Array([0.0, 1.0, -1.0, 1.5, -1.5]);
    const pcm = new Int16Array(floatTo16BitPCM(input));

    assert.equal(pcm.length, 5);
    assert.equal(pcm[0], 0);
    assert.equal(pcm[1], 32767);
    assert.equal(pcm[2], -32768);
    // Clipped values
    assert.equal(pcm[3], 32767);
    assert.equal(pcm[4], -32768);
});

test('encodeWAV creates valid RIFF/WAVE header and 16-bit PCM payload', () => {
    const samples = new Float32Array([0.0, 0.5, -0.5, 1.0]);
    const sampleRate = 16000;
    const wavBuffer = encodeWAV(samples, sampleRate);

    assert.equal(wavBuffer.byteLength, 44 + 4 * 2);
    const view = new DataView(wavBuffer);

    // RIFF magic
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    assert.equal(riff, 'RIFF');

    // WAVE magic
    const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
    assert.equal(wave, 'WAVE');

    // Channels = 1, SampleRate = 16000, BitsPerSample = 16
    assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getUint32(24, true), sampleRate);
    assert.equal(view.getUint16(34, true), 16);

    // Sample payload starts at byte 44
    assert.equal(view.getInt16(44, true), 0);
    assert.equal(view.getInt16(46, true), Math.round(0.5 * 32767));
    assert.equal(view.getInt16(48, true), Math.round(-0.5 * 32768));
    assert.equal(view.getInt16(50, true), 32767);
});

test('summarizeTimeline accurately computes stats, counts, and alert detection', () => {
    // 1. Empty timeline
    const emptySummary = summarizeTimeline([]);
    assert.equal(emptySummary.windowCount, 0);
    assert.equal(emptySummary.overallLevel, 'OK');

    // 2. All authentic timeline
    const okTimeline = [
        { t: 2.0, score: 0.8, level: 'OK' },
        { t: 3.0, score: 0.9, level: 'OK' },
        { t: 4.0, score: 0.7, level: 'OK' },
    ];
    const okSummary = summarizeTimeline(okTimeline);
    assert.equal(okSummary.windowCount, 3);
    assert.equal(okSummary.overallLevel, 'OK');
    assert.equal(okSummary.alertCount, 0);
    assert.ok(okSummary.avgScore > 0.7);

    // 3. Mixed timeline with warning and alert
    const mixedTimeline = [
        { t: 2.0, score: 0.8, level: 'OK' },
        { t: 3.0, score: -0.1, level: 'WARNING' },
        { t: 4.0, score: -0.9, level: 'ALERT' },
        { t: 5.0, score: -0.8, level: 'ALERT' },
    ];
    const mixedSummary = summarizeTimeline(mixedTimeline);
    assert.equal(mixedSummary.windowCount, 4);
    assert.equal(mixedSummary.overallLevel, 'ALERT');
    assert.equal(mixedSummary.alertCount, 2);
    assert.equal(mixedSummary.warningCount, 1);
    assert.equal(mixedSummary.peakAlertTimestamp, 4.0);
    assert.ok(mixedSummary.verdictTitle.includes('SYNTHETIC'));
});
