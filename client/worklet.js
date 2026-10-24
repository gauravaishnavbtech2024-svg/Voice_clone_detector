/**
 * Call Shield AudioWorkletProcessor
 *
 * Runs off the browser main thread in Web Audio rendering thread.
 * Downsamples incoming microphone audio from hardware sample rate to 16 kHz,
 * accumulates ~100 ms frames (1600 samples), quantizes to 16-bit PCM little-endian,
 * and transfers to the main thread via MessagePort.
 */

class CallShieldAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer16k = [];
        this.frameSamples16k = 1600; // 100 ms at 16,000 Hz
        this.srcRate = currentFrame ? 48000 : 48000; // Will be set dynamically from parameters or context
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }

        const channelData = input[0];
        if (!channelData || channelData.length === 0) {
            return true;
        }

        // Downsample incoming block to 16 kHz using linear interpolation
        const srcRate = sampleRate; // Global AudioWorkletGlobalScope.sampleRate
        const targetRate = 16000;
        const ratio = srcRate / targetRate;
        const outLength = Math.floor(channelData.length / ratio);

        for (let i = 0; i < outLength; i++) {
            const srcPos = i * ratio;
            const idx = Math.floor(srcPos);
            const frac = srcPos - idx;
            const s1 = channelData[idx] || 0.0;
            const s2 = (idx + 1 < channelData.length) ? channelData[idx + 1] : s1;
            this.buffer16k.push(s1 + frac * (s2 - s1));
        }

        // Whenever we accumulate >= 100 ms of 16 kHz audio, emit a PCM16 packet
        while (this.buffer16k.length >= this.frameSamples16k) {
            const chunk = this.buffer16k.splice(0, this.frameSamples16k);
            const pcmBuffer = new ArrayBuffer(this.frameSamples16k * 2);
            const view = new DataView(pcmBuffer);

            for (let j = 0; j < this.frameSamples16k; j++) {
                const s = Math.max(-1.0, Math.min(1.0, chunk[j]));
                const intVal = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
                view.setInt16(j * 2, intVal, true);
            }

            // Zero-copy transfer to main thread
            this.port.postMessage(pcmBuffer, [pcmBuffer]);
        }

        return true;
    }
}

registerProcessor('callshield-worklet', CallShieldAudioProcessor);
