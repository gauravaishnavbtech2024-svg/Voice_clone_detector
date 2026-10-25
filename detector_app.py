"""Native Desktop Voice Clone Detector.

Uses models/best.pt directly with PyTorch to detect AI voice clones.
No server, no browser, completely local desktop app.

Usage:
  python detector_app.py
"""

import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch

from training.model import VoiceCloneDetectorModel

# Load PyTorch Model
MODEL_PATH = BASE_DIR / "models" / "best.pt"
if not MODEL_PATH.is_file():
    messagebox.showerror("Error", f"Model not found at: {MODEL_PATH}")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Loading model on {DEVICE}...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
THRESHOLD = float(checkpoint.get("threshold", 1.632)) if isinstance(checkpoint, dict) else 1.632

model = VoiceCloneDetectorModel().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"[*] Ready! Using threshold: {THRESHOLD:.3f}")


import os
import ctypes
from ctypes import wintypes, c_void_p, c_uint32, POINTER, byref, Structure, c_ulong, c_int, c_longlong, cast


def _read_audio_windows_mf(file_path: str) -> np.ndarray:
    """Decodes audio using native Windows Media Foundation (supports .m4a, .aac, .mp4, .wma)."""
    mfplat = ctypes.windll.mfplat
    mfrw = ctypes.windll.mfreadwrite
    ole32 = ctypes.windll.ole32

    ole32.CoInitialize(None)
    hr = mfplat.MFStartup((2 << 16) | 138, 0)
    if hr != 0:
        ole32.CoUninitialize()
        raise RuntimeError(f"MFStartup failed: 0x{hr & 0xFFFFFFFF:08X}")

    abs_path = os.path.abspath(file_path)
    pReader = c_void_p()
    hr = mfrw.MFCreateSourceReaderFromURL(wintypes.LPCWSTR(abs_path), None, byref(pReader))
    if hr != 0 or not pReader.value:
        mfplat.MFShutdown()
        ole32.CoUninitialize()
        raise RuntimeError(f"MFCreateSourceReaderFromURL failed: 0x{hr & 0xFFFFFFFF:08X}")

    def call_v(obj, idx, restype, argtypes, *args):
        vtbl = cast(obj, POINTER(POINTER(c_void_p))).contents[idx]
        proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
        return proto(vtbl)(obj, *args)

    try:
        # Deselect all streams, select first audio stream
        call_v(pReader, 4, c_int, [c_uint32, wintypes.BOOL], c_uint32(0xFFFFFFFE), wintypes.BOOL(False))
        call_v(pReader, 4, c_int, [c_uint32, wintypes.BOOL], c_uint32(0xFFFFFFFD), wintypes.BOOL(True))

        pType = c_void_p()
        mfplat.MFCreateMediaType(byref(pType))

        class WAVEFORMATEX(Structure):
            _fields_ = [
                ("wFormatTag", wintypes.WORD),
                ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD),
            ]

        # Configure output format: 16kHz, 16-bit, Mono PCM
        wfx = WAVEFORMATEX(1, 1, 16000, 32000, 2, 16, 0)
        mfplat.MFInitMediaTypeFromWaveFormatEx(pType, byref(wfx), ctypes.sizeof(wfx))
        call_v(pReader, 7, c_int, [c_uint32, c_void_p, c_void_p], c_uint32(0xFFFFFFFD), None, pType)
        call_v(pType, 2, c_ulong, [])

        audio_bytes = bytearray()
        actual_stream = c_uint32()
        flags = c_uint32()
        timestamp = c_longlong()
        pSample = c_void_p()

        while True:
            pSample.value = None
            hr = call_v(
                pReader, 9, c_int,
                [c_uint32, c_uint32, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_longlong), POINTER(c_void_p)],
                c_uint32(0xFFFFFFFD),
                c_uint32(0),
                byref(actual_stream),
                byref(flags),
                byref(timestamp),
                byref(pSample),
            )
            if hr != 0 or (flags.value & 0x00000001) or not pSample.value:
                break

            pBuf = c_void_p()
            hr = call_v(pSample, 41, c_int, [POINTER(c_void_p)], byref(pBuf))
            if hr == 0 and pBuf.value:
                pData = c_void_p()
                maxLen = c_uint32()
                curLen = c_uint32()
                hr = call_v(
                    pBuf, 3, c_int,
                    [POINTER(c_void_p), POINTER(c_uint32), POINTER(c_uint32)],
                    byref(pData), byref(maxLen), byref(curLen)
                )
                if hr == 0 and pData.value:
                    if curLen.value > 0:
                        audio_bytes.extend(ctypes.string_at(pData.value, curLen.value))
                    call_v(pBuf, 4, c_int, [])
                call_v(pBuf, 2, c_ulong, [])
            call_v(pSample, 2, c_ulong, [])

        call_v(pReader, 2, c_ulong, [])
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return (pcm16.astype(np.float32) / 32768.0).astype(np.float32)
    finally:
        mfplat.MFShutdown()
        ole32.CoUninitialize()


def load_audio(file_path: str) -> np.ndarray:
    """Universal audio loader supporting .wav, .mp3, .m4a, .aac, .flac, .ogg."""
    ext = Path(file_path).suffix.lower()

    # Fast path for soundfile-supported formats
    if ext in [".wav", ".flac", ".ogg", ".mp3"]:
        try:
            data, sr = sf.read(file_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr), 16000)
                data = resample_poly(data, up=16000 // g, down=int(sr) // g).astype(np.float32)
            return data
        except Exception:
            pass

    # Windows Media Foundation decoder (handles .m4a, .aac, .mp4, .wma, etc.)
    try:
        return _read_audio_windows_mf(file_path)
    except Exception as e_mf:
        # Fallback: audioread
        try:
            import audioread
            with audioread.audio_open(file_path) as f:
                sr = f.samplerate
                channels = f.channels
                buf = bytearray()
                for chunk in f:
                    buf.extend(chunk)
                pcm16 = np.frombuffer(buf, dtype=np.int16)
                data = pcm16.astype(np.float32) / 32768.0
                if channels > 1:
                    data = data.reshape(-1, channels).mean(axis=1)
                if sr != 16000:
                    from scipy.signal import resample_poly
                    g = math.gcd(int(sr), 16000)
                    data = resample_poly(data, up=16000 // g, down=int(sr) // g).astype(np.float32)
                return data.astype(np.float32)
        except Exception as e_ar:
            raise ValueError(f"Could not decode audio file: {e_mf}") from e_ar


def classify_audio(file_path: str) -> dict[str, object]:
    """Reads audio file, slices into 4s windows, and runs model."""
    data = load_audio(file_path)
    total_samples = len(data)
    win_len = 64000  # 4 seconds at 16 kHz
    hop_len = 16000  # 1 second hop

    if total_samples < win_len:
        # Repeat-pad shorter audio
        reps = int(math.ceil(win_len / max(1, total_samples)))
        padded = np.tile(data, reps)[:win_len].astype(np.float32)
        windows = [padded]
    else:
        windows = []
        for start in range(0, total_samples - win_len + 1, hop_len):
            windows.append(data[start : start + win_len])
        if not windows:
            windows = [data[:win_len]]

    scores = []
    for w in windows:
        # RMS normalize to -26 dBFS
        rms = float(np.sqrt(np.mean(w**2)))
        target_rms = float(10.0 ** (-26.0 / 20.0))
        if rms > 1e-9:
            w_norm = (w * (target_rms / rms)).astype(np.float32)
        else:
            w_norm = w.astype(np.float32)

        tensor = torch.from_numpy(w_norm).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logit = model(tensor).item()
        scores.append(logit - THRESHOLD)

    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    is_clone = bool(mean_score < 0.0 or min_score < -2.0)

    return {
        "duration": round(total_samples / 16000.0, 2),
        "mean_score": round(mean_score, 2),
        "min_score": round(min_score, 2),
        "is_clone": is_clone,
        "window_count": len(scores),
    }


# Desktop GUI
class DetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Voice Clone Detector")
        self.root.geometry("620x460")
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f4f4")

        # Title
        lbl_title = tk.Label(
            root,
            text="AI Voice Clone Detector",
            font=("Arial", 18, "bold"),
            bg="#f4f4f4",
            fg="#222222",
        )
        lbl_title.pack(pady=(20, 4))

        lbl_sub = tk.Label(
            root,
            text=f"Powered by PyTorch (models/best.pt) on {DEVICE}",
            font=("Arial", 10),
            bg="#f4f4f4",
            fg="#666666",
        )
        lbl_sub.pack(pady=(0, 16))

        # Select Button
        self.btn_select = tk.Button(
            root,
            text="Select Audio File (.wav, .mp3, .m4a, .flac)",
            font=("Arial", 12, "bold"),
            bg="#0066cc",
            fg="#ffffff",
            activebackground="#004c99",
            activeforeground="#ffffff",
            padx=20,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT,
            command=self.select_file,
        )
        self.btn_select.pack(pady=10)

        # File label
        self.lbl_file = tk.Label(
            root,
            text="No file selected",
            font=("Arial", 10, "italic"),
            bg="#f4f4f4",
            fg="#555555",
        )
        self.lbl_file.pack(pady=(4, 16))

        # Result Card Frame
        self.card = tk.Frame(root, bg="#ffffff", bd=1, relief=tk.SOLID, padx=20, pady=20)
        self.card.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 25))

        self.lbl_verdict = tk.Label(
            self.card,
            text="WAITING FOR AUDIO",
            font=("Arial", 16, "bold"),
            bg="#ffffff",
            fg="#888888",
        )
        self.lbl_verdict.pack(pady=(4, 8))

        self.lbl_details = tk.Label(
            self.card,
            text="Select an audio recording to run neural clone detection.",
            font=("Arial", 11),
            bg="#ffffff",
            fg="#444444",
            wraplength=520,
        )
        self.lbl_details.pack(pady=(0, 14))

        self.lbl_scores = tk.Label(
            self.card,
            text="Mean Score: --  |  Lowest Window: --",
            font=("Consolas", 12, "bold"),
            bg="#ffffff",
            fg="#333333",
        )
        self.lbl_scores.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose Audio File",
            filetypes=[
                ("All Supported Audio", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.wma"),
                ("M4A Voice Recording (*.m4a)", "*.m4a"),
                ("WAV Audio (*.wav)", "*.wav"),
                ("MP3 Audio (*.mp3)", "*.mp3"),
                ("FLAC Lossless (*.flac)", "*.flac"),
                ("All Files (*.*)", "*.*"),
            ],
        )
        if not file_path:
            return

        p = Path(file_path)
        self.lbl_file.config(text=f"Selected: {p.name}")
        self.root.update()

        try:
            res = classify_audio(str(p))

            if res["is_clone"]:
                self.lbl_verdict.config(text="🚨 AI VOICE CLONE DETECTED", fg="#cc0000")
                self.lbl_details.config(
                    text="Synthetic voice clone patterns detected. This voice matches AI-generated speech."
                )
            else:
                self.lbl_verdict.config(text="🟢 AUTHENTIC NATURAL VOICE", fg="#008800")
                self.lbl_details.config(
                    text="Natural human speech dynamics verified. No clone anomalies found."
                )

            sign = "+" if res["mean_score"] > 0 else ""
            min_sign = "+" if res["min_score"] > 0 else ""
            self.lbl_scores.config(
                text=f"Mean Score: {sign}{res['mean_score']:.2f}  |  Lowest: {min_sign}{res['min_score']:.2f}  |  Duration: {res['duration']}s"
            )

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze audio: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI Mode
        fpath = sys.argv[1]
        print(f"[*] Analyzing {fpath}...")
        res = classify_audio(fpath)
        verdict = "AI CLONE DETECTED" if res["is_clone"] else "AUTHENTIC HUMAN VOICE"
        print(f"Verdict:    {verdict}")
        print(f"Mean Score: {res['mean_score']:+.2f}")
        print(f"Lowest:     {res['min_score']:+.2f}")
        print(f"Duration:   {res['duration']}s ({res['window_count']} windows)")
    else:
        # Native Desktop GUI Mode
        root = tk.Tk()
        app = DetectorApp(root)
        root.mainloop()
