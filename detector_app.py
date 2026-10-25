"""Call Shield - Native Desktop Voice Clone Detector & Forensic Inspector.

Matches the editorial architectural design system of the Call Shield web application:
Warm Linen (#f9ebd7), Heritage Deep Green (#046a38), Terracotta Vermilion (#e34222),
zero corner roundings, tactile physical ink shadows (4px 4px 0 #046a38), and print craftsman typography.

Uses models/best.pt directly with PyTorch to detect AI voice clones.
100% standalone desktop application with zero web server or browser dependencies.

Usage:
  python detector_app.py                      # Launches native desktop GUI
  python detector_app.py <path_to_audio_file> # Runs forensic CLI analysis
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_int, c_longlong, c_ulong, c_void_p, c_uint32, cast, POINTER, Structure, wintypes
import math
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, TypedDict, cast as type_cast
import webbrowser

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch

from training.model import VoiceCloneDetectorModel


class WindowRecord(TypedDict):
    index: int
    time_range: str
    time_start: float
    status: str
    score: float
    is_silent: bool


class ClassificationResult(TypedDict):
    status: str
    duration: float
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    is_clone: bool
    window_count: int
    active_windows: int
    silent_windows: int
    flagged_ratio: float
    confidence: int
    details: str
    window_records: list[WindowRecord]
    raw_data: np.ndarray


# ==============================================================================
# Model Loading & Torch Device Configuration
# ==============================================================================

MODEL_PATH = BASE_DIR / "models" / "best.pt"
if not MODEL_PATH.is_file():
    print(f"[!] Error: Model not found at: {MODEL_PATH}")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
THRESHOLD = float(checkpoint.get("threshold", 0.331)) if isinstance(checkpoint, dict) else 0.331

model = VoiceCloneDetectorModel().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

# Check for Windows sound playback module
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    winsound = None  # type: ignore[assignment]
    HAS_WINSOUND = False


# ==============================================================================
# Audio Ingestion & Preprocessing Routines
# ==============================================================================

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

    def call_v(obj: Any, idx: int, restype: Any, argtypes: list[Any], *args: Any) -> Any:
        vtbl = type_cast(Any, cast(obj, POINTER(POINTER(c_void_p))).contents)
        proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
        return proto(vtbl[idx])(obj, *args)

    try:
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

    try:
        return _read_audio_windows_mf(file_path)
    except Exception as e_mf:
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


def is_window_silent(w: np.ndarray, threshold_dbfs: float = -42.0) -> bool:
    """Returns True if window RMS is below threshold_dbfs or peak amplitude is near zero."""
    mean_sq = float(np.mean(w**2))
    if mean_sq <= 1e-12:
        return True
    rms = math.sqrt(mean_sq)
    dbfs = 20.0 * math.log10(max(1e-9, rms))
    peak = float(np.max(np.abs(w)))
    return bool(dbfs < threshold_dbfs or peak < 0.012)


def classify_audio(file_path: str) -> ClassificationResult:
    """Reads audio file, applies VAD silence filtering, slices into 4s windows, and runs neural classifier."""
    data = load_audio(file_path)
    total_samples = len(data)
    win_len = 64000  # 4 seconds at 16 kHz
    hop_len = 16000  # 1 second hop

    if total_samples < win_len:
        reps = int(math.ceil(win_len / max(1, total_samples)))
        padded = np.tile(data, reps)[:win_len].astype(np.float32)
        windows = [padded]
    else:
        windows = []
        for start in range(0, total_samples - win_len + 1, hop_len):
            windows.append(data[start : start + win_len])
        if not windows:
            windows = [data[:win_len]]

    active_scores: list[float] = []
    silent_windows = 0
    window_records: list[WindowRecord] = []

    for idx, w in enumerate(windows):
        t_start = round(idx * (hop_len / 16000.0), 2)
        t_end = round(t_start + (win_len / 16000.0), 2)

        if is_window_silent(w, threshold_dbfs=-42.0):
            silent_windows += 1
            window_records.append({
                "index": idx + 1,
                "time_range": f"{t_start:.1f}s - {t_end:.1f}s",
                "time_start": t_start,
                "status": "SILENCE",
                "score": 0.0,
                "is_silent": True,
            })
            continue

        w_clean = w - float(np.mean(w))
        rms = float(np.sqrt(np.mean(w_clean**2)))
        target_rms = float(10.0 ** (-26.0 / 20.0))
        if rms > 1e-9:
            w_norm = (w_clean * (target_rms / rms)).astype(np.float32)
        else:
            w_norm = w_clean.astype(np.float32)

        tensor = torch.from_numpy(w_norm).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logit = model(tensor).item()
        margin = logit - THRESHOLD
        active_scores.append(margin)

        if margin < -0.5:
            w_status = "ALERT"
        elif margin < 0.0:
            w_status = "WARNING"
        else:
            w_status = "OK"

        window_records.append({
            "index": idx + 1,
            "time_range": f"{t_start:.1f}s - {t_end:.1f}s",
            "time_start": t_start,
            "status": w_status,
            "score": round(margin, 2),
            "is_silent": False,
        })

    if not active_scores:
        return {
            "status": "SILENCE",
            "duration": round(total_samples / 16000.0, 2),
            "mean_score": 0.0,
            "median_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "is_clone": False,
            "window_count": len(windows),
            "active_windows": 0,
            "silent_windows": silent_windows,
            "flagged_ratio": 0.0,
            "confidence": 0,
            "details": "No audible speech detected. The audio contains only background silence or quiet ambient noise.",
            "window_records": window_records,
            "raw_data": data,
        }

    mean_score = float(np.mean(active_scores))
    median_score = float(np.median(active_scores))
    min_score = float(np.min(active_scores))
    max_score = float(np.max(active_scores))
    flagged_count = sum(1 for s in active_scores if s < 0.0)
    flag_ratio = flagged_count / len(active_scores)

    if (flag_ratio >= 0.40 and mean_score < 0.5) or mean_score < -1.0:
        is_clone = True
        status = "ALERT"
        conf = min(99, int(55 + min(44, abs(mean_score) * 4 + flag_ratio * 35)))
        details = (
            f"Synthetic voice clone patterns detected in {flagged_count}/{len(active_scores)} "
            f"({flag_ratio * 100:.0f}%) of active speech windows. Acoustic score exceeds clone threshold."
        )
    else:
        is_clone = False
        status = "OK"
        conf = min(99, int(55 + min(44, max(0.0, mean_score) * 4 + (1.0 - flag_ratio) * 35)))
        details = (
            f"Natural human speech dynamics verified across {len(active_scores) - flagged_count}/"
            f"{len(active_scores)} ({(1.0 - flag_ratio) * 100:.0f}%) of speech windows. No synthetic artifacts detected."
        )

    return {
        "status": status,
        "duration": round(total_samples / 16000.0, 2),
        "mean_score": round(mean_score, 2),
        "median_score": round(median_score, 2),
        "min_score": round(min_score, 2),
        "max_score": round(max_score, 2),
        "is_clone": is_clone,
        "window_count": len(windows),
        "active_windows": len(active_scores),
        "silent_windows": silent_windows,
        "flagged_ratio": round(flag_ratio * 100, 1),
        "confidence": conf,
        "details": details,
        "window_records": window_records,
        "raw_data": data,
    }


# ==============================================================================
# UI Design System Tokens (Matching client/style.css)
# ==============================================================================

BG_PRIMARY = "#f9ebd7"          # Warm Linen
BG_CARD = "#fffdf9"             # Architectural Off-White Surface
BG_SURFACE = "#f3e3cb"          # Warm Cream Ledger Surface
BG_SURFACE_INSET = "#ede0cb"    # Darker Inset Linen
BORDER_DARK = "#046a38"         # Heritage Deep Green Border
BORDER_INK = "#1b261e"          # Tactile Ink Dark
BORDER_SUBTLE = "#d8c5aa"       # Delicate Linen Border

TEXT_PRIMARY = "#1b261e"        # Deep Charcoal Ink
TEXT_SECONDARY = "#3d4f42"      # Muted Deep Forest
TEXT_MUTED = "#6b7c70"          # Architectural Slate

COLOR_PRIMARY = "#046a38"       # Heritage Deep Green
COLOR_PRIMARY_HOVER = "#03522b" # Dark Forest Green
COLOR_ALERT = "#e34222"         # Terracotta Vermilion
COLOR_WARN = "#c25e00"          # Warm Amber
COLOR_OK_BG = "#eef6f0"         # Pale Mint Green
COLOR_WARN_BG = "#fbf3ea"       # Pale Terracotta
COLOR_ALERT_BG = "#faeae7"      # Pale Vermilion

# Font Families (Available natively on Windows)
FONT_HEADING = ("Georgia", 18, "bold")
FONT_HEADING_SM = ("Georgia", 13, "bold")
FONT_EDITORIAL = ("Georgia", 10, "italic")
FONT_BODY = ("Inter", 10)
FONT_BODY_BOLD = ("Inter", 10, "bold")
FONT_BODY_SM = ("Inter", 9)
FONT_BODY_SM_BOLD = ("Inter", 9, "bold")
FONT_BODY_CAPS = ("Inter", 8, "bold")
FONT_MONO = ("Consolas", 10, "bold")
FONT_MONO_LG = ("Consolas", 18, "bold")
FONT_MONO_SM = ("Consolas", 8, "bold")


# ==============================================================================
# Tactile Architectural Widgets
# ==============================================================================

class SolidCard(tk.Frame):
    """Compound frame providing the signature tactile solid ink drop shadow (e.g. 4px 4px 0 #046a38)."""

    def __init__(
        self,
        parent: tk.Widget,
        shadow_color: str = BORDER_DARK,
        offset: int = 4,
        bg_card: str = BG_CARD,
        border_color: str = BORDER_DARK,
        border_width: int = 2,
        **kwargs,
    ):
        super().__init__(parent, bg=shadow_color)
        self.inner = tk.Frame(
            self,
            bg=bg_card,
            highlightthickness=border_width,
            highlightbackground=border_color,
            **kwargs,
        )
        self.inner.pack(fill=tk.BOTH, expand=True, padx=(0, offset), pady=(0, offset))


class TactileButton(tk.Frame):
    """Zero-radius physical press button with crisp ink drop shadow and translation on press."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command=None,
        bg: str = COLOR_PRIMARY,
        fg: str = "#f9ebd7",
        hover_bg: str = COLOR_PRIMARY_HOVER,
        shadow: str = BORDER_INK,
        font: tuple = FONT_BODY_BOLD,
        offset: int = 3,
        padx: int = 16,
        pady: int = 9,
        **kwargs,
    ):
        super().__init__(parent, bg=shadow, **kwargs)
        self.command = command
        self.normal_bg = bg
        self.hover_bg = hover_bg
        self.offset = offset

        self.inner = tk.Label(
            self,
            text=text,
            font=font,
            bg=bg,
            fg=fg,
            cursor="hand2",
            padx=padx,
            pady=pady,
            highlightthickness=2,
            highlightbackground=shadow,
        )
        self.inner.pack(fill=tk.BOTH, expand=True, padx=(0, offset), pady=(0, offset))

        self.inner.bind("<Enter>", lambda e: self.inner.configure(bg=self.hover_bg))
        self.inner.bind("<Leave>", lambda e: self.inner.configure(bg=self.normal_bg))
        self.inner.bind("<ButtonPress-1>", self._on_press)
        self.inner.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event):
        self.inner.pack_configure(padx=(self.offset, 0), pady=(self.offset, 0))

    def _on_release(self, event):
        self.inner.pack_configure(padx=(0, self.offset), pady=(0, self.offset))
        if self.command:
            self.command()

    def set_text(self, text: str):
        self.inner.configure(text=text)

    def set_bg(self, bg: str, hover_bg: str | None = None):
        self.normal_bg = bg
        self.hover_bg = hover_bg or bg
        self.inner.configure(bg=bg)


class TelemetryTile(tk.Frame):
    """Stamped card displaying telemetry metric label, large mono value, and subtitle."""

    def __init__(self, parent: tk.Widget, label: str, value: str, subtext: str):
        super().__init__(parent, bg=BORDER_SUBTLE)
        self.inner = tk.Frame(
            self,
            bg=BG_SURFACE,
            highlightthickness=2,
            highlightbackground=BORDER_SUBTLE,
            padx=12,
            pady=10,
        )
        self.inner.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 2))

        self.lbl_label = tk.Label(self.inner, text=label, font=FONT_BODY_CAPS, bg=BG_SURFACE, fg=TEXT_MUTED)
        self.lbl_label.pack(anchor="w")

        self.lbl_value = tk.Label(self.inner, text=value, font=FONT_MONO_LG, bg=BG_SURFACE, fg=COLOR_PRIMARY)
        self.lbl_value.pack(anchor="w", pady=(2, 2))

        self.lbl_sub = tk.Label(self.inner, text=subtext, font=("Inter", 7), bg=BG_SURFACE, fg=TEXT_MUTED)
        self.lbl_sub.pack(anchor="w")

    def update_value(self, value: str, fg: str | None = None, subtext: str | None = None) -> None:
        self.lbl_value.configure(text=value)
        if fg is not None:
            self.lbl_value.configure(fg=fg)
        if subtext is not None:
            self.lbl_sub.configure(text=subtext)


# ==============================================================================
# Native Desktop Application
# ==============================================================================

class CallShieldApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Call Shield - Real-Time AI Voice Clone Defense")
        self.root.geometry("980x880")
        self.root.minsize(860, 680)
        self.root.configure(bg=BG_PRIMARY)

        self.current_audio_file: str | None = None
        self.temp_wav_path: str | None = None
        self.is_playing = False
        self.analysis_result: ClassificationResult | None = None

        # Build UI Layout
        self._setup_scrollable_container()
        self._build_header()
        self._build_nav_tabs()
        self._build_inspector_card()
        self._build_footer()

    def _setup_scrollable_container(self):
        """Scrollable canvas with warm linen background and subtle drafting grid."""
        self.canvas = tk.Canvas(self.root, bg=BG_PRIMARY, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=BG_PRIMARY)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel support
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Centered container frame matching max-width: 920px
        self.app_container = tk.Frame(self.scroll_frame, bg=BG_PRIMARY)
        self.app_container.pack(padx=28, pady=24, fill=tk.BOTH, expand=True)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # --------------------------------------------------------------------------
    # Top Header & Brand Bar
    # --------------------------------------------------------------------------

    def _build_header(self):
        header_card = SolidCard(
            self.app_container,
            shadow_color=BORDER_DARK,
            offset=4,
            bg_card=BG_CARD,
            border_color=BORDER_DARK,
            border_width=2,
            padx=18,
            pady=14,
        )
        header_card.pack(fill=tk.X, pady=(0, 16))

        h_frame = header_card.inner
        left_brand = tk.Frame(h_frame, bg=BG_CARD)
        left_brand.pack(side=tk.LEFT, fill=tk.Y)

        # Brand Icon (Deep Green Square with Shield)
        icon_outer = tk.Frame(left_brand, bg=BORDER_INK)
        icon_outer.pack(side=tk.LEFT, padx=(0, 12))
        icon_inner = tk.Frame(
            icon_outer,
            bg=COLOR_PRIMARY,
            width=46,
            height=46,
            highlightthickness=2,
            highlightbackground=BORDER_INK,
        )
        icon_inner.pack(padx=(0, 2), pady=(0, 2))
        icon_inner.pack_propagate(False)

        lbl_shield = tk.Label(icon_inner, text="🛡️", font=("Segoe UI Emoji", 18), bg=COLOR_PRIMARY, fg="#f9ebd7")
        lbl_shield.pack(expand=True)

        # Brand Typography
        brand_text_frame = tk.Frame(left_brand, bg=BG_CARD)
        brand_text_frame.pack(side=tk.LEFT, fill=tk.Y, pady=2)

        lbl_title = tk.Label(
            brand_text_frame,
            text="Call Shield",
            font=("Georgia", 19, "bold"),
            bg=BG_CARD,
            fg=COLOR_PRIMARY,
        )
        lbl_title.pack(anchor="w")

        lbl_subtitle = tk.Label(
            brand_text_frame,
            text="REAL-TIME AI VOICE CLONE DEFENSE SYSTEM",
            font=FONT_BODY_CAPS,
            bg=BG_CARD,
            fg=TEXT_MUTED,
        )
        lbl_subtitle.pack(anchor="w")

        # Right Status Badge
        badge_outer = tk.Frame(h_frame, bg=BORDER_DARK)
        badge_outer.pack(side=tk.RIGHT, pady=4)
        badge_inner = tk.Frame(
            badge_outer,
            bg=BG_SURFACE,
            highlightthickness=2,
            highlightbackground=BORDER_DARK,
            padx=12,
            pady=6,
        )
        badge_inner.pack(padx=(0, 2), pady=(0, 2))

        lbl_badge_status = tk.Label(
            badge_inner,
            text="🟢 LIVE SHIELD ACTIVE",
            font=FONT_MONO,
            bg=BG_SURFACE,
            fg=COLOR_PRIMARY,
        )
        lbl_badge_status.pack(anchor="e")

        lbl_badge_meta = tk.Label(
            badge_inner,
            text=f"Threshold: {THRESHOLD:.3f} • {DEVICE}",
            font=FONT_MONO_SM,
            bg=BG_SURFACE,
            fg=TEXT_MUTED,
        )
        lbl_badge_meta.pack(anchor="e")

    # --------------------------------------------------------------------------
    # Navigation Mode Tabs
    # --------------------------------------------------------------------------

    def _build_nav_tabs(self):
        tabs_frame = tk.Frame(self.app_container, bg=BG_PRIMARY)
        tabs_frame.pack(fill=tk.X, pady=(0, 16))

        # Tab 1: Audio File Inspector (Active)
        tab_inspect_outer = tk.Frame(tabs_frame, bg=BORDER_INK)
        tab_inspect_outer.pack(side=tk.LEFT, padx=(0, 10))
        tab_inspect = tk.Label(
            tab_inspect_outer,
            text="🛡️ AUDIO FORENSIC INSPECTOR",
            font=FONT_BODY_BOLD,
            bg=COLOR_PRIMARY,
            fg="#f9ebd7",
            padx=16,
            pady=8,
            highlightthickness=2,
            highlightbackground=BORDER_INK,
        )
        tab_inspect.pack(padx=(0, 3), pady=(0, 3))

        # Tab 2: Live Web Shield (Quick launcher to web dashboard)
        btn_live_web = TactileButton(
            tabs_frame,
            text="🌐 OPEN LIVE WEB SHIELD",
            command=self._open_web_shield,
            bg=BG_SURFACE,
            fg=TEXT_PRIMARY,
            hover_bg="#fbf4ea",
            shadow=BORDER_DARK,
            font=FONT_BODY_BOLD,
            padx=14,
            pady=7,
        )
        btn_live_web.pack(side=tk.LEFT)

    def _open_web_shield(self):
        webbrowser.open("http://127.0.0.1:8000")

    # --------------------------------------------------------------------------
    # Main Inspector Instrument Panel
    # --------------------------------------------------------------------------

    def _build_inspector_card(self):
        self.inspector_card = SolidCard(
            self.app_container,
            shadow_color=BORDER_DARK,
            offset=4,
            bg_card=BG_CARD,
            border_color=BORDER_DARK,
            border_width=2,
            padx=24,
            pady=22,
        )
        self.inspector_card.pack(fill=tk.BOTH, expand=True, pady=(0, 16))

        card = self.inspector_card.inner

        # Card Top Green Accent Line
        accent_line = tk.Frame(card, bg=COLOR_PRIMARY, height=4)
        accent_line.pack(fill=tk.X, pady=(0, 16))

        # Panel Header
        p_header = tk.Frame(card, bg=BG_CARD)
        p_header.pack(fill=tk.X, pady=(0, 16))

        p_title_frame = tk.Frame(p_header, bg=BG_CARD)
        p_title_frame.pack(side=tk.LEFT)

        lbl_card_title = tk.Label(
            p_title_frame,
            text="Audio Forensic Inspector",
            font=FONT_HEADING,
            bg=BG_CARD,
            fg=COLOR_PRIMARY,
        )
        lbl_card_title.pack(anchor="w")

        lbl_card_sub = tk.Label(
            p_title_frame,
            text="Neural acoustic dynamics & synthetic voice clone detection (sliding 4s windows)",
            font=FONT_EDITORIAL,
            bg=BG_CARD,
            fg=TEXT_SECONDARY,
        )
        lbl_card_sub.pack(anchor="w")

        # Stamped Status Level Chip (Right)
        self.level_chip_outer = tk.Frame(p_header, bg=BORDER_DARK)
        self.level_chip_outer.pack(side=tk.RIGHT)
        self.level_chip_inner = tk.Frame(
            self.level_chip_outer,
            bg=BG_SURFACE,
            highlightthickness=2,
            highlightbackground=BORDER_DARK,
            padx=14,
            pady=7,
        )
        self.level_chip_inner.pack(padx=(0, 3), pady=(0, 3))

        self.lbl_level_chip = tk.Label(
            self.level_chip_inner,
            text="STANDBY",
            font=("Consolas", 12, "bold"),
            bg=BG_SURFACE,
            fg=TEXT_MUTED,
        )
        self.lbl_level_chip.pack()

        # ----------------------------------------------------------------------
        # Dropzone & File Selection Area
        # ----------------------------------------------------------------------
        dropzone = tk.Frame(
            card,
            bg=BG_SURFACE,
            highlightthickness=2,
            highlightbackground=COLOR_PRIMARY,
            padx=20,
            pady=20,
        )
        dropzone.pack(fill=tk.X, pady=(0, 18))

        lbl_dz_title = tk.Label(
            dropzone,
            text="Select an audio recording to run neural clone defense",
            font=FONT_HEADING_SM,
            bg=BG_SURFACE,
            fg=COLOR_PRIMARY,
        )
        lbl_dz_title.pack(pady=(0, 4))

        lbl_dz_sub = tk.Label(
            dropzone,
            text="Supports WAV, MP3, M4A, AAC, FLAC, OGG (In-Memory Processing • Zero Retention)",
            font=FONT_BODY_SM,
            bg=BG_SURFACE,
            fg=TEXT_SECONDARY,
        )
        lbl_dz_sub.pack(pady=(0, 14))

        # Main Tactile File Picker Button
        self.btn_select_file = TactileButton(
            dropzone,
            text="📂 SELECT AUDIO FILE (BROWSE)",
            command=self.select_file,
            bg=COLOR_PRIMARY,
            fg="#f9ebd7",
            hover_bg=COLOR_PRIMARY_HOVER,
            shadow=BORDER_INK,
            font=("Inter", 11, "bold"),
            offset=3,
            padx=24,
            pady=10,
        )
        self.btn_select_file.pack(pady=(0, 12))

        # Preset Chips Row
        presets_frame = tk.Frame(dropzone, bg=BG_SURFACE)
        presets_frame.pack()

        lbl_presets = tk.Label(
            presets_frame,
            text="QUICK SAMPLES:",
            font=FONT_BODY_CAPS,
            bg=BG_SURFACE,
            fg=TEXT_MUTED,
        )
        lbl_presets.pack(side=tk.LEFT, padx=(0, 8))

        btn_sample_real = TactileButton(
            presets_frame,
            text="Sample: Human Voice",
            command=self._load_sample_real,
            bg=BG_CARD,
            fg=COLOR_PRIMARY,
            hover_bg=COLOR_PRIMARY,
            shadow=BORDER_DARK,
            font=FONT_MONO_SM,
            offset=2,
            padx=10,
            pady=4,
        )
        btn_sample_real.pack(side=tk.LEFT, padx=(0, 6))

        btn_sample_clone = TactileButton(
            presets_frame,
            text="Sample: OpenAI Clone",
            command=self._load_sample_clone,
            bg=BG_CARD,
            fg=COLOR_ALERT,
            hover_bg=COLOR_ALERT,
            shadow=COLOR_ALERT,
            font=FONT_MONO_SM,
            offset=2,
            padx=10,
            pady=4,
        )
        btn_sample_clone.pack(side=tk.LEFT, padx=(0, 6))

        btn_sample_silence = TactileButton(
            presets_frame,
            text="Sample: Room Silence",
            command=self._load_sample_silence,
            bg=BG_CARD,
            fg=TEXT_MUTED,
            hover_bg=TEXT_MUTED,
            shadow=BORDER_SUBTLE,
            font=FONT_MONO_SM,
            offset=2,
            padx=10,
            pady=4,
        )
        btn_sample_silence.pack(side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Audio File Metadata & Playback Bar
        # ----------------------------------------------------------------------
        self.player_bar = tk.Frame(
            card,
            bg=BG_SURFACE,
            highlightthickness=2,
            highlightbackground=BORDER_DARK,
            padx=16,
            pady=12,
        )
        self.player_bar.pack(fill=tk.X, pady=(0, 18))

        player_left = tk.Frame(self.player_bar, bg=BG_SURFACE)
        player_left.pack(side=tk.LEFT)

        self.lbl_file_name = tk.Label(
            player_left,
            text="No audio file loaded",
            font=FONT_HEADING_SM,
            bg=BG_SURFACE,
            fg=COLOR_PRIMARY,
        )
        self.lbl_file_name.pack(anchor="w")

        self.lbl_file_specs = tk.Label(
            player_left,
            text="Duration: --  •  Sample Rate: 16 kHz  •  Sliding Windows: --",
            font=FONT_MONO_SM,
            bg=BG_SURFACE,
            fg=TEXT_SECONDARY,
        )
        self.lbl_file_specs.pack(anchor="w")

        # Audio Play / Stop Buttons
        player_right = tk.Frame(self.player_bar, bg=BG_SURFACE)
        player_right.pack(side=tk.RIGHT)

        self.btn_play = TactileButton(
            player_right,
            text="▶ Play Audio",
            command=self._toggle_audio_playback,
            bg=COLOR_PRIMARY,
            fg="#f9ebd7",
            hover_bg=COLOR_PRIMARY_HOVER,
            shadow=BORDER_INK,
            font=FONT_BODY_SM_BOLD,
            offset=2,
            padx=12,
            pady=5,
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = TactileButton(
            player_right,
            text="⏹ Stop",
            command=self._stop_audio_playback,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            hover_bg="#fbf4ea",
            shadow=BORDER_INK,
            font=FONT_BODY_SM_BOLD,
            offset=2,
            padx=10,
            pady=5,
        )
        self.btn_stop.pack(side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Verdict Banner
        # ----------------------------------------------------------------------
        self.verdict_outer = tk.Frame(card, bg=BORDER_DARK)
        self.verdict_outer.pack(fill=tk.X, pady=(0, 18))

        self.verdict_inner = tk.Frame(
            self.verdict_outer,
            bg=COLOR_OK_BG,
            highlightthickness=2,
            highlightbackground=COLOR_PRIMARY,
            padx=18,
            pady=16,
        )
        self.verdict_inner.pack(fill=tk.X, padx=(0, 4), pady=(0, 4))

        self.verdict_badge = tk.Label(
            self.verdict_inner,
            text="STANDBY",
            font=("Consolas", 11, "bold"),
            bg=COLOR_PRIMARY,
            fg="#f9ebd7",
            padx=12,
            pady=6,
            highlightthickness=2,
            highlightbackground=BORDER_INK,
        )
        self.verdict_badge.pack(side=tk.LEFT, padx=(0, 16))

        verdict_text_frame = tk.Frame(self.verdict_inner, bg=COLOR_OK_BG)
        verdict_text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_verdict_title = tk.Label(
            verdict_text_frame,
            text="Awaiting Audio Recording",
            font=FONT_HEADING_SM,
            bg=COLOR_OK_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
        )
        self.lbl_verdict_title.pack(fill=tk.X, pady=(0, 2))

        self.lbl_verdict_desc = tk.Label(
            verdict_text_frame,
            text="Select an audio recording or sample above to run multi-window neural clone defense.",
            font=FONT_BODY,
            bg=COLOR_OK_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            wraplength=640,
            justify=tk.LEFT,
        )
        self.lbl_verdict_desc.pack(fill=tk.X)

        # ----------------------------------------------------------------------
        # Telemetry Metrics Grid (4 Stamped Ledger Tiles)
        # ----------------------------------------------------------------------
        grid_frame = tk.Frame(card, bg=BG_CARD)
        grid_frame.pack(fill=tk.X, pady=(0, 18))

        self.tile_score = TelemetryTile(grid_frame, "ACOUSTIC SCORE", "+0.00", "Classifier Margin")
        self.tile_score.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.tile_min = TelemetryTile(grid_frame, "LOWEST WINDOW", "+0.00", "Peak Suspect Score")
        self.tile_min.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.tile_flagged = TelemetryTile(grid_frame, "THREAT WINDOWS", "0 / 0", "Flagged Speech Wins")
        self.tile_flagged.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.tile_conf = TelemetryTile(grid_frame, "CONFIDENCE", "-- %", "Ensemble Margin")
        self.tile_conf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # ----------------------------------------------------------------------
        # Window-by-Window Threat Progression Track (Interactive Canvas)
        # ----------------------------------------------------------------------
        track_section = tk.Frame(
            card,
            bg=BG_SURFACE_INSET,
            highlightthickness=2,
            highlightbackground=BORDER_DARK,
            padx=14,
            pady=12,
        )
        track_section.pack(fill=tk.X, pady=(0, 18))

        track_header = tk.Frame(track_section, bg=BG_SURFACE_INSET)
        track_header.pack(fill=tk.X, pady=(0, 8))

        lbl_track_title = tk.Label(
            track_header,
            text="WINDOW-BY-WINDOW THREAT PROGRESSION (SLIDING 4S BLOCKS)",
            font=FONT_MONO_SM,
            bg=BG_SURFACE_INSET,
            fg=COLOR_PRIMARY,
        )
        lbl_track_title.pack(side=tk.LEFT)

        # Legend
        legend_frame = tk.Frame(track_header, bg=BG_SURFACE_INSET)
        legend_frame.pack(side=tk.RIGHT)

        for name, col in [("Authentic", COLOR_PRIMARY), ("Warning", COLOR_WARN), ("Alert", COLOR_ALERT), ("Silent", BORDER_SUBTLE)]:
            dot = tk.Frame(legend_frame, bg=col, width=8, height=8, highlightthickness=1, highlightbackground=BORDER_INK)
            dot.pack(side=tk.LEFT, padx=(6, 2))
            lbl = tk.Label(legend_frame, text=name, font=FONT_MONO_SM, bg=BG_SURFACE_INSET, fg=TEXT_MUTED)
            lbl.pack(side=tk.LEFT, padx=(0, 6))

        self.canvas_track = tk.Canvas(
            track_section,
            height=90,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER_DARK,
        )
        self.canvas_track.pack(fill=tk.X)

        # ----------------------------------------------------------------------
        # Diagnostics Table Section (Collapsible Treeview)
        # ----------------------------------------------------------------------
        self.table_container = tk.Frame(card, bg=BG_CARD)
        self.table_container.pack(fill=tk.X)

        self.btn_toggle_table = TactileButton(
            self.table_container,
            text="▼ VIEW WINDOW DIAGNOSTICS LEDGER (0 WINDOWS)",
            command=self._toggle_table,
            bg=BG_SURFACE,
            fg=COLOR_PRIMARY,
            hover_bg="#fbf4ea",
            shadow=BORDER_DARK,
            font=FONT_BODY_SM_BOLD,
            offset=2,
            padx=12,
            pady=6,
        )
        self.btn_toggle_table.pack(anchor="w", pady=(0, 8))

        self.table_frame = tk.Frame(self.table_container, bg=BG_CARD)
        self.table_visible = False

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "CallShield.Treeview.Heading",
            background=COLOR_PRIMARY,
            foreground="#f9ebd7",
            font=FONT_MONO_SM,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "CallShield.Treeview",
            background=BG_CARD,
            fieldbackground=BG_CARD,
            foreground=TEXT_PRIMARY,
            font=FONT_MONO_SM,
            rowheight=24,
        )
        style.map("CallShield.Treeview", background=[("selected", BG_SURFACE)])

        self.tree = ttk.Treeview(
            self.table_frame,
            columns=("win", "threat", "score", "vad"),
            show="headings",
            style="CallShield.Treeview",
            height=6,
        )
        self.tree.heading("win", text="WINDOW TIME")
        self.tree.heading("threat", text="THREAT LEVEL")
        self.tree.heading("score", text="ACOUSTIC MARGIN")
        self.tree.heading("vad", text="VAD SPEECH STATUS")

        self.tree.column("win", width=160, anchor="center")
        self.tree.column("threat", width=140, anchor="center")
        self.tree.column("score", width=140, anchor="center")
        self.tree.column("vad", width=180, anchor="center")

        tree_scroll = tk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)


    def _toggle_table(self):
        if self.table_visible:
            self.table_frame.pack_forget()
            self.table_visible = False
            count = len(self.tree.get_children())
            self.btn_toggle_table.set_text(f"▼ VIEW WINDOW DIAGNOSTICS LEDGER ({count} WINDOWS)")
        else:
            self.table_frame.pack(fill=tk.X, pady=(4, 8))
            self.table_visible = True
            count = len(self.tree.get_children())
            self.btn_toggle_table.set_text(f"▲ HIDE WINDOW DIAGNOSTICS LEDGER ({count} WINDOWS)")

    # --------------------------------------------------------------------------
    # Privacy & Compliance Footer
    # --------------------------------------------------------------------------

    def _build_footer(self):
        footer_frame = tk.Frame(self.app_container, bg=BG_PRIMARY)
        footer_frame.pack(fill=tk.X, pady=(8, 16))

        lbl_footer = tk.Label(
            footer_frame,
            text=f"Call Shield • Zero-Audio-Retention In-Memory Architecture • RFC 3551 Telephony Compliant • PyTorch {DEVICE}",
            font=FONT_EDITORIAL,
            bg=BG_PRIMARY,
            fg=TEXT_MUTED,
        )
        lbl_footer.pack()

    # --------------------------------------------------------------------------
    # Analysis & Data Binding
    # --------------------------------------------------------------------------

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose Audio File for Neural Analysis",
            filetypes=[
                ("All Supported Audio", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.wma"),
                ("M4A Voice Recording (*.m4a)", "*.m4a"),
                ("WAV Audio (*.wav)", "*.wav"),
                ("MP3 Audio (*.mp3)", "*.mp3"),
                ("FLAC Lossless (*.flac)", "*.flac"),
                ("All Files (*.*)", "*.*"),
            ],
        )
        if file_path:
            self.analyze_file(file_path)

    def analyze_file(self, file_path: str):
        p = Path(file_path)
        if not p.is_file():
            messagebox.showerror("File Error", f"File not found: {file_path}")
            return

        self.current_audio_file = str(p)
        self._stop_audio_playback()

        # Update file header display
        self.lbl_file_name.configure(text=p.name)
        self.lbl_file_specs.configure(text="Analyzing acoustic spectral dynamics...")
        self.root.update()

        try:
            res = classify_audio(str(p))
            self.analysis_result = res
            self._render_results(p.name, res)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze audio file:\n{e}")

    def _render_results(self, filename: str, res: ClassificationResult):
        duration = res["duration"]
        win_count = res["window_count"]
        active_wins = res["active_windows"]
        silent_wins = res["silent_windows"]

        self.lbl_file_name.configure(text=filename)
        self.lbl_file_specs.configure(
            text=f"Duration: {duration:.2f}s  •  16 kHz Mono  •  {win_count} Windows ({active_wins} Active Speech, {silent_wins} Silent)"
        )

        status = res["status"]
        is_clone = res["is_clone"]
        mean_score = res["mean_score"]
        min_score = res["min_score"]
        conf = res["confidence"]
        flag_ratio = res["flagged_ratio"]

        # Update Verdict Banner & Level Chip
        if status == "SILENCE":
            self._update_verdict_style(
                badge_text="SILENCE",
                badge_bg=TEXT_MUTED,
                badge_fg="#ffffff",
                banner_bg=BG_SURFACE_INSET,
                border_color=BORDER_SUBTLE,
                shadow_color=BORDER_SUBTLE,
                title="VERDICT: NO AUDIBLE SPEECH (SILENCE / AMBIENT)",
                desc=res["details"],
            )
            self.lbl_level_chip.configure(text="SILENCE", fg=TEXT_MUTED)
            self.level_chip_inner.configure(bg=BG_SURFACE_INSET, highlightbackground=BORDER_SUBTLE)
            self.level_chip_outer.configure(bg=BORDER_SUBTLE)
            self.tile_score.update_value("0.00", fg=TEXT_MUTED)
            self.tile_min.update_value("0.00", fg=TEXT_MUTED)
        elif is_clone:
            self._update_verdict_style(
                badge_text="AI CLONE DETECTED",
                badge_bg=COLOR_ALERT,
                badge_fg="#ffffff",
                banner_bg=COLOR_ALERT_BG,
                border_color=COLOR_ALERT,
                shadow_color=COLOR_ALERT,
                title="VERDICT: SYNTHETIC VOICE CLONE DETECTED",
                desc=f"{res['details']}\nHang up and call back on a trusted known phone number.",
            )
            self.lbl_level_chip.configure(text="ALERT", fg=COLOR_ALERT)
            self.level_chip_inner.configure(bg=COLOR_ALERT_BG, highlightbackground=COLOR_ALERT)
            self.level_chip_outer.configure(bg=COLOR_ALERT)
            self.tile_score.update_value(f"{mean_score:+.2f}", fg=COLOR_ALERT)
            self.tile_min.update_value(f"{min_score:+.2f}", fg=COLOR_ALERT)
        else:
            self._update_verdict_style(
                badge_text="AUTHENTIC",
                badge_bg=COLOR_PRIMARY,
                badge_fg="#f9ebd7",
                banner_bg=COLOR_OK_BG,
                border_color=COLOR_PRIMARY,
                shadow_color=COLOR_PRIMARY,
                title="VERDICT: AUTHENTIC NATURAL SPEECH",
                desc=res["details"],
            )
            self.lbl_level_chip.configure(text="AUTHENTIC", fg=COLOR_PRIMARY)
            self.level_chip_inner.configure(bg=COLOR_OK_BG, highlightbackground=COLOR_PRIMARY)
            self.level_chip_outer.configure(bg=COLOR_PRIMARY)
            self.tile_score.update_value(f"{mean_score:+.2f}", fg=COLOR_PRIMARY)
            self.tile_min.update_value(f"{min_score:+.2f}", fg=COLOR_PRIMARY)

        # Update Telemetry Grid
        flagged_count = int(round((flag_ratio / 100.0) * active_wins))
        self.tile_flagged.update_value(f"{flagged_count} / {active_wins}", subtext=f"{flag_ratio:.0f}% Suspect Ratio")
        self.tile_conf.update_value(f"{conf}%", subtext=f"Operating Threshold: {THRESHOLD:.3f}")

        # Draw Window Progression Track
        self._draw_progression_track(res["window_records"])

        # Populate Diagnostics Table
        self._populate_diagnostics_table(res["window_records"])

    def _update_verdict_style(
        self,
        badge_text: str,
        badge_bg: str,
        badge_fg: str,
        banner_bg: str,
        border_color: str,
        shadow_color: str,
        title: str,
        desc: str,
    ):
        self.verdict_outer.configure(bg=shadow_color)
        self.verdict_inner.configure(bg=banner_bg, highlightbackground=border_color)
        self.verdict_badge.configure(text=badge_text, bg=badge_bg, fg=badge_fg)

        for widget in self.verdict_inner.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=banner_bg)

        self.lbl_verdict_title.configure(text=title, bg=banner_bg, fg=border_color)
        self.lbl_verdict_desc.configure(text=desc, bg=banner_bg)

    def _draw_progression_track(self, records: list[WindowRecord]):
        cv = self.canvas_track
        cv.delete("all")

        if not records:
            return

        w_canvas = cv.winfo_width() or 800
        h_canvas = 90
        pad_x = 16
        pad_y = 12

        n = len(records)
        usable_w = w_canvas - 2 * pad_x

        # Calculate bar geometry
        gap = 4 if n > 15 else 8
        bar_w = max(10, min(36, int((usable_w - (n - 1) * gap) / n)))

        for i, rec in enumerate(records):
            x0 = pad_x + i * (bar_w + gap)
            x1 = x0 + bar_w

            is_silent = bool(rec.get("is_silent", False))
            status = str(rec.get("status", "OK"))
            score = float(rec.get("score", 0.0))

            if is_silent:
                color = BORDER_SUBTLE
                bar_h = 24
            elif status == "ALERT":
                color = COLOR_ALERT
                bar_h = min(56, 32 + int(min(24, abs(score) * 8)))
            elif status == "WARNING":
                color = COLOR_WARN
                bar_h = min(48, 28 + int(min(20, abs(score) * 8)))
            else:
                color = COLOR_PRIMARY
                bar_h = min(56, 28 + int(min(28, max(0.0, score) * 7)))

            y1 = h_canvas - pad_y - 12
            y0 = y1 - bar_h

            # Draw bar
            cv.create_rectangle(x0, y0, x1, y1, fill=color, outline=BORDER_INK, width=1)

            # Draw time label below bar
            t_label = f"{rec.get('time_start', float(i)):.0f}s"
            if n <= 24 or (i % 2 == 0):
                cv.create_text((x0 + x1) // 2, h_canvas - 10, text=t_label, font=FONT_MONO_SM, fill=TEXT_MUTED)

            # Draw score label on top of bar if space permits
            if bar_w >= 24 and not is_silent:
                score_str = f"{score:+.1f}"
                cv.create_text((x0 + x1) // 2, y0 - 7, text=score_str, font=("Consolas", 7, "bold"), fill=TEXT_PRIMARY)

    def _populate_diagnostics_table(self, records: list[WindowRecord]):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for rec in records:
            time_range = str(rec.get("time_range", "--"))
            status = str(rec.get("status", "OK"))
            score = float(rec.get("score", 0.0))
            is_silent = bool(rec.get("is_silent", False))

            if is_silent:
                threat_label = "SILENT"
                score_label = "--"
                vad_label = "Below -42 dBFS Silence"
            elif status == "ALERT":
                threat_label = "🚨 ALERT"
                score_label = f"{score:+.2f}"
                vad_label = "Active Speech (Synthetic)"
            elif status == "WARNING":
                threat_label = "⚠️ WARNING"
                score_label = f"{score:+.2f}"
                vad_label = "Active Speech (Borderline)"
            else:
                threat_label = "🟢 AUTHENTIC"
                score_label = f"{score:+.2f}"
                vad_label = "Active Speech (Natural)"

            self.tree.insert("", "end", values=(time_range, threat_label, score_label, vad_label))

        count = len(records)
        text_prefix = "▲ HIDE" if self.table_visible else "▼ VIEW"
        self.btn_toggle_table.set_text(f"{text_prefix} WINDOW DIAGNOSTICS LEDGER ({count} WINDOWS)")

    # --------------------------------------------------------------------------
    # Audio Playback Engine
    # --------------------------------------------------------------------------

    def _toggle_audio_playback(self):
        if not HAS_WINSOUND or not self.current_audio_file:
            return

        if self.is_playing:
            self._stop_audio_playback()
            return

        try:
            p = Path(self.current_audio_file)
            if p.suffix.lower() == ".wav":
                play_path = str(p)
            else:
                # Convert in-memory float array to temporary wav for playback
                if self.analysis_result and "raw_data" in self.analysis_result:
                    raw_data = self.analysis_result["raw_data"]
                    if self.temp_wav_path is None:
                        self.temp_wav_path = os.path.join(tempfile.gettempdir(), f"callshield_preview_{os.getpid()}.wav")
                    sf.write(self.temp_wav_path, raw_data, 16000)
                    play_path = self.temp_wav_path
                else:
                    play_path = str(p)

            if winsound is not None:
                winsound.PlaySound(play_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
                self.is_playing = True
                self.btn_play.set_text("🔊 Playing...")
                self.btn_play.set_bg(COLOR_ALERT)
        except Exception as e:
            messagebox.showwarning("Audio Error", f"Unable to play audio recording:\n{e}")

    def _stop_audio_playback(self):
        if winsound is not None and self.is_playing:
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.is_playing = False
            self.btn_play.set_text("▶ Play Audio")
            self.btn_play.set_bg(COLOR_PRIMARY)

    # --------------------------------------------------------------------------
    # Quick Sample Presets Loaders
    # --------------------------------------------------------------------------

    def _load_sample_real(self):
        real_cand = BASE_DIR / "datasets" / "new training dataset" / "real_samples" / "1089_134686_000002_000001.wav"
        if not real_cand.is_file():
            real_cand = BASE_DIR / "tests" / "fixtures" / "genuine_like.wav"
        if real_cand.is_file():
            self.analyze_file(str(real_cand))
        else:
            messagebox.showinfo("Preset", "Real speech sample fixture not found.")

    def _load_sample_clone(self):
        clone_cand = BASE_DIR / "datasets" / "new training dataset" / "OpenAI" / "alloy_0.wav"
        if not clone_cand.is_file():
            clone_cand = BASE_DIR / "tests" / "fixtures" / "fake_like.wav"
        if clone_cand.is_file():
            self.analyze_file(str(clone_cand))
        else:
            messagebox.showinfo("Preset", "Synthetic clone sample fixture not found.")

    def _load_sample_silence(self):
        silence_cand = BASE_DIR / "tests" / "fixtures" / "silence.wav"
        if silence_cand.is_file():
            self.analyze_file(str(silence_cand))
        else:
            messagebox.showinfo("Preset", "Silence fixture not found.")


# Alias for backward compatibility
DetectorApp = CallShieldApp


# ==============================================================================
# CLI and Main Entry Points
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI Mode
        fpath = sys.argv[1]
        print(f"[*] Analyzing {fpath} with Call Shield...")
        res = classify_audio(fpath)
        status = res["status"]
        if status == "SILENCE":
            print("Verdict:    [SILENCE] NO AUDIBLE SPEECH (SILENCE / AMBIENT NOISE)")
        elif res["is_clone"]:
            print(f"Verdict:    [ALERT] AI VOICE CLONE DETECTED (Confidence: {res['confidence']}%)")
        else:
            print(f"Verdict:    [OK] AUTHENTIC NATURAL VOICE (Confidence: {res['confidence']}%)")
        print(f"Mean Score: {res['mean_score']:+.2f}")
        print(f"Lowest:     {res['min_score']:+.2f}")
        print(f"Duration:   {res['duration']}s ({res['active_windows']} active speech, {res['silent_windows']} silent)")
        print(f"Details:    {res['details']}")
    else:
        # Native Desktop GUI Mode
        root = tk.Tk()
        app = CallShieldApp(root)
        root.mainloop()
