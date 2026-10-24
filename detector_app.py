"""Call Shield - Native Local Desktop Voice Clone Detector.

Runs locally as a native desktop GUI window (NOT in the browser).
Uses the trained PyTorch model (models/best.pt) on GPU/CPU for identification.

Usage:
  # Launch the native desktop GUI window:
  python detector_app.py

  # Or analyze an audio file directly from terminal:
  python detector_app.py path/to/audio.wav
"""

from __future__ import annotations

import io
import math
import sys
import threading
from pathlib import Path

# Add src and base directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch

from callshield.core.audio import crop_or_pad, is_silent, rms_normalize
from training.model import VoiceCloneDetectorModel

# -----------------------------------------------------------------------------
# PyTorch Model Initialization
# -----------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "best.pt"
if not MODEL_PATH.is_file():
    print(f"Error: Model checkpoint not found at: {MODEL_PATH}")
    sys.exit(1)

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
# Audio Decoding & Window-by-Window Detection
# -----------------------------------------------------------------------------
def decode_audio_file(file_path_or_bytes: str | Path | bytes) -> tuple[np.ndarray, float]:
    """Decodes audio into 16 kHz mono float32 and returns (samples, duration_s)."""
    if isinstance(file_path_or_bytes, (str, Path)):
        data, sr = sf.read(str(file_path_or_bytes), dtype="float32")
    else:
        data, sr = sf.read(io.BytesIO(file_path_or_bytes), dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr), 16000)
        up = 16000 // g
        down = int(sr) // g
        data = resample_poly(data, up=up, down=down).astype(np.float32)

    duration_s = round(len(data) / 16000.0, 2)
    return data.astype(np.float32), duration_s


def analyze_audio(samples_16k: np.ndarray) -> dict[str, object]:
    """Evaluates audio window-by-window using the trained PyTorch model."""
    total_samples = len(samples_16k)
    duration_s = round(total_samples / 16000.0, 2)

    window_samples = 64000  # 4.0s
    hop_samples = 16000  # 1.0s
    min_samples = 32000  # 2.0s

    timeline: list[dict[str, object]] = []

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

    if alert_count > 0:
        overall_level = "ALERT"
        verdict_title = "VERDICT: SYNTHETIC CLONE DETECTED"
        verdict_desc = f"Strong vocal clone signatures detected starting at {peak_time:.1f}s. High probability of AI synthesis."
    elif warning_count >= 2:
        overall_level = "WARNING"
        verdict_title = "VERDICT: ELEVATED ANOMALY SUSPICION"
        verdict_desc = "Mild acoustic irregularities detected in pitch and spectral dynamics. Exercise caution."
    else:
        overall_level = "OK"
        verdict_title = "VERDICT: AUTHENTIC NATURAL SPEECH"
        verdict_desc = "Vocal harmonics, breath pauses, and formant structures match authentic human speech."

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


# -----------------------------------------------------------------------------
# Native Local Desktop GUI (Tkinter)
# -----------------------------------------------------------------------------
def run_desktop_gui() -> None:
    """Launches the native desktop GUI window with Call Shield styling."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    # Color palette
    BG_LINEN = "#f9ebd7"
    BG_CARD = "#f4dfc4"
    TEXT_MAIN = "#1a1a1a"
    TEXT_MUTED = "#5c554e"
    BORDER_COLOR = "#2c2c2c"
    COLOR_OK = "#046a38"
    COLOR_WARN = "#c25e00"
    COLOR_ALERT = "#e34222"

    root = tk.Tk()
    root.title("Call Shield - Local Voice Clone Detector")
    root.geometry("880x780")
    root.minsize(780, 680)
    root.configure(bg=BG_LINEN)

    # Style ttk widgets
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background="#ffffff", foreground=TEXT_MAIN, rowheight=26, fieldbackground="#ffffff")
    style.configure("Treeview.Heading", background="#f0dfcb", foreground=TEXT_MAIN, font=("Segoe UI", 10, "bold"))
    style.map("Treeview", background=[("selected", "#dcd0c0")], foreground=[("selected", TEXT_MAIN)])

    # Top Header
    header_frame = tk.Frame(root, bg=BG_LINEN, pady=10)
    header_frame.pack(fill=tk.X, padx=20)

    title_label = tk.Label(
        header_frame,
        text="CALL SHIELD // NATIVE DETECTOR",
        font=("Georgia", 18, "bold"),
        fg=TEXT_MAIN,
        bg=BG_LINEN,
    )
    title_label.pack(anchor=tk.W)

    status_str = f"Model: models/best.pt ({DEVICE})  |  Calibrated Threshold: {MODEL_THRESHOLD:.3f}"
    sub_label = tk.Label(header_frame, text=status_str, font=("Consolas", 10), fg=TEXT_MUTED, bg=BG_LINEN)
    sub_label.pack(anchor=tk.W)

    # Separator
    sep1 = tk.Frame(root, height=2, bg=BORDER_COLOR)
    sep1.pack(fill=tk.X, padx=20, pady=5)

    # Preset & File Control Row
    ctrl_frame = tk.Frame(root, bg=BG_LINEN, pady=8)
    ctrl_frame.pack(fill=tk.X, padx=20)

    btn_file = tk.Button(
        ctrl_frame,
        text="Select Audio File (.wav, .mp3, .flac)...",
        font=("Segoe UI", 10, "bold"),
        bg="#ffffff",
        fg=TEXT_MAIN,
        activebackground=BORDER_COLOR,
        activeforeground="#ffffff",
        relief=tk.SOLID,
        bd=2,
        padx=12,
        pady=6,
        cursor="hand2",
    )
    btn_file.pack(side=tk.LEFT, padx=(0, 10))

    # Preset 1: Authentic
    btn_p1 = tk.Button(
        ctrl_frame,
        text="Sample 1: Authentic",
        font=("Segoe UI", 9, "bold"),
        bg="#ffffff",
        fg=COLOR_OK,
        relief=tk.SOLID,
        bd=1,
        padx=10,
        pady=6,
        cursor="hand2",
    )
    btn_p1.pack(side=tk.LEFT, padx=4)

    # Preset 2: Mixed
    btn_p2 = tk.Button(
        ctrl_frame,
        text="Sample 2: Mixed Call",
        font=("Segoe UI", 9, "bold"),
        bg="#ffffff",
        fg=COLOR_WARN,
        relief=tk.SOLID,
        bd=1,
        padx=10,
        pady=6,
        cursor="hand2",
    )
    btn_p2.pack(side=tk.LEFT, padx=4)

    # Preset 3: Clone
    btn_p3 = tk.Button(
        ctrl_frame,
        text="Sample 3: AI Clone",
        font=("Segoe UI", 9, "bold"),
        bg="#ffffff",
        fg=COLOR_ALERT,
        relief=tk.SOLID,
        bd=1,
        padx=10,
        pady=6,
        cursor="hand2",
    )
    btn_p3.pack(side=tk.LEFT, padx=4)

    # File info label
    file_info_var = tk.StringVar(value="No audio file loaded. Click a sample button or browse an audio file.")
    lbl_file_info = tk.Label(root, textvariable=file_info_var, font=("Segoe UI", 9, "italic"), fg=TEXT_MUTED, bg=BG_LINEN)
    lbl_file_info.pack(anchor=tk.W, padx=20, pady=(2, 8))

    # Verdict Card Banner
    verdict_frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.SOLID, padx=16, pady=14)
    verdict_frame.pack(fill=tk.X, padx=20, pady=8)

    badge_var = tk.StringVar(value="IDLE")
    lbl_badge = tk.Label(
        verdict_frame,
        textvariable=badge_var,
        font=("Consolas", 10, "bold"),
        fg="#ffffff",
        bg=BORDER_COLOR,
        padx=8,
        pady=2,
    )
    lbl_badge.pack(anchor=tk.W)

    verdict_title_var = tk.StringVar(value="READY FOR INSPECTION")
    lbl_vtitle = tk.Label(
        verdict_frame,
        textvariable=verdict_title_var,
        font=("Georgia", 14, "bold"),
        fg=TEXT_MAIN,
        bg="#ffffff",
        pady=4,
    )
    lbl_vtitle.pack(anchor=tk.W)

    verdict_desc_var = tk.StringVar(value="Select an audio recording to run GPU acoustic classification.")
    lbl_vdesc = tk.Label(
        verdict_frame,
        textvariable=verdict_desc_var,
        font=("Georgia", 10),
        fg=TEXT_MUTED,
        bg="#ffffff",
        wraplength=800,
        justify=tk.LEFT,
    )
    lbl_vdesc.pack(anchor=tk.W)

    # Telemetry Grid (4 Cards)
    telemetry_frame = tk.Frame(root, bg=BG_LINEN)
    telemetry_frame.pack(fill=tk.X, padx=20, pady=8)

    # Helper function to create telemetry card
    def make_stat_card(parent, label_text):
        f = tk.Frame(parent, bg="#ffffff", bd=2, relief=tk.SOLID, padx=12, pady=10)
        lbl = tk.Label(f, text=label_text, font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg="#ffffff")
        lbl.pack(anchor=tk.W)
        val_var = tk.StringVar(value="--")
        val = tk.Label(f, textvariable=val_var, font=("Consolas", 16, "bold"), fg=TEXT_MAIN, bg="#ffffff")
        val.pack(anchor=tk.W, pady=(4, 0))
        return f, val_var, val

    f1, var_mean, lbl_mean = make_stat_card(telemetry_frame, "MEAN CLASSIFIER SCORE")
    f1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

    f2, var_min, lbl_min = make_stat_card(telemetry_frame, "LOWEST WINDOW SCORE")
    f2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

    f3, var_alerts, lbl_alerts = make_stat_card(telemetry_frame, "ALERT WINDOWS")
    f3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

    f4, var_peak, lbl_peak = make_stat_card(telemetry_frame, "FIRST ANOMALY TIME")
    f4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    # Diagnostics Table
    table_label = tk.Label(
        root,
        text="WINDOW-BY-WINDOW DIAGNOSTIC TIMELINE",
        font=("Segoe UI", 9, "bold"),
        fg=TEXT_MAIN,
        bg=BG_LINEN,
    )
    table_label.pack(anchor=tk.W, padx=20, pady=(12, 4))

    table_frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.SOLID)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    columns = ("t", "level", "score", "flag")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
    tree.heading("t", text="Window Horizon (s)")
    tree.heading("level", text="Threat Level")
    tree.heading("score", text="Acoustic Score")
    tree.heading("flag", text="Classification Result")

    tree.column("t", width=160, anchor=tk.CENTER)
    tree.column("level", width=160, anchor=tk.CENTER)
    tree.column("score", width=180, anchor=tk.CENTER)
    tree.column("flag", width=260, anchor=tk.W)

    tree.tag_configure("OK", foreground=COLOR_OK)
    tree.tag_configure("WARNING", foreground=COLOR_WARN)
    tree.tag_configure("ALERT", foreground=COLOR_ALERT)

    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Core Action to Process Audio File
    def process_file_path(path_to_audio: str | Path, display_name: str | None = None) -> None:
        p = Path(path_to_audio)
        if not p.is_file():
            file_info_var.set(f"File not found: {p}")
            return

        name = display_name or p.name
        file_info_var.set(f"Analyzing '{name}' on {DEVICE}...")
        root.update()

        try:
            samples_16k, dur = decode_audio_file(p)
            res = analyze_audio(samples_16k)

            # Update verdict card
            badge_var.set(res["overall_level"])
            if res["overall_level"] == "OK":
                lbl_badge.config(bg=COLOR_OK)
                verdict_frame.config(bg="#f4faee")
            elif res["overall_level"] == "WARNING":
                lbl_badge.config(bg=COLOR_WARN)
                verdict_frame.config(bg="#fffaf2")
            else:
                lbl_badge.config(bg=COLOR_ALERT)
                verdict_frame.config(bg="#fef2f0")

            verdict_title_var.set(str(res["verdict_title"]))
            verdict_desc_var.set(str(res["verdict_desc"]))
            file_info_var.set(f"Inspected '{name}' ({dur:.1f}s)  -  Sample Rate: 16 kHz")

            # Update telemetry cards
            avg_s = float(res["avg_score"])
            var_mean.set(f"{'+' if avg_s > 0 else ''}{avg_s:.2f}")
            lbl_mean.config(fg=COLOR_OK if avg_s > 0 else COLOR_ALERT)

            min_s = float(res["min_score"])
            var_min.set(f"{'+' if min_s > 0 else ''}{min_s:.2f}")
            lbl_min.config(fg=COLOR_OK if min_s > 0 else COLOR_ALERT)

            var_alerts.set(f"{res['alert_count']} Windows")
            var_peak.set(f"{res['peak_time']:.1f}s" if res["peak_time"] is not None else "None")

            # Clear and populate treeview
            for item in tree.get_children():
                tree.delete(item)

            for pt in res["timeline"]:
                score_str = f"{'+' if pt['score'] > 0 else ''}{pt['score']:.2f}"
                tree.insert(
                    "",
                    tk.END,
                    values=(f"{pt['t']:.1f}s", pt["level"], score_str, pt["flag"]),
                    tags=(pt["level"],),
                )

        except Exception as e:
            file_info_var.set(f"Error processing audio: {e}")

    # Connect buttons
    def on_choose_file():
        file_path = filedialog.askopenfilename(
            title="Select Audio Recording",
            filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.m4a;*.ogg"), ("All Files", "*.*")],
        )
        if file_path:
            process_file_path(file_path)

    btn_file.config(command=on_choose_file)
    btn_p1.config(command=lambda: process_file_path(BASE_DIR / "client" / "samples" / "sample_authentic.wav", "sample_authentic.wav"))
    btn_p2.config(command=lambda: process_file_path(BASE_DIR / "client" / "samples" / "sample_mixed.wav", "sample_mixed.wav"))
    btn_p3.config(command=lambda: process_file_path(BASE_DIR / "client" / "samples" / "sample_clone.wav", "sample_clone.wav"))

    print("[*] Local desktop window opened. Ready.")
    root.mainloop()


# -----------------------------------------------------------------------------
# CLI Evaluation Mode
# -----------------------------------------------------------------------------
def run_cli_mode(audio_file_path: str | Path) -> None:
    """Evaluates an audio file directly from the command line."""
    p = Path(audio_file_path)
    if not p.is_file():
        print(f"Error: File not found: {p}")
        sys.exit(1)

    print("\n========================================================")
    print(f"CALL SHIELD: EVALUATING {p.name}")
    print("========================================================")
    samples_16k, dur = decode_audio_file(p)
    res = analyze_audio(samples_16k)

    print(f"\nDuration:              {dur}s")
    print(f"Mean Classifier Score: {res['avg_score']:+.2f}")
    print(f"Lowest Window Score:   {res['min_score']:+.2f}")
    print(f"Alert Windows Count:   {res['alert_count']}")
    print(f"First Anomaly Time:    {res['peak_time']}")
    print(f"\n>>> {res['verdict_title']} <<<")
    print(f"Details: {res['verdict_desc']}")

    print("\nWindow-by-Window Diagnostic Timeline:")
    print(f"{'Time':<8} | {'Score':<8} | {'Level':<10} | {'Flag'}")
    print("-" * 45)
    for pt in res["timeline"]:
        print(f"{pt['t']:<6.1f}s | {pt['score']:<+8.2f} | {pt['level']:<10} | {pt['flag']}")
    print("========================================================\n")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--gui"):
        target_path = sys.argv[1]
        if target_path.startswith("--file="):
            target_path = target_path.split("=", 1)[1]
        elif target_path == "--file" and len(sys.argv) > 2:
            target_path = sys.argv[2]
        run_cli_mode(target_path)
    else:
        run_desktop_gui()
