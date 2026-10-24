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


def classify_audio(file_path: str) -> dict[str, object]:
    """Reads audio file, slices into 4s windows, and runs model."""
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16 kHz if necessary
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 16000)
        data = resample_poly(data, up=16000 // g, down=int(sr) // g).astype(np.float32)

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
            text="Select Audio File (.wav, .mp3, .flac)",
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
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All Files", "*.*"),
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
