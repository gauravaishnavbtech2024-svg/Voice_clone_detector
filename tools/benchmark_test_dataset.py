"""Benchmark Call Shield Voice Clone Detector against In-The-Wild Test Dataset.

Evaluates models/best.pt on the testing dataset in datasets/TEST/download.
Computes EER, Accuracy, Sensitivity, Specificity, AUC-ROC, and GPU Inference Latency.
"""

from __future__ import annotations

import argparse
import csv
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from callshield.core.audio import crop_or_pad, rms_normalize
from callshield.detectors.torch_detector import TorchDetector


def compute_roc_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Computes EER and Area under ROC curve.

    labels: 1.0 for genuine / bona-fide, 0.0 for spoof / synthetic
    scores: higher score = more likely genuine (> 0 Genuine, < 0 Synthetic)
    """
    genuine_scores = scores[labels == 1.0]
    spoof_scores = scores[labels == 0.0]

    if len(genuine_scores) == 0 or len(spoof_scores) == 0:
        return 0.5, 0.0, 0.5

    thresholds = np.sort(np.unique(scores))
    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))

    best_eer = 1.0
    best_thr = 0.0
    fars: list[float] = []
    tprs: list[float] = []

    for thr in thresholds:
        # False Alarm (Spoof scored as Genuine: score >= thr)
        far = np.mean(spoof_scores >= thr)
        # False Rejection (Genuine scored as Spoof: score < thr)
        frr = np.mean(genuine_scores < thr)
        # True Positive (Genuine scored as Genuine)
        tpr = 1.0 - frr

        fars.append(far)
        tprs.append(tpr)

        diff = abs(far - frr)
        eer_approx = (far + frr) / 2.0
        if diff < 0.01 or eer_approx < best_eer:
            best_eer = eer_approx
            best_thr = float(thr)

    # Approximate AUC
    fars_arr = np.array(fars)
    tprs_arr = np.array(tprs)
    sort_idx = np.argsort(fars_arr)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        auc = float(trapz_fn(tprs_arr[sort_idx], fars_arr[sort_idx]))
    else:
        auc = 0.5

    if auc < 0:
        auc = 1.0 + auc

    return float(best_eer), float(best_thr), max(0.5, min(1.0, auc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Voice Clone Detector on TEST dataset")
    parser.add_argument("--test-zip", type=str, default="datasets/TEST/download", help="Path to test dataset zip")
    parser.add_argument("--model-path", type=str, default="models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of test clips to evaluate (stratified)")
    args = parser.parse_args()

    zip_path = Path(args.test_zip)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Test dataset not found at: {zip_path}")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print("==================================================================")
    print("CALL SHIELD - BENCHMARK REPORT: IN-THE-WILD TEST DATASET")
    print("==================================================================")

    detector = TorchDetector(weights_path=model_path)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {detector.device} ({device_name})")
    print(f"Model Checkpoint: {model_path}")
    print(f"Test Dataset: {zip_path}")

    # Load and parse meta.csv
    print("\n[1/3] Reading metadata from test archive...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        with z.open("release_in_the_wild/meta.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            all_rows = list(reader)

        bonafide_rows = [r for r in all_rows if "bona-fide" in r["label"].lower()]
        spoof_rows = [r for r in all_rows if "spoof" in r["label"].lower()]

        print(f"Total dataset files: {len(all_rows)}")
        print(f"Available Authentic (Bona-fide): {len(bonafide_rows)}")
        print(f"Available AI Clones (Spoof): {len(spoof_rows)}")

        # Stratified sampling
        half = args.num_samples // 2
        rng = np.random.RandomState(42)
        idx_b = rng.choice(len(bonafide_rows), size=min(half, len(bonafide_rows)), replace=False)
        idx_s = rng.choice(len(spoof_rows), size=min(half, len(spoof_rows)), replace=False)

        selected_items = [(bonafide_rows[i]["file"], 1.0) for i in idx_b] + [
            (spoof_rows[j]["file"], 0.0) for j in idx_s
        ]
        rng.shuffle(selected_items)

        print(f"\n[2/3] Evaluating {len(selected_items)} stratified test samples (50% Authentic, 50% AI Clones)...")
        scores: list[float] = []
        labels: list[float] = []
        inference_latencies: list[float] = []

        start_time = time.time()
        for i, (filename, label) in enumerate(selected_items):
            try:
                wav_bytes = z.read(f"release_in_the_wild/{filename}")
                data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # Preprocess: RMS normalize to -26 dBFS and crop/pad to 64,000 samples (4.0s @ 16 kHz)
                normed = rms_normalize(data, target_dbfs=-26.0)
                window = crop_or_pad(normed, n=64000)

                t0 = time.perf_counter()
                score = detector.score(window)
                dt = (time.perf_counter() - t0) * 1000.0  # ms

                scores.append(score)
                labels.append(label)
                inference_latencies.append(dt)

            except Exception as e:
                continue

            if (i + 1) % 100 == 0 or (i + 1) == len(selected_items):
                print(f"  Processed [{i + 1}/{len(selected_items)}] clips... (avg latency: {np.mean(inference_latencies):.2f} ms/window)")

    total_time = time.time() - start_time
    scores_arr = np.array(scores, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32)

    # Compute Metrics
    eer, opt_thr, auc = compute_roc_eer(scores_arr, labels_arr)

    # Standard 0.0 Threshold Predictions (score > 0 => Authentic, score <= 0 => AI Voice)
    preds = (scores_arr > 0.0).astype(np.float32)
    accuracy = float(np.mean(preds == labels_arr)) * 100.0

    # Authentic Voice (label = 1): True Genuine Rate
    authentic_mask = labels_arr == 1.0
    true_authentic_rate = float(np.mean(preds[authentic_mask] == 1.0)) * 100.0
    false_alarm_rate = 100.0 - true_authentic_rate

    # AI Voice Clone (label = 0): AI Detection Rate (Recall on Spoof)
    spoof_mask = labels_arr == 0.0
    ai_detection_rate = float(np.mean(preds[spoof_mask] == 0.0)) * 100.0
    ai_miss_rate = 100.0 - ai_detection_rate

    avg_latency = float(np.mean(inference_latencies))
    p95_latency = float(np.percentile(inference_latencies, 95))

    print("\n==================================================================")
    print("FINAL BENCHMARK PERFORMANCE RESULTS")
    print("==================================================================")
    print(f"Overall Accuracy:                  {accuracy:.2f}%")
    print(f"Equal Error Rate (EER):            {eer * 100.0:.2f}%")
    print(f"AUC-ROC Score:                     {auc:.4f}")
    print(f"AI Voice Clone Detection Rate:     {ai_detection_rate:.2f}% (Recall on AI voices)")
    print(f"AI Voice Miss Rate (False Accept): {ai_miss_rate:.2f}%")
    print(f"Authentic Human Voice Pass Rate:   {true_authentic_rate:.2f}% (Specificity)")
    print(f"False Alarm Rate (False Alert):    {false_alarm_rate:.2f}%")
    print("------------------------------------------------------------------")
    print(f"Inference Latency (Mean):          {avg_latency:.2f} ms per 4-second window")
    print(f"Inference Latency (p95):           {p95_latency:.2f} ms per 4-second window")
    print(f"Throughput:                        {len(selected_items) / total_time:.1f} audio windows / second")
    print("==================================================================")


if __name__ == "__main__":
    main()
