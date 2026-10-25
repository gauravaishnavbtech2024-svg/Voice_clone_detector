"""Training script for Voice Clone Detector on ASVspoof 2019 dataset.

Trains for 10 epochs (or custom --epochs) using GPU mixed precision (AMP)
and saves the best model checkpoint to models/best.pt.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import ASVspoofLADataset, DirectoryVoiceCloneDataset
from training.model import VoiceCloneDetectorModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Computes Equal Error Rate (EER) and threshold.

    labels: 1.0 for genuine, 0.0 for spoof
    scores: higher score = more likely genuine
    """
    genuine_scores = scores[labels == 1.0]
    spoof_scores = scores[labels == 0.0]

    if len(genuine_scores) == 0 or len(spoof_scores) == 0:
        return 0.5, 0.0

    thresholds = np.sort(np.unique(scores))
    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))

    best_eer = 1.0
    best_thr = 0.0
    min_diff = float("inf")

    for thr in thresholds:
        # False Alarm (Spoof scored as Genuine: score >= thr)
        far = np.mean(spoof_scores >= thr)
        # False Rejection (Genuine scored as Spoof: score < thr)
        frr = np.mean(genuine_scores < thr)

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer = (far + frr) / 2.0
            best_thr = float(thr)

    return float(best_eer), float(best_thr)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = len(loader)

    start_time = time.time()
    for batch_idx, (waveforms, labels) in enumerate(loader):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(waveforms)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{total_batches}] "
                f"- Loss: {avg_loss:.4f} ({elapsed:.1f}s)"
            )

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[float] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(waveforms)
                loss = criterion(logits, labels_dev)
        else:
            logits = model(waveforms)
            loss = criterion(logits, labels_dev)
        total_loss += loss.item()

        all_scores.extend(logits.detach().cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    scores_arr = np.array(all_scores, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.float32)

    # Predictions: score > 0 => genuine (1), score <= 0 => spoof (0)
    preds = (scores_arr > 0.0).astype(np.float32)
    accuracy = float(np.mean(preds == labels_arr)) * 100.0

    eer, best_thr = compute_eer(scores_arr, labels_arr)
    return avg_loss, accuracy, eer * 100.0, best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Call Shield Voice Clone Detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--data-root", type=str, default="datasets/new training dataset", help="Path to training dataset")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save best.pt")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training from")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "eval", "remaining", "all", "all-la-pa"],
        help="Dataset split (for ASVspoof datasets)",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for fast testing")
    parser.add_argument("--max-dev-samples", type=int, default=None, help="Limit dev samples for fast testing")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    data_root = Path(args.data_root)

    # Check if data_root is a directory-based dataset (e.g. 'new training dataset' with real_samples)
    is_directory_dataset = (data_root / "real_samples").is_dir() or any(
        (data_root / sub / "real_samples").is_dir() for sub in ["", "datasets"] if (data_root / sub).is_dir()
    )

    print("\n--- Loading Datasets ---")
    if is_directory_dataset:
        target_dir = data_root if (data_root / "real_samples").is_dir() else (data_root / "datasets" / "new training dataset")
        print(f"Dataset Mode: Directory structure at '{target_dir}'")
        train_dataset, dev_dataset = DirectoryVoiceCloneDataset.create_train_val_split(
            data_dir=target_dir,
            val_ratio=0.15,
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
    else:
        # ASVspoof protocol loader
        if (data_root / "LA" / "LA").is_dir():
            la_root = data_root / "LA" / "LA"
        elif (data_root / "LA" / "ASVspoof2019_LA_cm_protocols").is_dir():
            la_root = data_root / "LA"
        else:
            la_root = data_root

        train_protocol = la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        train_audio_dir = la_root / "ASVspoof2019_LA_train" / "flac"

        eval_protocol = la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
        eval_audio_dir = la_root / "ASVspoof2019_LA_eval" / "flac"

        dev_protocol = la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
        dev_audio_dir = la_root / "ASVspoof2019_LA_dev" / "flac"

        if args.split in ("remaining", "eval"):
            train_pairs = [(eval_protocol, eval_audio_dir)]
        elif args.split == "all":
            train_pairs = [(train_protocol, train_audio_dir), (eval_protocol, eval_audio_dir)]
        else:
            train_pairs = [(train_protocol, train_audio_dir)]

        train_dataset = ASVspoofLADataset(
            protocol_path=train_pairs,
            is_train=True,
            max_samples=args.max_train_samples,
        )
        dev_dataset = ASVspoofLADataset(
            protocol_path=dev_protocol,
            audio_dir=dev_audio_dir,
            is_train=False,
            max_samples=args.max_dev_samples,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Class weighting for BCE loss (bonafide label 1, spoof label 0)
    pos_weight = torch.tensor([train_dataset.class_weights], device=device)
    print(f"Class imbalance pos_weight (spoof / bonafide): {train_dataset.class_weights:.2f}")

    model = VoiceCloneDetectorModel().to(device)
    start_epoch = 1
    best_dev_loss = float("inf")
    best_eer = float("inf")

    # Resume from checkpoint if requested
    if args.resume and Path(args.resume).is_file():
        print(f"\n[*] Loading checkpoint to resume training: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "dev_loss" in checkpoint:
                best_dev_loss = checkpoint["dev_loss"]
            if "dev_eer" in checkpoint:
                best_eer = checkpoint["dev_eer"]
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"[*] Successfully resumed! Previous Best Dev EER: {best_eer:.2f}%, Continuing at Epoch {start_epoch}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best.pt"

    end_epoch = start_epoch + args.epochs - 1
    print(f"\n--- Starting Training ({args.epochs} Epochs: from Epoch {start_epoch} to {end_epoch}) ---")
    start_total = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        print(f"\n>>> Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        dev_loss, dev_acc, dev_eer, dev_thr = evaluate(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
        )
        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.2f}% | "
            f"Dev EER: {dev_eer:.2f}% (Thr: {dev_thr:.3f}) | Time: {epoch_duration:.1f}s"
        )

        # Save checkpoint if dev EER improved (EER is the definitive anti-spoofing metric)
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_dev_loss = dev_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "dev_eer": dev_eer,
                "threshold": dev_thr,
            }
            torch.save(checkpoint, best_model_path)
            print(f"[*] Saved new best checkpoint to {best_model_path} (Dev EER: {dev_eer:.2f}%)")

    total_time = time.time() - start_total
    print(
        f"\n========================================\n"
        f"Training Complete in {total_time/60:.2f} minutes!\n"
        f"Best Checkpoint: {best_model_path}\n"
        f"Best Dev EER: {best_eer:.2f}%\n"
        f"========================================"
    )


if __name__ == "__main__":
    main()
