"""Training Speech-to-Text Bahasa Bengkulu.

Contoh:
    python train.py --epochs 30 --batch-size 8 --lr 3e-4
    python train.py --device cuda

Jika dataset Bahasa Bengkulu belum tersedia, script akan berhenti dengan
pesan yang jelas. Untuk uji fungsi (bukan data penelitian) jalankan dulu:
    python scripts/make_dummy_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, RESULTS_DIR, SEED, ensure_dirs
from src.dataset import SpeechDataset, build_splits, collate_fn, load_metadata
from src.evaluate import evaluate_model
from src.model import SpeechRecognitionModel
from src.trainer import TrainConfig, fit, load_model_from_checkpoint, resolve_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Speech-to-Text (CTC).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    df = load_metadata()
    print(f"[INFO] Total sampel metadata: {len(df)} | speaker: {df['speaker_id'].nunique()}")

    train_df, val_df, test_df = build_splits(df, seed=args.seed)
    print(f"[INFO] Split -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    if train_df.empty or val_df.empty:
        raise ValueError("Data training/validasi kosong setelah split. Periksa dataset Anda.")

    train_ds = SpeechDataset(train_df)
    val_ds = SpeechDataset(val_df)
    test_ds = SpeechDataset(test_df)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model = SpeechRecognitionModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    checkpoint_path = MODELS_DIR / "best_model.pt"
    config = fit(model, train_loader, val_loader, config, checkpoint_path)

    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(config.history, f, indent=2)

    best_model, ckpt = load_model_from_checkpoint(checkpoint_path, map_location=str(device))
    print(f"[INFO] Checkpoint terbaik dari epoch {ckpt.get('epoch')} (val_loss={ckpt.get('val_loss'):.4f})")

    metrics = evaluate_model(best_model, test_loader, device=str(device), output_dir=RESULTS_DIR)
    print("\n[EVALUASI TEST SET]")
    print(f"    WER = {metrics['wer']:.4f}")
    print(f"    CER = {metrics['cer']:.4f}")
    print(f"    Jumlah sampel: {metrics['n_samples']}")

    summary = {
        "device": str(device),
        "epochs_run": len(config.history),
        "best_checkpoint_epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("val_loss"),
        "test": {"wer": metrics["wer"], "cer": metrics["cer"], "n_samples": metrics["n_samples"]},
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "hyperparameters": vars(args),
        "note": (
            "Jika model ini dilatih dengan scripts/make_dummy_dataset.py, "
            "hasilnya BUKAN evaluasi Bahasa Bengkulu dan tidak boleh dilaporkan "
            "sebagai hasil penelitian."
        ),
    }
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Checkpoint: {checkpoint_path}")
    print(f"[OK] Hasil evaluasi: {RESULTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
