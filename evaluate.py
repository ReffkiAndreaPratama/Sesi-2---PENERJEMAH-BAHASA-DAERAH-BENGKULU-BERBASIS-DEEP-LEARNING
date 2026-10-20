"""Evaluasi model Speech-to-Text Bahasa Bengkulu (WER & CER).

Contoh:
    python evaluate.py --checkpoint models/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import RESULTS_DIR, ensure_dirs
from src.dataset import SpeechDataset, build_splits, collate_fn, load_metadata
from src.evaluate import evaluate_model
from src.trainer import load_model_from_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi Speech-to-Text (WER/CER).")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    device = resolve_device(args.device)

    df = load_metadata()
    _, _, test_df = build_splits(df, seed=args.seed)
    test_ds = SpeechDataset(test_df)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model, ckpt = load_model_from_checkpoint(args.checkpoint, map_location=str(device))
    result = evaluate_model(model, test_loader, device=str(device), output_dir=RESULTS_DIR)

    print("=" * 55)
    print("EVALUASI SPEECH-TO-TEXT")
    print("=" * 55)
    print(f"Checkpoint     : {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"Jumlah sampel  : {result['n_samples']}")
    print(f"WER            : {result['wer']:.4f} ({result['wer']*100:.2f}%)")
    print(f"CER            : {result['cer']:.4f} ({result['cer']*100:.2f}%)")
    print("-" * 55)
    print("Contoh prediksi:")
    for s in result["samples"][:10]:
        print(f"  REF: {s['reference']!r}")
        print(f"  HYP: {s['hypothesis']!r}")
    print(f"\n[OK] Detail hasil: {RESULTS_DIR / 'evaluation.json'}")

    if (RESULTS_DIR / "training_summary.json").exists():
        with open(RESULTS_DIR / "training_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"[INFO] {summary.get('note', '')}")


if __name__ == "__main__":
    main()
