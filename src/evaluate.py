"""Evaluasi Speech-to-Text: Word Error Rate (WER) & Character Error Rate (CER).

Metrik dihitung dengan library `jiwer` dari hasil prediksi model yang
benar-benar dijalankan. Tidak ada angka yang dikarang.
"""

from __future__ import annotations

import json
from pathlib import Path

import jiwer
import torch

from src.model import greedy_ctc_decode


def transcribe_loader(model, loader, device="cpu") -> tuple[list[str], list[str]]:
    """Menghasilkan (references, hypotheses) untuk seluruh DataLoader."""
    model = model.to(device)
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            log_probs = model(features)
            preds = greedy_ctc_decode(log_probs)
            for ref, hyp in zip(batch["texts"], preds):
                references.append(ref if ref.strip() else " ")
                hypotheses.append(hyp if hyp.strip() else " ")
    return references, hypotheses


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> dict:
    """Menghitung WER dan CER agregat."""
    wer = float(jiwer.wer(references, hypotheses))
    cer = float(jiwer.cer(references, hypotheses))
    return {"wer": wer, "cer": cer, "n_samples": len(references)}


def evaluate_model(model, loader, device="cpu", output_dir: str | Path | None = None,
                   max_samples: int = 20) -> dict:
    """Evaluasi penuh + simpan hasil ke results/evaluation.json."""
    references, hypotheses = transcribe_loader(model, loader, device)
    metrics = compute_wer_cer(references, hypotheses)

    samples = [
        {"reference": r, "hypothesis": h}
        for r, h in list(zip(references, hypotheses))[:max_samples]
    ]
    result = {**metrics, "samples": samples}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def transcribe_file(model, audio_path: str | Path, device="cpu") -> str:
    """Transkripsi satu file audio."""
    from src.preprocessing import preprocess_file

    model = model.to(device)
    model.eval()
    features = preprocess_file(audio_path).float().unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(features)
    return greedy_ctc_decode(log_probs)[0]


def transcribe_with_confidence(model, audio_path: str | Path, device="cpu") -> dict:
    """Transkripsi satu file + estimasi confidence sederhana.

    Confidence dihitung sebagai rata-rata probabilitas frame yang dipilih
    (bukan frame blank). Ini adalah heuristik, bukan probabilitas terkalibrasi.
    """
    from src.config import BLANK_IDX
    from src.preprocessing import preprocess_file

    model = model.to(device)
    model.eval()
    features = preprocess_file(audio_path).float().unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(features)  # (T', 1, vocab)

    probs = log_probs.exp().squeeze(1)  # (T', vocab)
    best_prob, best_idx = probs.max(dim=-1)
    mask = best_idx != BLANK_IDX
    if mask.any():
        confidence = float(best_prob[mask].mean().item())
    else:
        confidence = float(best_prob.mean().item())

    text = greedy_ctc_decode(log_probs)[0]
    return {"text": text, "confidence": confidence}

