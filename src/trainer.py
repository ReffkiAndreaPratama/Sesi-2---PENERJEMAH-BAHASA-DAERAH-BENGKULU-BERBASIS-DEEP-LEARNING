"""Training loop untuk model Speech-to-Text (CTC).

Fitur: checkpoint terbaik, early stopping, gradient clipping, logging
loss per epoch, dan dukungan CPU/GPU otomatis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import SpeechRecognitionModel


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 30
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    grad_clip: float = 5.0
    patience: int = 6
    seed: int = 42
    device: str = "cpu"
    history: list[dict] = field(default_factory=list)


def resolve_device(requested: str = "auto") -> torch.device:
    """Memilih GPU bila tersedia, jika tidak memakai CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _ctc_lengths(input_lengths: torch.Tensor, reduction: int) -> torch.Tensor:
    return torch.clamp(input_lengths // reduction, min=1)


def _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip: float = 5.0):
    """Menjalankan satu epoch. optimizer=None berarti mode validasi."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0

    torch.set_grad_enabled(training)
    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        log_probs = model(features)
        in_lens = _ctc_lengths(input_lengths, model.time_reduction()).to(device)

        if (target_lengths > in_lens).any():
            continue  # hindari error CTC untuk sampel yang terlalu panjang

        loss = criterion(log_probs, targets, in_lens, target_lengths)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    torch.set_grad_enabled(True)
    return total_loss / max(n_batches, 1)


def fit(
    model: SpeechRecognitionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    checkpoint_path: str | Path,
) -> TrainConfig:
    """Melatih model dan menyimpan checkpoint terbaik."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, config.grad_clip)
        val_loss = _run_epoch(model, val_loader, criterion, device)

        config.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_val - 1e-4
        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{config.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f}{marker}")

        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "hidden_size": config.hidden_size,
                        "num_layers": config.num_layers,
                        "dropout": config.dropout,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping pada epoch {epoch} (best val_loss={best_val:.4f})")
                break

    return config


def load_model_from_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu"):
    """Memuat model dari checkpoint."""
    from src.config import N_MELS, VOCAB_SIZE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan di '{checkpoint_path}'. Jalankan dulu: python train.py"
        )
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("config", {})
    model = SpeechRecognitionModel(
        n_mels=N_MELS,
        vocab_size=VOCAB_SIZE,
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
