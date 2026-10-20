"""Arsitektur model Speech-to-Text (CTC) untuk Bahasa Bengkulu.

Model: CNN (ekstraksi fitur spektral) + BiLSTM (konteks temporal) + CTC.
Arsitektur ini cukup ringan untuk dijalankan pada CPU saat pengembangan,
dan dapat dipindahkan ke GPU/HPC untuk training skala besar.

Catatan:
- Input  : log-Mel spectrogram (batch, n_mels, T)
- Output : log-probabilitas per frame (T', batch, vocab) untuk CTCLoss
- Downsampling waktu total = 4x (dua MaxPool2d dengan stride 2 di sumbu waktu)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import N_MELS, VOCAB_SIZE


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SpeechRecognitionModel(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels

        self.conv = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 64, pool=(2, 1)),
        )

        freq_after = n_mels // 8
        lstm_input = 64 * freq_after

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    @staticmethod
    def time_reduction() -> int:
        """Faktor kompresi sumbu waktu (untuk menyesuaikan input_lengths CTC)."""
        return 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, T) -> log_probs: (T', batch, vocab)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, n_mels, T)

        x = self.conv(x)  # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')
        x, _ = self.lstm(x)
        x = self.dropout(x)
        logits = self.fc(x)  # (B, T', vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # (T', B, vocab)


def greedy_ctc_decode(log_probs: torch.Tensor) -> list[str]:
    """Greedy decoding untuk CTC.

    log_probs: (T', B, vocab) -> list teks (panjang B).
    Aturan: ambil argmax per frame, gabungkan yang berulang, buang blank (0).
    """
    from src.config import BLANK_IDX, IDX_TO_CHAR

    indices = log_probs.argmax(dim=-1)  # (T', B)
    indices = indices.transpose(0, 1)  # (B, T')
    results: list[str] = []
    for seq in indices:
        chars: list[str] = []
        prev = -1
        for idx in seq.tolist():
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


__all__ = ["SpeechRecognitionModel", "greedy_ctc_decode"]
