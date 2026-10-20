"""Preprocessing audio untuk Speech-to-Text.

Langkah:
1. Load audio (soundfile) -> torch tensor.
2. Konversi ke mono.
3. Resample ke SAMPLE_RATE bila perlu.
4. Normalisasi amplitudo (peak / RMS).
5. Trim silence (opsional, berbasis energi).
6. Hitung log-Mel spectrogram sebagai representasi fitur.

Setiap langkah dijelaskan agar mudah dipertanggungjawabkan secara akademik.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import (
    HOP_LENGTH,
    MAX_AUDIO_SECONDS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)

_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
_DB_TRANSFORM = torchaudio.transforms.AmplitudeToDB(stype="power")


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    """Memuat audio sebagai tensor mono float32 pada sample rate target.

    Decoding memakai `soundfile` (bukan decoder bawaan torchaudio yang
    memerlukan torchcodec) agar mudah dijalankan di Windows/laptop.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # (channels, frames)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Normalisasi peak ke rentang kira-kira [-1, 1]."""
    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / peak
    return waveform


def trim_silence(
    waveform: torch.Tensor, threshold_ratio: float = 0.05, frame: int = 400
) -> torch.Tensor:
    """Memangkas bagian hening di awal/akhir berbasis energi RMS per frame."""
    if waveform.numel() < frame * 2:
        return waveform
    x = waveform.squeeze(0)
    n_frames = x.numel() // frame
    frames = x[: n_frames * frame].reshape(n_frames, frame)
    energy = frames.pow(2).mean(dim=1).sqrt()
    if energy.max() <= 1e-8:
        return waveform
    mask = energy > (threshold_ratio * energy.max())
    if not mask.any():
        return waveform
    start = int(mask.nonzero(as_tuple=True)[0][0].item()) * frame
    end = int((mask.nonzero(as_tuple=True)[0][-1].item() + 1) * frame)
    return waveform[:, start:end]


def pad_or_truncate(
    waveform: torch.Tensor, max_seconds: float = MAX_AUDIO_SECONDS, sr: int = SAMPLE_RATE
) -> torch.Tensor:
    """Menyeragamkan panjang audio (padding zero atau potong)."""
    max_len = int(max_seconds * sr)
    length = waveform.size(1)
    if length > max_len:
        waveform = waveform[:, :max_len]
    elif length < max_len:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - length))
    return waveform


def waveform_to_logmel(waveform: torch.Tensor) -> torch.Tensor:
    """Mengubah waveform (1, T) menjadi log-Mel spectrogram (n_mels, T')."""
    mel = _MEL_TRANSFORM(waveform)  # (1, n_mels, T')
    logmel = _DB_TRANSFORM(mel)
    return logmel.squeeze(0)  # (n_mels, T')


def preprocess_file(path: str | Path, apply_trim: bool = True) -> torch.Tensor:
    """Pipeline lengkap untuk satu file audio -> log-Mel (n_mels, T')."""
    waveform, _ = load_audio(path)
    if apply_trim:
        waveform = trim_silence(waveform)
    waveform = normalize_waveform(waveform)
    waveform = pad_or_truncate(waveform)
    return waveform_to_logmel(waveform)


def audio_duration_seconds(path: str | Path) -> float:
    """Durasi audio dalam detik tanpa memuat seluruh sinyal ke memori."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def find_audio_issues(
    waveform: torch.Tensor, sr: int, min_seconds: float
) -> list[str]:
    """Mengembalikan daftar masalah pada sebuah waveform."""
    issues: list[str] = []
    duration = waveform.size(1) / sr
    if duration < min_seconds:
        issues.append(f"durasi terlalu pendek ({duration:.2f}s)")
    if torch.isnan(waveform).any():
        issues.append("mengandung NaN")
    if waveform.abs().max() < 1e-6:
        issues.append("audio hening / amplitude nol")
    return issues


__all__ = [
    "load_audio",
    "normalize_waveform",
    "trim_silence",
    "pad_or_truncate",
    "waveform_to_logmel",
    "preprocess_file",
    "audio_duration_seconds",
    "find_audio_issues",
]
