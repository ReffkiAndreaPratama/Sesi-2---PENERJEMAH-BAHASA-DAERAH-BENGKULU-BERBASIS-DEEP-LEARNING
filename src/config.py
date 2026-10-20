"""Konfigurasi global project Speech-to-Text Bahasa Bengkulu."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_PATH = DATASET_DIR / "metadata.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Audio
SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MAX_AUDIO_SECONDS = 10.0
MIN_AUDIO_SECONDS = 0.2

# Reproducibility
SEED = 42

# Vocabulary karakter (huruf kecil + spasi + apostrof).
# CTC memakai indeks 0 sebagai <blank>.
BLANK_IDX = 0
CHARS = list("abcdefghijklmnopqrstuvwxyz '")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 1  # +1 untuk blank

METADATA_COLUMNS = ["file_name", "text", "speaker_id"]


def ensure_dirs() -> None:
    for d in (DATASET_DIR, AUDIO_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
