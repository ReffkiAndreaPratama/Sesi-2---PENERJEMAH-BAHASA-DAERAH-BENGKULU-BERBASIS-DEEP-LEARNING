"""PERINGATAN PENTING
===================
Script ini membuat dataset NADA SINTETIS (bukan rekaman suara manusia dan
BUKAN Bahasa Bengkulu). Tujuannya HANYA untuk menguji agar seluruh pipeline
(dataset -> training -> evaluasi -> inference -> Streamlit) benar-benar
berjalan.

JANGAN menganggap hasil training/evaluasi dari dataset ini sebagai hasil
penelitian Bahasa Bengkulu.

Setiap nada dipetakan ke satu karakter sehingga model dapat belajar
pemetaan frekuensi -> karakter (uji fungsional end-to-end).

Jalankan:
    python scripts/make_dummy_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import AUDIO_DIR, DATASET_DIR, METADATA_PATH, SAMPLE_RATE  # noqa: E402

DUMMY_CHARS = "abcdef"
SPEAKERS = [
    "dummy_speaker_01",
    "dummy_speaker_02",
    "dummy_speaker_03",
    "dummy_speaker_04",
    "dummy_speaker_05",
    "dummy_speaker_06",
]
REPEATS = 4
DURATION = 1.0
BASE_FREQ = 300.0
FREQ_STEP = 120.0


def synth_tone(freq: float, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # sedikit vibrato agar sinyal lebih kaya
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)
    signal = 0.6 * np.sin(2 * np.pi * freq * vibrato * t)
    signal += 0.2 * np.sin(2 * np.pi * 2 * freq * t)
    noise = rng.normal(0, 0.02, size=t.shape)
    fade = np.ones_like(t)
    n_fade = int(0.02 * sr)
    fade[:n_fade] = np.linspace(0, 1, n_fade)
    fade[-n_fade:] = np.linspace(1, 0, n_fade)
    return np.clip((signal + noise) * fade, -1, 1).astype(np.float32)


def main() -> None:
    print("=" * 64)
    print("MEMBUAT DATASET DUMMY (NADA SINTETIS) - HANYA UNTUK UJI FUNGSI")
    print("BUKAN dataset Bahasa Bengkulu dan BUKAN hasil penelitian.")
    print("=" * 64)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows = []
    counter = 0
    for s_idx, speaker in enumerate(SPEAKERS):
        offset = s_idx * 15.0  # variasi antar "speaker"
        for c_idx, ch in enumerate(DUMMY_CHARS):
            freq = BASE_FREQ + c_idx * FREQ_STEP + offset
            for r in range(REPEATS):
                counter += 1
                fname = f"dummy_{counter:04d}.wav"
                audio = synth_tone(freq, DURATION, SAMPLE_RATE, rng)
                sf.write(AUDIO_DIR / fname, audio, SAMPLE_RATE)
                rows.append({"file_name": fname, "text": ch, "speaker_id": speaker})

    import pandas as pd

    pd.DataFrame(rows).to_csv(METADATA_PATH, index=False)

    print(f"[OK] {len(rows)} file audio dummy dibuat di: {AUDIO_DIR}")
    print(f"[OK] Metadata: {METADATA_PATH}")
    print(f"     Jumlah 'speaker' dummy: {len(SPEAKERS)}, karakter: {list(DUMMY_CHARS)}")
    print("\nIngat: dataset ini hanya alat uji fungsional.")


if __name__ == "__main__":
    main()
