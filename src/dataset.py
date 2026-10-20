"""Dataset & split untuk Speech-to-Text Bahasa Bengkulu.

Format metadata (dataset/metadata.csv):

    file_name,text,speaker_id
    audio_001.wav,contoh transkripsi,speaker_01

Prinsip:
- Data dari speaker yang sama TIDAK dicampur antara train dan test
  (speaker-independent split) bila jumlah speaker mencukupi.
- Tidak ada dataset suara palsu yang dianggap data penelitian.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from src.config import AUDIO_DIR, METADATA_PATH, SEED


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Memuat dan memvalidasi metadata.csv."""
    path = Path(path) if path else METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"metadata.csv tidak ditemukan di '{path}'.\n"
            "Siapkan dataset Anda: dataset/audio/*.wav + dataset/metadata.csv "
            "(kolom: file_name,text,speaker_id).\n"
            "Untuk uji fungsi saja (BUKAN dataset penelitian) jalankan: "
            "python scripts/make_dummy_dataset.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in ["file_name", "text", "speaker_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom metadata tidak lengkap: {missing}")
    df = df.dropna(subset=["file_name", "text"]).reset_index(drop=True)
    df["file_name"] = df["file_name"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip().str.lower()
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    return df


def encode_text(text: str) -> list[int]:
    """Mengubah teks menjadi daftar indeks karakter (abaikan karakter tak dikenal)."""
    from src.config import CHAR_TO_IDX

    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def decode_indices(indices) -> str:
    """Mengubah indeks karakter menjadi teks (tanpa blank)."""
    from src.config import IDX_TO_CHAR

    return "".join(IDX_TO_CHAR.get(int(i), "") for i in indices if int(i) != 0)


class SpeechDataset(Dataset):
    """Dataset PyTorch untuk satu split (train/val/test)."""

    def __init__(self, df: pd.DataFrame, audio_dir: Path | None = None):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir) if audio_dir else AUDIO_DIR

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        from src.preprocessing import preprocess_file

        row = self.df.iloc[idx]
        audio_path = self.audio_dir / row["file_name"]
        features = preprocess_file(audio_path)  # (n_mels, T)
        target = torch.tensor(encode_text(row["text"]), dtype=torch.long)
        return {
            "features": features.float(),
            "target": target,
            "text": str(row["text"]),
            "file_name": str(row["file_name"]),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Menyusun batch CTC: padding time + input_lengths + target_lengths."""
    feats = [b["features"] for b in batch]
    n_mels = feats[0].size(0)
    max_t = max(f.size(1) for f in feats)

    padded = torch.zeros(len(batch), n_mels, max_t, dtype=torch.float32)
    input_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, f in enumerate(feats):
        t = f.size(1)
        padded[i, :, :t] = f
        input_lengths[i] = t

    targets = [b["target"] for b in batch]
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)

    return {
        "features": padded,
        "input_lengths": input_lengths,
        "targets": targets_cat,
        "target_lengths": target_lengths,
        "texts": [b["text"] for b in batch],
        "file_names": [b["file_name"] for b in batch],
    }


def build_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membagi data menjadi train/val/test secara speaker-independent.

    Split speaker-independent 3-arah membutuhkan cukup banyak speaker.
    Jika jumlah speaker < 4 (atau split gagal), dipakai pembagian acak
    berbasis baris disertai peringatan.
    """
    df = df.reset_index(drop=True)
    n_speakers = df["speaker_id"].nunique()

    if n_speakers >= 4:
        try:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            trainval_idx, test_idx = next(gss_test.split(df, groups=df["speaker_id"]))
            trainval = df.iloc[trainval_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            val_frac = val_ratio / (train_ratio + val_ratio)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            train_idx, val_idx = next(gss_val.split(trainval, groups=trainval["speaker_id"]))
            train_df = trainval.iloc[train_idx].reset_index(drop=True)
            val_df = trainval.iloc[val_idx].reset_index(drop=True)
            return train_df, val_df, test_df
        except ValueError:
            pass

    print(
        "[WARN] Split speaker-independent tidak dapat dilakukan (speaker < 4). "
        "Memakai split acak per baris; evaluasi belum speaker-independent."
    )
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_df = shuffled.iloc[:n_test]
    val_df = shuffled.iloc[n_test : n_test + n_val]
    train_df = shuffled.iloc[n_test + n_val :]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
