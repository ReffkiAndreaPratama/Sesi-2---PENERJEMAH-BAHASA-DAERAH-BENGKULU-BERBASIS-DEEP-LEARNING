"""Aplikasi Streamlit: Speech-to-Text Bahasa Bengkulu.

Jalankan:
    streamlit run app.py

Catatan: aplikasi ini hanyalah antarmuka pipeline. Klaim kemampuan Bahasa
Bengkulu hanya valid jika model dilatih dengan dataset Bahasa Bengkulu nyata.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, RESULTS_DIR
from src.config import AUDIO_DIR, RESULTS_DIR
from src.dataset import SpeechDataset, build_splits, collate_fn, load_metadata
from src.evaluate import transcribe_with_confidence
from src.preprocessing import audio_duration_seconds
from src.trainer import TrainConfig, fit, load_model_from_checkpoint, resolve_device

CHECKPOINT_PATH = "models/best_model.pt"

DUMMY_NOTICE = (
    "CATATAN: Jika model dilatih memakai `scripts/make_dummy_dataset.py`, maka "
    "dataset tersebut adalah **nada sintetis**, bukan Bahasa Bengkulu. Hasil "
    "transkripsi/WER/CER dari dataset dummy **bukan** hasil penelitian."
)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str):
    try:
        model, ckpt = load_model_from_checkpoint(checkpoint_path)
        return model, ckpt
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def page_dataset_overview() -> None:
    st.header("Dataset Overview")
    try:
        df = load_metadata()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.info("Jalankan `python scripts/make_dummy_dataset.py` untuk uji fungsi (dummy).")
        return

    durations = []
    for name in df["file_name"]:
        try:
            durations.append(audio_duration_seconds(AUDIO_DIR / name))
        except Exception:
            durations.append(None)
    df = df.assign(duration_sec=durations)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Audio", f"{len(df)}")
    c2.metric("Jumlah Speaker", f"{df['speaker_id'].nunique()}")
    total = float(df["duration_sec"].fillna(0).sum())
    c3.metric("Total Durasi", f"{total:.1f} s")
    c4.metric("Rata-rata Durasi", f"{df['duration_sec'].fillna(0).mean():.2f} s")

    st.subheader("Distribusi per Speaker")
    st.bar_chart(df["speaker_id"].value_counts())

    st.subheader("Distribusi per Transkripsi (top 20)")
    st.bar_chart(df["text"].value_counts().head(20))

    st.subheader("Tabel Metadata")
    st.dataframe(df.head(100), width="stretch")
    st.warning(DUMMY_NOTICE)


def page_training() -> None:
    st.header("Training")
    st.markdown(
        "Training dijalankan lewat CLI:\n\n"
        "```bash\npython train.py --epochs 30 --batch-size 8 --lr 3e-4\n```"
    )

    history = load_json(str(Path("results") / "training_history.json"))
    if history:
        st.subheader("Kurva Loss")
        hist_df = pd.DataFrame(history).set_index("epoch")
        st.line_chart(hist_df)

    st.subheader("Training Cepat (demo, di dalam aplikasi)")
    st.caption("Berguna untuk menguji pipeline pada dataset kecil. Model terbaik akan disimpan.")
    epochs = st.number_input("Epoch", min_value=1, max_value=50, value=5, step=1)
    if st.button("Mulai training demo"):
        try:
            with st.spinner("Melatih model..."):
                df = load_metadata()
                train_df, val_df, _ = build_splits(df)
                train_loader = _make_loader(train_df, shuffle=True)
                val_loader = _make_loader(val_df, shuffle=False)
                from src.model import SpeechRecognitionModel

                model = SpeechRecognitionModel()
                config = TrainConfig(epochs=int(epochs), device="auto")
                fit(model, train_loader, val_loader, config, CHECKPOINT_PATH)
            st.success("Training selesai. Muat ulang halaman Speech Recognition untuk memakai model baru.")
            st.cache_resource.clear()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training gagal: {exc}")


def _make_loader(df, shuffle: bool):
    from torch.utils.data import DataLoader

    ds = SpeechDataset(df)
    return DataLoader(ds, batch_size=8, shuffle=shuffle, collate_fn=collate_fn)


def page_evaluation() -> None:
    st.header("Evaluation")
    result = load_json(str(Path("results") / "evaluation.json"))
    if not result:
        st.warning("Belum ada hasil evaluasi. Jalankan: `python evaluate.py` atau `python train.py`")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("WER", f"{result.get('wer', 0) * 100:.2f}%")
    c2.metric("CER", f"{result.get('cer', 0) * 100:.2f}%")
    c3.metric("Jumlah Sampel", result.get("n_samples", 0))

    st.subheader("Contoh Prediksi")
    samples = result.get("samples", [])
    if samples:
        st.dataframe(samples, width="stretch")
    st.warning(DUMMY_NOTICE)


def page_speech_recognition(model, checkpoint) -> None:
    st.header("Speech Recognition")
    if model is None:
        st.error("Model belum tersedia. Jalankan `python train.py` terlebih dahulu.")
        return

    device = resolve_device("auto")
    st.caption(f"Model: {CHECKPOINT_PATH} | device: {device}")

    audio_file = st.file_uploader("Upload Audio (wav/flac/ogg)", type=["wav", "flac", "ogg", "mp3"])
    if audio_file is None:
        st.info("Silakan upload file audio untuk ditranskripsi.")
        return

    suffix = Path(audio_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path)

    if st.button("TRANSCRIBE"):
        try:
            with st.spinner("Mentranskripsi audio..."):
                result = transcribe_with_confidence(model, tmp_path, device=str(device))
            st.subheader("Hasil Transkripsi")
            st.success(result["text"] or "(kosong)")
            st.metric("Confidence (heuristik)", f"{result['confidence'] * 100:.2f}%")
            st.download_button(
                "Download TXT",
                data=(result["text"] or "").encode("utf-8"),
                file_name="transkripsi.txt",
                mime="text/plain",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal mentranskripsi: {exc}")

    st.warning(DUMMY_NOTICE)


def main() -> None:
    st.set_page_config(page_title="Speech-to-Text Bahasa Bengkulu", page_icon="🎙️", layout="wide")
    st.title("Speech-to-Text Bahasa Bengkulu")
    st.caption("Prototype pipeline Deep Learning (CNN + BiLSTM + CTC)")

    model, checkpoint = load_model(CHECKPOINT_PATH)

    pages = ["Dataset Overview", "Training", "Evaluation", "Speech Recognition"]
    page = st.sidebar.radio("Navigasi", pages)
    st.sidebar.markdown("---")
    st.sidebar.caption("Siapkan dataset: `python prepare_dataset.py`")
    st.sidebar.caption("Training: `python train.py`")

    if page == "Dataset Overview":
        page_dataset_overview()
    elif page == "Training":
        page_training()
    elif page == "Evaluation":
        page_evaluation()
    else:
        page_speech_recognition(model, checkpoint)


if __name__ == "__main__":
    main()
