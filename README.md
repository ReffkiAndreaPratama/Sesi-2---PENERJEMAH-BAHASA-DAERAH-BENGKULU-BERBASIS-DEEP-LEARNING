# Speech-to-Text Bahasa Daerah Bengkulu Menggunakan Deep Learning

Prototype **Speech-to-Text (ASR)** untuk bahasa daerah Bengkulu, dibuat untuk tugas
**Informatics Practitioner Lecture Series** dengan materi
**"Bridging Global Infrastructure to Address Local Challenge with HPC"**
(Pemateri: Asst. Prof. Dr. Worawan Diaz Carballo — Thammasat University).

> **Status penting.** Dataset Bahasa Bengkulu **belum tersedia**. Project ini
> menyediakan **pipeline lengkap yang siap menerima dataset** (validasi → preprocessing →
> training → evaluasi → inference → aplikasi). Model **belum** diklaim mampu mengenali
> Bahasa Bengkulu sampai dilatih dengan dataset Bahasa Bengkulu yang nyata.

---

## 1. Latar Belakang

Bahasa daerah minoritas sering tertinggal dalam teknologi AI karena minimnya data.
Materi pemateri menunjukkan bagaimana infrastruktur komputasi global (HPC) dan AI
dapat dipakai untuk menyelesaikan permasalahan lokal. Salah satunya adalah
membangun **speech-to-text** untuk bahasa daerah seperti bahasa Bengkulu.

## 2. Kaitan dengan Materi Pemateri

| Materi | Implementasi di Project |
|---|---|
| High Performance Computing (HPC) | Model & training dirancang agar bisa dijalankan di CPU (pengembangan) maupun GPU/HPC untuk skala besar (`resolve_device`, batch, checkpoint) |
| AI untuk permasalahan lokal | Studi kasus ASR bahasa daerah Bengkulu |
| Peran "Integrator" | Menggabungkan audio processing, deep learning (PyTorch), evaluasi (jiwer), dan aplikasi web (Streamlit) menjadi satu sistem utuh |
| Speech-to-Text / Text-to-Speech | Modul ASR berbasis CNN + BiLSTM + CTC |
| Data dalam jumlah besar & computational scale | Arsitektur & pipeline dirancang untuk dataset besar; speaker-independent split, checkpoint, early stopping |
| Linux / SSH / parallel / GPU | Dijalankan lewat CLI Python yang portabel (Linux/Windows), `torch.cuda` bila tersedia |

## 3. Permasalahan

Bagaimana membangun pipa (*pipeline*) ASR yang **benar** (bisa dijalankan, dievaluasi,
dan dipertanggungjawabkan) untuk bahasa daerah Bengkulu walaupun dataset pengucapannya
belum terkumpul?

## 4. Tujuan

1. Menyediakan format dataset & **validator** yang jelas.
2. Menyediakan **preprocessing audio** (mono, resample, normalisasi, trim, log-Mel).
3. Menyediakan **arsitektur deep learning** ASR (CTC) yang realistis.
4. Menyediakan pipeline **training** (checkpoint, early stopping, CPU/GPU).
5. Menyediakan **evaluasi** dengan WER & CER.
6. Menyediakan **aplikasi Streamlit** untuk transkripsi audio.

## 5. Dataset

Format yang diharapkan:

```
dataset/
├── audio/
│   ├── audio_001.wav
│   └── ...
└── metadata.csv
```

`metadata.csv`:

```csv
file_name,text,speaker_id
audio_001.wav,contoh transkripsi,speaker_01
audio_002.wav,contoh lain,speaker_02
```

Aturan:
- `file_name` = nama file di `dataset/audio/`.
- `text` = transkripsi (huruf kecil, tanpa tanda baca kompleks).
- `speaker_id` = penanda penutur (untuk **evaluasi speaker-independent**).

**Cara menyiapkan dataset nyata:**
1. Rekam kalimat per file (mis. format WAV, mono, 16 kHz lebih disukai).
- Waktu file: 0,2–10 detik.
- Simpan ke `dataset/audio/`.
- Catat `file_name`, `text`, `speaker_id` di `metadata.csv`.
- Jalankan `python prepare_dataset.py` untuk validasi.

### Dataset dummy (HANYA untuk uji fungsi)

Karena dataset Bengkulu belum ada, tersedia generator **nada sintetis**:

```bash
python scripts/make_dummy_dataset.py
```

Script ini membuat 144 file nada + metadata (6 "speaker" dummy). Dataset ini
**bukan** rekaman manusia dan **bukan** Bahasa Bengkulu. Tujuannya hanya memastikan
seluruh pipeline berjalan. **Jangan** memakai hasilnya sebagai hasil penelitian.

## 6. Preprocessing Audio

Setiap langkah dijelaskan di `src/preprocessing.py`:

1. **Load** audio dengan `soundfile` (menghindari ketergantungan decoder eksternal).
2. **Mono** — rata-rata kanal bila stereo.
3. **Resample** ke 16 kHz.
4. **Normalisasi** peak.
5. **Trim silence** berbasis energi (opsional).
6. **Pad/truncate** ke panjang maksimum 10 detik.
7. **Log-Mel spectrogram** (n_mels=80, n_fft=400, hop=160) sebagai fitur input.

## 7. Arsitektur Model

`src/model.py` — CNN + BiLSTM + CTC:

```
log-Mel (80 x T)
  -> Conv2d block x3 (+BatchNorm, GELU, MaxPool)   # kompresi waktu 4x
  -> BiLSTM (2 layer, hidden 256)
  -> Linear (vocab)
  -> log_softmax  -> CTC loss / greedy decoding
```

- Vocabulary karakter: `a-z`, spasi, apostrof. Indeks 0 = `<blank>` untuk CTC.
- Decoding: **greedy CTC** (collapsing repeated + membuang blank).

## 8. Training

```bash
python train.py --epochs 60 --batch-size 8 --lr 1e-3
python train.py --device cuda        # bila tersedia GPU
```

Parameter: `--epochs`, `--batch-size`, `--lr`, `--hidden-size`, `--num-layers`,
`--dropout`, `--patience`, `--seed`, `--device`.

Fitur training:
- split **train/val/test**; speaker-independent bila jumlah speaker ≥ 4,
- checkpoint terbaik disimpan ke `models/best_model.pt`,
- early stopping berdasarkan validation loss,
- gradient clipping, reproducible (`--seed`).

## 9. Evaluasi

```bash
python evaluate.py --checkpoint models/best_model.pt
```

Metrik:
- **WER (Word Error Rate)** — berbasis kata (pemisah spasi).
- **CER (Character Error Rate)** — berbasis karakter.

Keduanya dihitung dengan library `jiwer` dari prediksi model yang benar-benar dijalankan.

## 10. Aplikasi Streamlit

```bash
streamlit run app.py
```

Halaman:
- **Dataset Overview** — jumlah audio, speaker, total & rata-rata durasi, distribusi.
- **Training** — panduan CLI, kurva loss, tombol training demo (dataset kecil).
- **Evaluation** — WER, CER, contoh prediksi.
- **Speech Recognition** — upload audio, putar, transcribe, confidence, unduh TXT.

## 11. Struktur Project

```
speech-to-text-bahasa-bengkulu/
├── app.py
├── train.py
├── evaluate.py
├── prepare_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   ├── audio/
│   └── metadata.csv
├── models/best_model.pt
├── results/
│   ├── dataset_report.json
│   ├── evaluation.json
│   ├── training_history.json
│   └── training_summary.json
├── scripts/
│   └── make_dummy_dataset.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocessing.py
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    └── evaluate.py
```

## 12. Instalasi

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux  : source .venv/bin/activate

# PyTorch CPU (ringan, untuk laptop):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Selebihnya:
pip install -r requirements.txt
```

Untuk GPU, ikuti perintah resmi PyTorch (index CUDA yang sesuai), lalu
`pip install -r requirements.txt`.

## 13. Alur Menjalankan dari Nol

```bash
# (0) Opsional: buat dataset dummy untuk uji fungsi
python scripts/make_dummy_dataset.py

# (1) Validasi dataset
python prepare_dataset.py

# (2) Training
python train.py --epochs 60 --lr 1e-3

# (3) Evaluasi
python evaluate.py

# (4) Aplikasi
streamlit run app.py
```

## 14. Hasil Aktual (DATASET DUMMY — BUKAN HASIL PENELITIAN)

Pada uji fungsional dengan **dataset nada sintetis** (`make_dummy_dataset.py`,
6 speaker dummy, 144 klip, split speaker-independent 96/24/24), model berhasil
belajar memetakan nada → karakter:

| Metrik | Nilai |
|---|---|
| WER | 0.00 |
| CER | 0.00 |

**Peringatan:** angka di atas berasal dari dataset **nada sintetis**, **bukan** Bahasa
Bengkulu, dan **bukan** hasil penelitian. Tujuannya hanya membuktikan bahwa pipeline
end-to-end berjalan. Hasil Bahasa Bengkulu baru dapat dilaporkan **setelah** dataset
rekaman asli dikumpulkan dan model dilatih ulang.

## 15. Limitasi

1. Dataset Bahasa Bengkulu belum tersedia → model belum valid untuk bahasa tersebut.
2. Confidenсe pada UI hanya heuristik (bukan probabilitas terkalibrasi).
3. Decoding masih greedy (belum beam search / language model).
4. Vocabulary masih karakter ASCII dasar (belum menangani diakritik).
5. Arsitektur belum dioptimalkan untuk dataset besar.

## 16. Pengembangan Selanjutnya

- Mengumpulkan & memvalidasi dataset rekaman Bahasa Bengkulu (kolaborasi penutur asli).
- Augmentasi data (speed/pitch perturbation, SpecAugment).
- Beam search + language model (kenLM) untuk decoding.
- Transfer learning dari model multilingual (mis. wav2vec2/XLS-R).
- Pelatihan skala besar di GPU/HPC, mixed precision, distributed training.
- Text-to-Speech untuk bahasa yang sama.

## 17. Upload ke GitHub

```bash
git init
git add .
git commit -m "Initial speech-to-text implementation"
git branch -M main
git remote add origin [GITHUB_REPOSITORY]
git push -u origin main
```

> File audio (`dataset/audio/*.wav`), `metadata.csv`, dan checkpoint `models/*.pt`
> di-ignore (lihat `.gitignore`). README ini menjelaskan cara menyiapkan ulang dataset
> dan menjalankan training.
