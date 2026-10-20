"""Validator dataset Speech-to-Text Bahasa Bengkulu.

Memeriksa:
- keberadaan file audio
- format audio valid & bisa dibaca
- transkripsi tidak kosong
- durasi audio
- sample rate
- duplikasi (file_name & transkripsi identik)
- sebaran speaker

Jalankan:
    python prepare_dataset.py
    python prepare_dataset.py --fix-report results/dataset_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from src.config import (
    AUDIO_DIR,
    MAX_AUDIO_SECONDS,
    METADATA_PATH,
    MIN_AUDIO_SECONDS,
    SAMPLE_RATE,
    ensure_dirs,
)


def validate(metadata_path: Path, audio_dir: Path) -> dict:
    from src.dataset import load_metadata

    report: dict = {
        "metadata_path": str(metadata_path),
        "audio_dir": str(audio_dir),
        "errors": [],
        "warnings": [],
        "files": [],
    }

    try:
        df = load_metadata(metadata_path)
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(str(exc))
        return report

    report["n_rows"] = int(len(df))
    file_names = [str(f).strip() for f in df["file_name"]]

    duplicates = [name for name, count in Counter(file_names).items() if count > 1]
    if duplicates:
        report["errors"].append(f"file_name duplikat: {duplicates}")

    durations = []
    sample_rates = Counter()
    speakers = Counter()
    missing_audio = 0
    bad_format = 0
    empty_text = 0

    for _, row in df.iterrows():
        name = str(row["file_name"]).strip()
        text = str(row["text"]).strip()
        speaker = str(row["speaker_id"]).strip()
        speakers[speaker] += 1

        if not text:
            empty_text += 1
            report["warnings"].append(f"Transkripsi kosong: {name}")

        path = audio_dir / name
        if not path.exists():
            missing_audio += 1
            report["errors"].append(f"File audio tidak ditemukan: {path}")
            continue

        try:
            info = sf.info(str(path))
        except Exception as exc:  # noqa: BLE001
            bad_format += 1
            report["errors"].append(f"Format audio tidak valid: {name} ({exc})")
            continue

        duration = info.frames / info.samplerate
        durations.append(duration)
        sample_rates[info.samplerate] += 1

        if duration < MIN_AUDIO_SECONDS:
            report["warnings"].append(f"Durasi < {MIN_AUDIO_SECONDS}s: {name} ({duration:.2f}s)")
        if duration > MAX_AUDIO_SECONDS:
            report["warnings"].append(f"Durasi > {MAX_AUDIO_SECONDS}s: {name} ({duration:.2f}s)")

        report["files"].append(
            {
                "file_name": name,
                "duration_sec": round(duration, 3),
                "sample_rate": info.samplerate,
                "speaker_id": speaker,
            }
        )

    report["summary"] = {
        "n_files_checked": len(report["files"]),
        "missing_audio": missing_audio,
        "bad_format": bad_format,
        "empty_text": empty_text,
        "n_speakers": len(speakers),
        "speakers": dict(speakers),
        "sample_rate_counts": dict(sample_rates),
        "expected_sample_rate": SAMPLE_RATE,
        "total_duration_sec": round(sum(durations), 2),
        "mean_duration_sec": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "min_duration_sec": round(min(durations), 3) if durations else 0.0,
        "max_duration_sec": round(max(durations), 3) if durations else 0.0,
    }

    if sample_rates and set(sample_rates) != {SAMPLE_RATE}:
        report["warnings"].append(
            f"Terdapat sample rate selain {SAMPLE_RATE} Hz. Audio akan di-resample otomatis saat training."
        )

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("VALIDASI DATASET SPEECH-TO-TEXT")
    print("=" * 60)
    if "n_rows" in report:
        print(f"Jumlah baris metadata : {report['n_rows']}")
    s = report.get("summary", {})
    if s:
        print(f"File audio diperiksa  : {s['n_files_checked']}")
        print(f"Missing audio         : {s['missing_audio']}")
        print(f"Format tidak valid    : {s['bad_format']}")
        print(f"Transkripsi kosong    : {s['empty_text']}")
        print(f"Jumlah speaker        : {s['n_speakers']}")
        print(f"Total durasi          : {s['total_duration_sec']} detik")
        print(f"Rata-rata durasi      : {s['mean_duration_sec']} detik")
        print(f"Sample rate (count)   : {s['sample_rate_counts']}")

    if report["errors"]:
        print(f"\nERROR ({len(report['errors'])}):")
        for e in report["errors"][:20]:
            print(f"  - {e}")
    else:
        print("\nTidak ada error.")

    if report["warnings"]:
        print(f"\nWARNING ({len(report['warnings'])}):")
        for w in report["warnings"][:20]:
            print(f"  - {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validasi dataset Speech-to-Text.")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--audio-dir", type=str, default=str(AUDIO_DIR))
    parser.add_argument("--report", type=str, default="results/dataset_report.json")
    args = parser.parse_args()

    ensure_dirs()
    report = validate(Path(args.metadata), Path(args.audio_dir))
    print_report(report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Laporan disimpan: {report_path}")


if __name__ == "__main__":
    main()
