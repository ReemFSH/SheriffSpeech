"""
whisper_local.py
================
Batch-transcribe a directory of WAV files using Whisper Large-v3 *locally*
via HuggingFace Transformers (no vLLM server required).

Results are written to a CSV file with two columns:
  file_name  |  transcript

Use this script to:
  - Transcribe audio files directly on a GPU machine without a running server.
  - Pre-generate transcripts for offline analysis or debugging.
  - Validate Whisper accuracy on the dataset before pipeline integration.

In production the pipeline uses the vLLM-served version
(see ``servers/whisper_vllm_client.py`` and ``pipeline/pipeline.py``).

Usage
-----
    python servers/whisper_local.py --audio-dir /path/to/wav/files
    python servers/whisper_local.py --audio-dir /path/to/wav/files --output transcripts.csv

Prerequisites
-------------
- A CUDA-capable GPU (CPU inference is supported but very slow for large-v3).
- ``pip install transformers torch``
- The model will be auto-downloaded from HuggingFace Hub on first run
  (~3 GB for float16 weights).

Project
-------
SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
KFUPM, Term 242
"""

import argparse
import csv
import warnings
from pathlib import Path

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Suppress noisy transformer / user warnings for cleaner output
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
MODEL_NAME  = "openai/whisper-large-v3"
LANGUAGE    = "en"               # change to "ar" for Arabic audio
CHUNK_LEN_S = 30                 # audio chunk length in seconds for long-form ASR
BATCH_SIZE  = 8                  # number of chunks processed in parallel on GPU

# Default audio directory — relative to the repo root
DEFAULT_AUDIO_DIR = Path(__file__).parent.parent / "dataset" / "audio"
DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "results" / "whisper_transcripts.csv"


def build_asr_pipeline(model_name: str = MODEL_NAME) -> pipeline:
    """
    Create and return a HuggingFace ASR pipeline backed by Whisper Large-v3.

    The pipeline automatically uses GPU (device 0) if CUDA is available,
    otherwise falls back to CPU (device -1).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Defaults to ``"openai/whisper-large-v3"``.

    Returns
    -------
    transformers.Pipeline
        Configured automatic-speech-recognition pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    device_label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"[*] Loading {model_name} on {device_label}…")

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    print("[*] Model loaded.\n")
    return asr


def transcribe_directory(
    audio_dir: Path,
    asr: pipeline,
    output_csv: Path,
    language: str = LANGUAGE,
) -> None:
    """
    Transcribe all WAV files in ``audio_dir`` and save results to a CSV file.

    Long audio files are handled via chunked inference (``chunk_length_s``),
    and chunks are batched for GPU efficiency (``batch_size``).

    Parameters
    ----------
    audio_dir : Path
        Directory (searched recursively) containing ``*.wav`` files.
    asr : transformers.Pipeline
        Loaded Whisper ASR pipeline.
    output_csv : Path
        Destination CSV file. Created (or overwritten) by this function.
    language : str, optional
        ISO 639-1 language code passed to Whisper (e.g. ``"en"`` or ``"ar"``).

    Returns
    -------
    None
        Results are written to ``output_csv``.
    """
    # Collect all WAV files recursively
    files = sorted(audio_dir.rglob("*.wav"))
    if not files:
        print(f"[!] No WAV files found in: {audio_dir}")
        return

    print(f"[*] Found {len(files)} WAV file(s) in {audio_dir}")
    print(f"[*] Output CSV: {output_csv}\n")

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcript"])   # CSV header

        for i, audio_file in enumerate(files, start=1):
            try:
                print(f"[{i}/{len(files)}] Transcribing: {audio_file.name}")

                result = asr(
                    str(audio_file),
                    chunk_length_s=CHUNK_LEN_S,        # split long audio into 30-s chunks
                    batch_size=BATCH_SIZE,              # process multiple chunks in parallel
                    generate_kwargs={"language": language},
                )

                # The pipeline returns {"text": "…"} for ASR tasks
                transcript = result["text"].strip()
                writer.writerow([audio_file.name, transcript])
                print(f"    → {transcript[:80]}{'…' if len(transcript) > 80 else ''}")

            except Exception as exc:
                # Log the error but continue with remaining files
                print(f"    [ERROR] {audio_file.name}: {exc}")
                writer.writerow([audio_file.name, f"ERROR: {exc}"])

    print(f"\n[✓] Transcription complete. Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-transcribe WAV files using Whisper Large-v3 locally."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory of WAV files to transcribe (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=LANGUAGE,
        help='ISO 639-1 language code: "en" for English, "ar" for Arabic (default: en)',
    )
    args = parser.parse_args()

    asr = build_asr_pipeline()
    transcribe_directory(
        audio_dir=args.audio_dir,
        asr=asr,
        output_csv=args.output,
        language=args.language,
    )


if __name__ == "__main__":
    main()
