"""
whisper_vllm_client.py
======================
Quick-test client for OpenAI Whisper Large-v3 served via vLLM.

Sends a single WAV audio file to the vLLM-hosted Whisper model through the
OpenAI-compatible audio transcription endpoint and prints the resulting
transcript.

Usage
-----
    python servers/whisper_vllm_client.py --audio path/to/audio.wav

Or edit ``AUDIO_PATH`` directly in the Configuration section below.

Prerequisites
-------------
- Whisper Large-v3 must be running on SERVER_A via vLLM (port 8001).
  See ``servers/server_startup.sh`` for the launch command.
- ``pip install openai``

Configuration
-------------
Replace ``YOUR_SERVER_A_IP`` with the IP of your Whisper GPU server before
running.

Language
--------
Set ``LANGUAGE = "ar"`` for Arabic audio, ``"en"`` for English.

Project
-------
SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
KFUPM, Term 242
"""

import argparse
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — update SERVER_A_IP to match your vLLM GPU server
# ---------------------------------------------------------------------------
SERVER_A_IP = "YOUR_SERVER_A_IP"       # e.g. "38.x.x.x"
PORT        = 8001
MODEL_NAME  = "openai/whisper-large-v3"
LANGUAGE    = "en"                     # change to "ar" for Arabic audio

# Default test audio path — override via --audio argument or edit directly
AUDIO_PATH  = Path(__file__).parent.parent / "dataset" / "audio" / "test.wav"


def transcribe(audio_path: Path, server_ip: str = SERVER_A_IP) -> str:
    """
    Transcribe a WAV audio file using the vLLM-served Whisper Large-v3 model.

    The function uses the OpenAI Python SDK pointed at the vLLM base URL,
    which exposes an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

    Parameters
    ----------
    audio_path : Path
        Absolute or relative path to the input WAV file.
    server_ip : str, optional
        IP address of the Whisper vLLM GPU server (default: ``SERVER_A_IP``).

    Returns
    -------
    str
        Decoded transcript text.

    Raises
    ------
    FileNotFoundError
        If ``audio_path`` does not exist.
    openai.APIError
        On HTTP-level errors from the vLLM server.

    Notes
    -----
    - ``temperature=0.0`` forces greedy decoding for deterministic transcripts.
    - ``api_key="EMPTY"`` satisfies the OpenAI SDK's key requirement; vLLM
      does not validate the key value.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = OpenAI(
        api_key="EMPTY",                           # vLLM ignores the key value
        base_url=f"http://{server_ip}:{PORT}/v1",
    )

    print(f"[*] Transcribing: {audio_path.name}")
    print(f"[*] Server      : http://{server_ip}:{PORT}")
    print(f"[*] Model       : {MODEL_NAME}\n")

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model=MODEL_NAME,
            language=LANGUAGE,          # ISO 639-1 language code
            response_format="text",     # return plain string (not JSON object)
            temperature=0.0,            # deterministic / greedy decoding
        )

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="Test the vLLM-served Whisper transcription endpoint."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=AUDIO_PATH,
        help="Path to the WAV file to transcribe (default: dataset/audio/test.wav)",
    )
    args = parser.parse_args()

    transcript = transcribe(args.audio)

    print("=== Transcript ===")
    print(transcript)


if __name__ == "__main__":
    main()
