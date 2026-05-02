"""
app.py  —  SheriffSpeech Demo Server
======================================
A lightweight Flask web application that exposes the full SEC619 three-stage
threat-detection pipeline as HTTP endpoints, backed by a browser-based UI.

This is an exact replication of the ``SystemA_vs_SystemB_FusionEnabled``
notebook logic, packaged for interactive demonstration.

Architecture
------------
The server calls three remote GPU services in sequence:

  ┌────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
  │  Browser   │────▶│  Flask (this server) │────▶│ Stage 1: Whisper ASR │
  │ (UI / drop │     │    localhost:5000     │     │ vLLM · SERVER_A:8001 │
  │  audio)    │     └──────────┬───────────┘     └──────────┬───────────┘
  └────────────┘                │                            │ transcript
                                │                 ┌──────────▼───────────┐
                                │                 │ Stage 2: SpeechBrain │
                                │                 │ SER · SERVER_A:9100  │
                                │                 └──────────┬───────────┘
                                │                            │ emotion tone
                                │                 ┌──────────▼───────────┐
                                │                 │  Fusion Layer        │
                                │                 │  (deterministic)     │
                                │                 └──────────┬───────────┘
                                │                            │ enriched prompt
                                │                 ┌──────────▼───────────┐
                                │                 │ Stage 3: Qwen3Guard  │
                                │                 │ vLLM · SERVER_B:8000 │
                                │◀────────────────│ Safety verdict       │
                                │                 └──────────────────────┘
                    JSON response to browser

Endpoints
---------
  GET  /          Serve the browser UI (static/index.html)
  GET  /health    Check connectivity to all three GPU servers
  POST /process   Run the full pipeline on an uploaded audio file

Pipeline Modes
--------------
  System A (text-only)  : FUSION_ENABLED=False  — Whisper → Qwen3Guard
  System B (full fusion): FUSION_ENABLED=True   — Whisper → SpeechBrain
                          → Fusion Layer → Qwen3Guard

Run
---
    pip install flask requests openai
    python demo/app.py
    # Open http://localhost:5000

Configuration
-------------
Set the server IPs in the block below before running.

Project
-------
SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
KFUPM, Term 242
"""

import re
import json
import time
import tempfile
import traceback
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# ─── Server configuration ─────────────────────────────────────────────────────
# Replace with your actual GPU server IPs before running.
WHISPER_SERVER_IP = "YOUR_SERVER_A_IP"    # GPU Server A — runs Whisper + SpeechBrain
WHISPER_PORT      = 8001                   # Whisper vLLM port
TONE_API_URL      = f"http://{WHISPER_SERVER_IP}:9100/v1/audio/tone"   # SpeechBrain SER
TONE_API_KEY      = ""                     # leave blank unless your SER server requires a key
QWENGUARD_URL     = "http://YOUR_SERVER_B_IP:8000/v1/chat/completions"  # Qwen3Guard vLLM
QWENGUARD_MODEL   = "Qwen/Qwen3Guard-Gen-8B"

# ─── Pipeline settings ────────────────────────────────────────────────────────
LANGUAGE         = "en"       # audio language passed to Whisper ("ar" for Arabic)
TEMPERATURE      = 0.0        # greedy decoding for reproducible transcripts
MAX_TOKENS_GUARD = 128        # Qwen3Guard verdict is always short

# STRICT_MODE: promotes "Controversial" verdicts to "Unsafe" for security contexts
# Set to False to preserve the original three-class output (Safe / Controversial / Unsafe)
STRICT_MODE = True

# ─── Retry / timeout settings ─────────────────────────────────────────────────
WHISPER_TIMEOUT_S = 300   # long audio may take up to 5 min
TONE_TIMEOUT_S    = 180
GUARD_TIMEOUT_S   = 180
RETRIES           = 3     # number of attempts before raising an error
RETRY_SLEEP_S     = 5.0   # seconds between retries

# ─── Supported audio MIME types ───────────────────────────────────────────────
MIME_MAP = {
    ".wav":  "audio/wav",
    ".mp3":  "audio/mpeg",
    ".m4a":  "audio/mp4",
    ".flac": "audio/flac",
    ".ogg":  "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".aac":  "audio/aac",
}

# ─── Flask app ────────────────────────────────────────────────────────────────
# static_folder points to demo/static/ which contains index.html
app = Flask(__name__, static_folder="static")


# ══════════════════════════════════════════════════════════════════════════════
# Guard output parsing
# ══════════════════════════════════════════════════════════════════════════════

# Qwen3Guard outputs a short structured verdict:
#   Safety: <Safe|Unsafe|Controversial>
#   Categories: <comma-separated list>
SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATS_PATTERN = re.compile(r"Categories:\s*(.*)", re.IGNORECASE | re.DOTALL)


def parse_guard(content: str) -> tuple[str | None, list[str]]:
    """
    Parse the raw text verdict produced by Qwen3Guard.

    Parameters
    ----------
    content : str
        Raw model output, e.g. ``"Safety: Unsafe\\nCategories: Violent"``.

    Returns
    -------
    safety : str or None
        One of ``"Safe"``, ``"Unsafe"``, or ``"Controversial"``.
    categories : list[str]
        Deduplicated list of harm category strings (empty for safe content).
    """
    safety, categories = None, []

    m = SAFE_PATTERN.search(content or "")
    if m:
        safety = m.group(1).capitalize()

    m2 = CATS_PATTERN.search(content or "")
    if m2:
        raw = (m2.group(1) or "").strip()
        if raw:
            # Handle both comma-separated and newline-separated category lists
            categories = (
                [p.strip() for p in raw.split(",")]
                if "," in raw
                else [ln.strip() for ln in raw.splitlines() if ln.strip()]
            )
        # A single "None" category means the content is safe
        if len(categories) == 1 and categories[0].lower() == "none":
            categories = []

    # Deduplicate while preserving order
    seen, dedup = set(), []
    for c in categories:
        if c not in seen:
            seen.add(c)
            dedup.append(c)

    return safety, dedup


def normalize_safety_label(safety: str | None, strict_mode: bool = True) -> str | None:
    """
    Apply STRICT_MODE escalation: treat ``"Controversial"`` as ``"Unsafe"``.

    In security contexts any controversial content should be flagged rather
    than passed through, so this mapping eliminates false negatives.

    Parameters
    ----------
    safety : str or None
        Raw verdict from the model.
    strict_mode : bool
        If True, escalate "Controversial" → "Unsafe".

    Returns
    -------
    str or None
        Final normalized verdict.
    """
    if strict_mode and safety == "Controversial":
        return "Unsafe"
    return safety


# ══════════════════════════════════════════════════════════════════════════════
# Retry wrapper
# ══════════════════════════════════════════════════════════════════════════════

def retry_call(fn, name: str = "call"):
    """
    Execute ``fn()`` up to ``RETRIES`` times, sleeping between attempts.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to retry.
    name : str
        Label used in the error message if all attempts fail.

    Returns
    -------
    Any
        Return value of the first successful call to ``fn()``.

    Raises
    ------
    RuntimeError
        If all attempts raise exceptions.
    """
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"{name} failed after {RETRIES} attempts: {repr(last)}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Whisper ASR
# ══════════════════════════════════════════════════════════════════════════════

def whisper_transcribe(audio_path: Path) -> str:
    """
    Transcribe an audio file using the vLLM-served Whisper Large-v3 model.

    Parameters
    ----------
    audio_path : Path
        Path to the temporary audio file saved from the upload.

    Returns
    -------
    str
        Plain-text transcript, stripped of surrounding whitespace.
    """
    ext      = audio_path.suffix.lower()
    mimetype = MIME_MAP.get(ext, "audio/wav")

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1",
    )

    with open(audio_path, "rb") as f:
        out = client.audio.transcriptions.create(
            file=(audio_path.name, f, mimetype),
            model="openai/whisper-large-v3",
            language=LANGUAGE,
            response_format="text",
            temperature=TEMPERATURE,
        )
    return str(out).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — SpeechBrain SER
# ══════════════════════════════════════════════════════════════════════════════

def tone_from_api(audio_path: Path) -> dict:
    """
    Classify the speaker's emotion using the SpeechBrain REST API.

    The API returns a JSON object with:
      - ``label_full``   : dominant emotion (e.g. ``"Angry"``)
      - ``label_short``  : abbreviated label
      - ``top_p``        : probability of the dominant emotion
      - ``top3``         : list of top-3 {label_full, label_short, p} dicts

    Parameters
    ----------
    audio_path : Path
        Path to the audio file to classify.

    Returns
    -------
    dict
        Parsed JSON response from the SpeechBrain API.

    Raises
    ------
    requests.HTTPError
        On non-2xx HTTP responses.
    """
    headers = {"X-API-Key": TONE_API_KEY} if TONE_API_KEY else {}

    with open(audio_path, "rb") as f:
        r = requests.post(
            TONE_API_URL,
            headers=headers,
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=TONE_TIMEOUT_S,
        )
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# Fusion layer
# ══════════════════════════════════════════════════════════════════════════════

def clean_transcript_text(transcript: str) -> str:
    """
    Normalise the transcript string returned by Whisper.

    Whisper via vLLM occasionally returns a JSON-wrapped string instead of
    plain text. This helper unwraps it if needed.

    Parameters
    ----------
    transcript : str
        Raw transcript, either plain or JSON-encoded.

    Returns
    -------
    str
        Clean transcript text.
    """
    t = (transcript or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "text" in obj:
            return str(obj["text"]).strip()
    except Exception:
        pass
    return t


def build_fused_input(transcript: str, tone: dict) -> str:
    """
    Combine the transcript and emotion tone into an enriched prompt for Qwen3Guard.

    System B prepends a natural-language emotion context sentence to the
    transcript before sending it to the Guard model. This allows Qwen3Guard
    to weigh intent signals from the speaker's tone alongside the text content.

    Example output (System B)::

        [Audio context: The speaker sounds very angry (ang=0.92, neu=0.05, hap=0.02).]

        Transcript: I'm going to find out where you live.

    For System A (text-only, ``tone={}``), only the clean transcript is returned.

    Parameters
    ----------
    transcript : str
        Raw transcript from Whisper.
    tone : dict
        Tone API response. Pass ``{}`` to disable fusion (System A mode).

    Returns
    -------
    str
        Prompt string to be sent to Qwen3Guard.
    """
    clean_text = clean_transcript_text(transcript)

    if not tone:
        return clean_text  # System A: text-only

    label_full = tone.get("label_full", "")
    top_p      = tone.get("top_p", None)
    top3       = tone.get("top3", []) or []

    # Map confidence score to adverb for the context sentence
    if isinstance(top_p, (int, float)):
        conf = "very" if top_p >= 0.80 else ("noticeably" if top_p >= 0.55 else "somewhat")
    else:
        conf = "somewhat"

    if label_full:
        parts = [
            f"{i.get('label_full', i.get('label_short', ''))}={i.get('p', 0):.2f}"
            for i in top3[:3]
        ]
        emotion_sentence = f"The speaker sounds {conf} {label_full.lower()} ({', '.join(parts)})."
    else:
        emotion_sentence = "Speaker tone could not be determined."

    return f"[Audio context: {emotion_sentence}]\n\nTranscript: {clean_text}"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Qwen3Guard
# ══════════════════════════════════════════════════════════════════════════════

def run_guard(transcript: str, tone: dict, fusion_enabled: bool) -> dict:
    """
    Run the Qwen3Guard safety classifier on the (optionally fused) prompt.

    Parameters
    ----------
    transcript : str
        Whisper transcript.
    tone : dict
        SpeechBrain tone response (ignored when ``fusion_enabled=False``).
    fusion_enabled : bool
        True → System B (fusion); False → System A (text-only).

    Returns
    -------
    dict
        Keys: ``raw``, ``safety``, ``categories``, ``fused_text``, ``guard_json``.
    """
    tone_for_fusion = tone if fusion_enabled else {}
    fused_text = build_fused_input(transcript, tone_for_fusion)

    headers = {
        "Authorization": "Bearer EMPTY",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      QWENGUARD_MODEL,
        "messages":   [{"role": "user", "content": fused_text}],
        "temperature": 0,
        "max_tokens":  MAX_TOKENS_GUARD,
    }

    r = requests.post(QWENGUARD_URL, headers=headers, json=payload,
                      timeout=GUARD_TIMEOUT_S)
    r.raise_for_status()
    guard_json = r.json()

    raw      = guard_json["choices"][0]["message"]["content"].strip()
    safety, categories = parse_guard(raw)
    safety   = normalize_safety_label(safety, strict_mode=STRICT_MODE)

    return {
        "raw":        raw,
        "safety":     safety or "",
        "categories": categories,
        "fused_text": fused_text,
        "guard_json": guard_json,
    }


# ══════════════════════════════════════════════════════════════════════════════
# API endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health_check():
    """
    Probe all three GPU services and return their connectivity status.

    Returns
    -------
    JSON
        ``{"all_ok": bool, "servers": {"whisper": {...}, "tone": {...}, "guard": {...}}}``

    Each server entry contains:
      - ``url``   : target URL
      - ``ok``    : True if HTTP status < 500
      - ``error`` : error string (empty on success)
      - ``label`` : human-readable service label
    """
    status = {
        "whisper": {
            "url":   f"http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1",
            "ok":    False, "error": "",
            "label": "Whisper ASR · Server A :8001",
        },
        "tone": {
            "url":   TONE_API_URL,
            "ok":    False, "error": "",
            "label": "SpeechBrain SER · Server A :9100",
        },
        "guard": {
            "url":   QWENGUARD_URL,
            "ok":    False, "error": "",
            "label": "Qwen3Guard LLM · Server B :8000",
        },
    }

    # Probe Whisper vLLM models endpoint
    try:
        r = requests.get(
            f"http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1/models", timeout=6
        )
        status["whisper"]["ok"] = r.status_code < 500
    except Exception as e:
        status["whisper"]["error"] = str(e)[:100]

    # Probe SpeechBrain root (HEAD to avoid side-effects)
    try:
        r = requests.head(TONE_API_URL.replace("/v1/audio/tone", ""), timeout=6)
        status["tone"]["ok"] = r.status_code < 500
    except Exception as e:
        status["tone"]["error"] = str(e)[:100]

    # Probe Qwen3Guard vLLM models endpoint
    try:
        guard_models = QWENGUARD_URL.replace("/v1/chat/completions", "/v1/models")
        r = requests.get(guard_models, timeout=6)
        status["guard"]["ok"] = r.status_code < 500
    except Exception as e:
        status["guard"]["error"] = str(e)[:100]

    return jsonify({
        "all_ok":  all(v["ok"] for v in status.values()),
        "servers": status,
    })


@app.route("/process", methods=["POST"])
def process_audio():
    """
    Run the full SEC619 pipeline on an uploaded audio file.

    Form fields
    -----------
    file        : audio file (multipart/form-data)
    mode        : ``"text-only"`` (System A) or ``"fusion"`` (System B)
    strict_mode : ``"true"`` or ``"false"`` — whether to escalate Controversial→Unsafe

    Returns
    -------
    JSON
        Result object containing transcript, tone, safety verdict, categories,
        per-stage latencies, and status.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded       = request.files["file"]
    mode           = request.form.get("mode", "text-only")
    strict_mode    = request.form.get("strict_mode", "true").lower() == "true"
    fusion_enabled = (mode == "fusion")

    filename = uploaded.filename or "audio.wav"
    ext      = Path(filename).suffix.lower()

    # Save upload to a temporary file so the pipeline functions can read it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        uploaded.save(tmp.name)
        tmp_path = Path(tmp.name)

    # Result skeleton — populated stage by stage
    result = {
        "name":        filename,
        "size":        tmp_path.stat().st_size,
        "mode":        mode,
        "strict":      strict_mode,
        "transcript":  "",
        "tone":        {},
        "fused_text":  "",
        "guard_raw":   "",
        "safety":      "",
        "categories":  [],
        "latency":     {"asr": 0, "ser": 0, "fusion": 0, "guard": 0, "total": 0},
        "status":      "OK",
        "error":       "",
    }
    t0_total = time.time()

    # ── Stage 1: Whisper ASR ─────────────────────────────────────────────────
    try:
        t0 = time.time()
        result["transcript"] = retry_call(
            lambda: whisper_transcribe(tmp_path), name="Whisper"
        )
        result["latency"]["asr"] = round(time.time() - t0, 3)
    except Exception as e:
        result["status"] = "FAIL_WHISPER"
        result["error"]  = str(e)
        tmp_path.unlink(missing_ok=True)
        result["latency"]["total"] = round(time.time() - t0_total, 3)
        return jsonify(result)

    if not result["transcript"].strip():
        result["status"] = "SKIP_EMPTY_TRANSCRIPT"
        result["error"]  = "Whisper returned an empty transcript."
        tmp_path.unlink(missing_ok=True)
        result["latency"]["total"] = round(time.time() - t0_total, 3)
        return jsonify(result)

    # ── Stage 2: SpeechBrain SER ─────────────────────────────────────────────
    # SER always runs regardless of mode to ensure fair latency measurements.
    # The tone output is only used in System B (fusion_enabled=True).
    try:
        t0 = time.time()
        result["tone"] = retry_call(
            lambda: tone_from_api(tmp_path), name="ToneAPI"
        )
        result["latency"]["ser"] = round(time.time() - t0, 3)
    except Exception as e:
        # Non-fatal: fall back to text-only if SER fails
        result["status"] = "WARN_TONE_FAILED"
        result["error"]  = f"SER failed (fallback to text-only): {e}"
        result["tone"]   = {}

    # ── Fusion layer (instantaneous, deterministic) ──────────────────────────
    t0 = time.time()
    result["latency"]["fusion"] = round(time.time() - t0, 4)

    # ── Stage 3: Qwen3Guard ───────────────────────────────────────────────────
    try:
        t0 = time.time()
        g = retry_call(
            lambda: run_guard(result["transcript"], result["tone"], fusion_enabled),
            name="QwenGuard",
        )
        result["fused_text"]        = g["fused_text"]
        result["guard_raw"]         = g["raw"]
        result["safety"]            = g["safety"]
        result["categories"]        = g["categories"]
        result["latency"]["guard"]  = round(time.time() - t0, 3)
    except Exception as e:
        result["status"] = "FAIL_GUARD"
        result["error"]  = str(e)

    result["latency"]["total"] = round(time.time() - t0_total, 3)
    tmp_path.unlink(missing_ok=True)
    return jsonify(result)


@app.route("/")
def index():
    """Serve the browser UI from demo/static/index.html."""
    return send_from_directory("static", "index.html")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  ⚑  SheriffSpeech — Voice Threat Detection Server")
    print("  " + "─" * 62)
    print(f"  Whisper  ASR : http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1")
    print(f"  SpeechBrain  : {TONE_API_URL}")
    print(f"  Qwen3Guard   : {QWENGUARD_URL}")
    print()
    print("  Mode: System A = text-only | System B = fusion")
    print("  API:  GET /health   POST /process")
    print("  UI:   http://localhost:5000")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
