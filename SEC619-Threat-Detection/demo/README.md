# SheriffSpeech — Interactive Demo

A browser-based demonstration of the SEC619 three-stage multimodal
threat-detection pipeline. Drop an audio file, choose a mode, and watch
the system transcribe, analyse emotion, and classify safety in real time.

---

## Contents

```
demo/
├── app.py               Flask backend — full pipeline exposed as REST API
├── static/
│   └── index.html       Browser UI (drop-zone, live results, latency panel)
├── architecture.html    Interactive pipeline architecture diagram
└── README.md            This file
```

## Quick Start

### 1 — Prerequisites

```bash
# The demo server only needs these three packages
pip install flask requests openai
```

All three GPU services (Whisper, SpeechBrain, Qwen3Guard) must be running.
See [`servers/server_startup.sh`](../servers/server_startup.sh) for launch commands.

### 2 — Configure server IPs

Open `demo/app.py` and set the IP addresses at the top of the file:

```python
WHISPER_SERVER_IP = "YOUR_SERVER_A_IP"   # runs Whisper :8001 + SpeechBrain :9100
QWENGUARD_URL     = "http://YOUR_SERVER_B_IP:8000/v1/chat/completions"
```

### 3 — Start the demo server

```bash
# From the repo root
python demo/app.py
```

Expected output:

```
  ⚑  SheriffSpeech — Voice Threat Detection Server
  ──────────────────────────────────────────────────────────────
  Whisper  ASR : http://<SERVER_A>:8001/v1
  SpeechBrain  : http://<SERVER_A>:9100/v1/audio/tone
  Qwen3Guard   : http://<SERVER_B>:8000/v1/chat/completions

  Mode: System A = text-only | System B = fusion
  API:  GET /health   POST /process
  UI:   http://localhost:5000
```

### 4 — Open the UI

Navigate to **http://localhost:5000** in your browser.

---

## Pipeline Modes

| Mode | System | What runs |
|------|--------|-----------|
| **Text-only** | System A | Whisper → Qwen3Guard |
| **Fusion** | System B | Whisper → SpeechBrain → Fusion Layer → Qwen3Guard |

Toggle the mode using the switch in the UI before uploading audio.

---

## API Reference

### `GET /health`

Checks connectivity to all three GPU services.

```json
{
  "all_ok": true,
  "servers": {
    "whisper": { "url": "...", "ok": true,  "error": "", "label": "Whisper ASR · Server A :8001" },
    "tone":    { "url": "...", "ok": true,  "error": "", "label": "SpeechBrain SER · Server A :9100" },
    "guard":   { "url": "...", "ok": true,  "error": "", "label": "Qwen3Guard LLM · Server B :8000" }
  }
}
```

### `POST /process`

Run the full pipeline on an uploaded audio file.

**Form fields:**

| Field | Type | Values |
|-------|------|--------|
| `file` | file | `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.webm`, `.aac` |
| `mode` | string | `"text-only"` (default) or `"fusion"` |
| `strict_mode` | string | `"true"` (default) or `"false"` |

**Response (JSON):**

```json
{
  "name":       "audio.wav",
  "size":       44100,
  "mode":       "fusion",
  "strict":     true,
  "transcript": "I want to harm someone.",
  "tone":       { "label_full": "Angry", "top_p": 0.94, "top3": [...] },
  "fused_text": "[Audio context: The speaker sounds very angry (ang=0.94).]\n\nTranscript: ...",
  "guard_raw":  "Safety: Unsafe\nCategories: Violent",
  "safety":     "Unsafe",
  "categories": ["Violent"],
  "latency":    { "asr": 1.23, "ser": 0.87, "fusion": 0.001, "guard": 2.11, "total": 4.22 },
  "status":     "OK",
  "error":      ""
}
```

---

## Architecture Diagram

Open `demo/architecture.html` directly in any browser for an interactive
visual overview of the full pipeline — from audio input through ASR, SER,
fusion, and Guard classification to the final safety verdict.

---

## Notes

- **STRICT_MODE** (enabled by default): promotes `Controversial` verdicts to
  `Unsafe`. Set `STRICT_MODE = False` in `app.py` to preserve the three-class
  output.
- **Retry logic**: each GPU call is retried up to 3 times with a 5-second
  delay between attempts to handle transient network issues.
- **Language**: change `LANGUAGE = "ar"` in `app.py` for Arabic audio.
