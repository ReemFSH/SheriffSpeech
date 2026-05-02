#!/bin/bash
# ============================================================================
# SEC619 — GPU Server Startup Commands
# File    : servers/server_startup.sh
# Author  : Reem Fuad Shareef
# ============================================================================
#
# PURPOSE
# -------
# Documents the exact commands to launch each AI model server before
# running the pipeline. This project uses TWO GPU servers (TensorDock):
#
#   Server A — Whisper ASR (port 8001) + SpeechBrain SER (port 9100)
#   Server B — Qwen3Guard LLM (port 8000)
#
# USAGE
# -----
# Run each command block in its own terminal session on the respective server.
# Whisper and SpeechBrain must BOTH be running on Server A before starting
# the pipeline.
#
# ============================================================================


# ============================================================================
# SERVER A — Whisper Large-v3 (ASR)
# ============================================================================
# Uses vLLM's OpenAI-compatible server to serve Whisper as a REST API.
# The pipeline sends WAV files and receives plain text transcripts.
#
# Port : 8001
# Model: openai/whisper-large-v3
# ============================================================================

echo "=== Starting Whisper (Server A, port 8001) ==="

# Activate the Whisper virtual environment
source ~/whisper_vllm/bin/activate

# Launch vLLM server with Whisper Large-v3
# --host 0.0.0.0 makes it reachable from the pipeline client machine
# --port 8001    matches WHISPER_PORT in pipeline/pipeline.py
vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8001

# Test that Whisper is running (run in a separate terminal):
# curl http://localhost:8001/v1/models


# ============================================================================
# SERVER A — SpeechBrain Emotion API (SER)
# ============================================================================
# FastAPI server wrapping the SpeechBrain wav2vec2-IEMOCAP classifier.
# See servers/speechbrain_api.py for the full server implementation.
#
# Port : 9100
# Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
# ============================================================================

echo "=== Starting SpeechBrain API (Server A, port 9100) ==="

# Activate the VoiceGuard virtual environment (contains speechbrain + uvicorn)
source ~/venvs/voiceguard/bin/activate

# Launch FastAPI server with uvicorn
# The speechbrain_api module is servers/speechbrain_api.py
# --host 0.0.0.0 makes it reachable externally
# --port 9100    matches TONE_API_URL in pipeline/pipeline.py
uvicorn servers.speechbrain_api:app \
    --host 0.0.0.0 \
    --port 9100

# Optional: set API key for authentication
# export TONE_API_KEY="your_strong_key_here"
# Then update TONE_API_KEY in pipeline/pipeline.py to match.

# Test that SpeechBrain is running (run in a separate terminal):
# curl -X POST http://localhost:9100/v1/audio/tone -F "file=@test.wav"


# ============================================================================
# SERVER B — Qwen3Guard-Gen-8B (LLM Safety Classifier)
# ============================================================================
# Uses vLLM to serve Qwen3Guard as an OpenAI-compatible chat completions API.
# The pipeline sends fused (emotion + transcript) text and receives
# structured "Safety: / Categories:" responses.
#
# Port  : 8000
# Model : Qwen/Qwen3Guard-Gen-8B
# Note  : --max-model-len 8192 limits context window to 8K tokens
#         (sufficient for all transcripts in this dataset)
# ============================================================================

echo "=== Starting Qwen3Guard (Server B, port 8000) ==="

# Change to the Qwen3Guard vLLM directory and activate its environment
cd qwen3guard_vllm/
source bin/activate

# Launch vLLM server with Qwen3Guard
# --host 0.0.0.0      makes it reachable from Server A / pipeline client
# --port 8000         matches QWENGUARD_URL in pipeline/pipeline.py
# --max-model-len 8192  limit context window (reduces VRAM requirement)
vllm serve Qwen/Qwen3Guard-Gen-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192

# Test that Qwen3Guard is running:
# curl http://localhost:8000/v1/models


# ============================================================================
# VERIFICATION CHECKLIST
# ============================================================================
# Before running pipeline/pipeline.py, confirm all three services are up:
#
#   Server A - Whisper    : curl http://SERVER_A_IP:8001/v1/models
#   Server A - SpeechBrain: curl http://SERVER_A_IP:9100/docs
#   Server B - Qwen3Guard : curl http://SERVER_B_IP:8000/v1/models
#
# Update pipeline/pipeline.py SECTION D with your actual server IPs:
#   WHISPER_SERVER_IP = "YOUR_SERVER_A_IP"
#   QWENGUARD_URL     = "http://YOUR_SERVER_B_IP:8000/v1/chat/completions"
# ============================================================================
