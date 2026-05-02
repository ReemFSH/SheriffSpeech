# ============================================================================
# SEC619 — SpeechBrain Emotion Recognition REST API Server
# File    : servers/speechbrain_api.py
# Author  : Reem Fuad Shareef
# Supervisor: Dr. Waleed Algobi
# ============================================================================
#
# PURPOSE
# -------
# FastAPI server that wraps the SpeechBrain wav2vec2-IEMOCAP emotion classifier
# and exposes it as a REST API endpoint for the pipeline to call remotely.
#
# This server runs on GPU Server A alongside Whisper.
# The pipeline (pipeline/pipeline.py) sends audio files to this server via HTTP.
#
# ENDPOINT
# --------
#   POST /v1/audio/tone
#   Content-Type: multipart/form-data
#   Body: file=<wav_file>
#   Header: X-API-Key: <key>  (optional — set TONE_API_KEY env var to enable)
#
#   Response JSON:
#   {
#       "model": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#       "device": "cpu",
#       "label_short": "ang",
#       "label_full": "Angry",
#       "top_p": 0.87,
#       "top3": [
#           {"label_short": "ang", "label_full": "Angry",   "p": 0.87},
#           {"label_short": "neu", "label_full": "Neutral",  "p": 0.08},
#           {"label_short": "sad", "label_full": "Sad",      "p": 0.05}
#       ]
#   }
#
# EMOTION CLASSES
# ---------------
# The wav2vec2-IEMOCAP model returns 4-class labels:
#   ang → Angry    (high arousal, potentially threat-correlated)
#   hap → Happy    (alias for Cheerful in 100-sample dataset)
#   sad → Sad      (low arousal, distress)
#   neu → Neutral  (baseline calm speech)
#
# STARTUP
# -------
#   pip install fastapi uvicorn speechbrain
#   uvicorn servers.speechbrain_api:app --host 0.0.0.0 --port 9100
#
# ENVIRONMENT VARIABLES
# ---------------------
#   TONE_DEVICE  : "cpu" (default) or "cuda"  — inference device
#   TONE_API_KEY : Optional API key for endpoint protection
# ============================================================================

import os
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from speechbrain.inference.interfaces import foreign_class


# ============================================================================
# CONFIGURATION
# ============================================================================

# Hugging Face model ID for the SpeechBrain emotion classifier
MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# Inference device: "cpu" recommended for stability on shared GPU servers
# Set to "cuda" if the server has a dedicated GPU and sufficient VRAM
DEVICE = os.getenv("TONE_DEVICE", "cpu")

# Optional API key protection — set via environment variable to avoid hardcoding
# Example: export TONE_API_KEY="your_strong_key"
# Leave empty ("") to disable authentication
API_KEY = os.getenv("TONE_API_KEY", "")

# Mapping from SpeechBrain short labels to human-readable full labels
# The model outputs short codes (ang, hap, sad, neu) from the IEMOCAP dataset
EMO_MAP = {
    "ang": "Angry",
    "hap": "Happy",     # Note: pipeline maps this to "Cheerful" for 100-sample dataset
    "sad": "Sad",
    "neu": "Neutral",
}


# ============================================================================
# FastAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="SpeechBrain Tone API",
    description="SEC619 — Emotion Recognition REST API (wav2vec2-IEMOCAP)",
    version="1.0.0",
)

# ── Load SpeechBrain model at startup (once, not per request) ─────────────────
# foreign_class loads a custom interface class from the model's Hugging Face repo.
# This is the correct way to load SpeechBrain models that use a custom_interface.py.
clf = foreign_class(
    source=MODEL_ID,
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": DEVICE},
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def require_key(x_api_key: str):
    """
    Validate the API key from the request header.
    Raises HTTP 401 if API_KEY is set and the provided key does not match.
    Authentication is disabled (no-op) when API_KEY is empty.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_ind2lab() -> list:
    """
    Build an index-to-label list from the SpeechBrain label encoder.

    This ensures the correct mapping between out_prob tensor indices and
    emotion labels. The order of indices in the probability tensor may
    differ from the EMO_MAP order, so we must use the model's own encoder.

    Returns:
        List[str]: Label names indexed by their position in the probability tensor.
                   Returns empty list if the encoder cannot be read.
    """
    le = clf.hparams.label_encoder   # SpeechBrain LabelEncoder object
    lab2ind = getattr(le, "lab2ind", None)

    if not isinstance(lab2ind, dict) or not lab2ind:
        return []   # Fallback: encoder not accessible

    # Invert lab2ind dict to get ind2lab list
    ind2lab = [None] * (max(lab2ind.values()) + 1)
    for lab, idx in lab2ind.items():
        if 0 <= idx < len(ind2lab):
            ind2lab[idx] = lab

    return [x for x in ind2lab if x is not None]


# Cache the label order once at startup (fast and consistent across requests)
IND2LAB = get_ind2lab()


# ============================================================================
# ENDPOINT
# ============================================================================

@app.post("/v1/audio/tone")
async def tone(
    file: UploadFile = File(...),
    x_api_key: str = Header(default="")
) -> Dict[str, Any]:
    """
    Classify the vocal emotion of an uploaded audio file.

    Accepts a WAV file via multipart form upload, saves it to a temporary
    file (required by SpeechBrain's classify_file method), runs inference,
    and returns the emotion prediction with confidence scores.

    The temporary file is always deleted after inference, even if an error
    occurs, to prevent disk space accumulation on long-running servers.

    Args:
        file       : Uploaded WAV audio file (multipart/form-data).
        x_api_key  : Optional API key header for authentication.

    Returns:
        JSON dict with fields: model, device, label_short, label_full,
        top_p, top3 (list of top-3 emotion probabilities).
    """
    require_key(x_api_key)

    # Preserve the original file extension for SpeechBrain compatibility
    suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "wav")

    tmp_path = None
    try:
        # Save uploaded bytes to a named temp file
        # SpeechBrain's classify_file() requires a file path, not a file object
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run SpeechBrain inference
        # Returns: (out_prob, score, index, text_lab)
        #   out_prob : Tensor of shape [1, num_classes] — softmax probabilities
        #   score    : Probability of the top-1 prediction
        #   index    : Index of the top-1 prediction
        #   text_lab : Short label of the top-1 prediction (e.g. "ang")
        out_prob, score, index, text_lab = clf.classify_file(tmp_path)

        # ── Extract dominant emotion label ─────────────────────────────────────
        lab_short = text_lab[0] if isinstance(text_lab, (list, tuple)) and len(text_lab) else str(text_lab)
        lab_short = str(lab_short).strip()
        lab_full  = EMO_MAP.get(lab_short, lab_short)   # Fall back to raw label if not in map

        # ── Extract top-1 confidence score ─────────────────────────────────────
        try:
            top_p = float(score.squeeze().item())
        except Exception:
            try:
                top_p = float(score)
            except Exception:
                top_p = None

        # ── Extract top-3 probability distribution ────────────────────────────
        # Uses IND2LAB to correctly map tensor indices to label names.
        # Falls back to empty list on any error to avoid breaking the response.
        top3 = []
        try:
            probs      = out_prob.squeeze()
            probs_list = [float(x) for x in probs]

            if not IND2LAB or len(IND2LAB) != len(probs_list):
                # Length mismatch or missing encoder — skip top3 to avoid wrong mapping
                pairs = []
            else:
                pairs = list(zip(IND2LAB, probs_list))

            pairs.sort(key=lambda x: x[1], reverse=True)   # Sort by probability descending

            for k, p in pairs[:3]:
                top3.append({
                    "label_short": k,
                    "label_full" : EMO_MAP.get(k, k),
                    "p"          : float(p),
                })
        except Exception:
            pass   # top3 stays [] — non-fatal

        return {
            "model"      : MODEL_ID,
            "device"     : DEVICE,
            "label_short": lab_short,
            "label_full" : lab_full,
            "top_p"      : top_p,
            "top3"       : top3,
        }

    finally:
        # Always clean up the temp file — prevents disk from filling up on long runs
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass   # Non-fatal — file will be cleaned by OS eventually
