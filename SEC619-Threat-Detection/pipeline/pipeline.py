# ============================================================================
# SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
# File    : pipeline/pipeline.py
# Author  : Reem Fuad Shareef
# Supervisor: Dr. Waleed Algobi
# Course  : SEC619 — Graduation Project · KFUPM · Term 242
# ============================================================================
#
# PURPOSE
# -------
# Main evaluation pipeline that processes audio files through three sequential
# AI stages and classifies each as Safe or Unsafe. Supports two run modes
# (System A baseline vs System B fusion) for A/B comparison experiments.
#
# PIPELINE OVERVIEW
# -----------------
#   Stage 1 — ASR  : Whisper Large-v3
#                    Converts raw WAV audio into plain text transcript.
#                    Runs on GPU Server A, port 8001.
#
#   Stage 2 — SER  : SpeechBrain (wav2vec2-IEMOCAP)
#                    Extracts dominant vocal emotion from the raw waveform.
#                    Returns a 4-class label: Angry · Neutral · Cheerful · Sad
#                    Runs on GPU Server A, port 9100.
#
#   Stage 3 — LLM  : Qwen3Guard-Gen-8B
#                    Classifies the FUSED input (emotion prefix + transcript)
#                    into: Safe / Controversial / Unsafe
#                    and identifies threat categories from the 9-class taxonomy.
#                    Runs on GPU Server B, port 8000.
#
# FUSION DESIGN
# -------------
# Qwen3Guard is a text-only LLM. To give it acoustic emotion signals,
# SpeechBrain's output is converted into a natural-language prefix prepended
# to the Whisper transcript before classification:
#
#   System B input to Qwen3Guard:
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ [Audio context: The speaker sounds very angry                   │
#   │  (Angry=0.87, Neutral=0.08, Sad=0.05).]                        │
#   │                                                                  │
#   │ Transcript: I know where you live and I will make you regret... │
#   └──────────────────────────────────────────────────────────────────┘
#
#   System A input to Qwen3Guard (baseline — no emotion context):
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ I know where you live and I will make you regret this.         │
#   └──────────────────────────────────────────────────────────────────┘
#
# A/B EVALUATION
# --------------
# Change the single flag FUSION_ENABLED between two runs on the same dataset:
#   Run 1: FUSION_ENABLED = False  → System A results  → multimodal_result.csv
#   Run 2: FUSION_ENABLED = True   → System B results  → multimodal_result_fusion.csv
# Then run evaluation/compare_systems.py to compute the delta metrics.
#
# CYBERSECURITY POLICY
# --------------------
# STRICT_MODE = True promotes any "Controversial" Qwen3Guard verdict to
# "Unsafe". Rationale: in security contexts, the cost of a missed threat
# (False Negative) outweighs the cost of a false alarm (False Positive).
#
# OUTPUT FILES (per run)
# ----------------------
#   results/output_csv/multimodal_result.csv           — System A summary
#   results/output_csv/multimodal_result_fusion.csv    — System B summary
#   results/output_csv/results_json/<id>_<hash>.json   — Detailed per-file bundles
# ============================================================================

# ── Standard library ─────────────────────────────────────────────────────────
import re           # Regular expressions — parse Qwen3Guard text output
import csv          # CSV writer — log one result row per audio file
import json         # JSON serialisation — manifest loading & per-file bundles
import time         # Timing — measure latency of each pipeline stage
from pathlib import Path        # OS-independent file path handling
from datetime import datetime   # ISO timestamp on each processed file

# ── Third-party ───────────────────────────────────────────────────────────────
import requests             # HTTP calls to SpeechBrain and Qwen3Guard REST APIs
from openai import OpenAI   # OpenAI-compatible client → Whisper vLLM endpoint
from tqdm import tqdm       # Progress bar during batch processing


# ============================================================================
# SECTION A — ANSI COLOUR CODES
# ============================================================================
# Applied only to terminal output. Stripped before writing to CSV/JSON so
# output files contain clean plain text.
# ============================================================================

class C:
    """
    ANSI escape code constants for coloured terminal output.

    Usage:
        print(C.GREEN + "Success!" + C.RESET)
        plain = C.strip(colored_string)  # remove codes before file write
    """
    RESET   = "\033[0m"    # Cancel all active styles
    BOLD    = "\033[1m"    # Bold text
    DIM     = "\033[2m"    # Dimmed text (secondary info)

    # Foreground colours
    RED     = "\033[91m"   # Errors, UNSAFE verdict
    GREEN   = "\033[92m"   # Success, SAFE verdict, correct predictions
    YELLOW  = "\033[93m"   # Warnings, Controversial
    CYAN    = "\033[96m"   # Section headers and banners
    WHITE   = "\033[97m"   # Primary text on coloured backgrounds
    MAGENTA = "\033[95m"   # Emotion labels and SpeechBrain output
    BLUE    = "\033[94m"   # Sad emotion colour

    # Background colours (used in FINAL VERDICT box)
    BG_RED    = "\033[41m"    # Red background   → UNSAFE
    BG_GREEN  = "\033[42m"    # Green background → SAFE
    BG_YELLOW = "\033[43m"    # Yellow background → CONTROVERSIAL
    BG_GRAY   = "\033[100m"   # Grey background  → UNKNOWN / error

    @staticmethod
    def strip(text: str) -> str:
        """Remove all ANSI escape sequences — called before writing to file."""
        return re.sub(r"\033\[[0-9;]*m", "", text)


# ============================================================================
# SECTION B — PATH CONFIGURATION
# ============================================================================
# All paths are defined relative to this file's location so the project
# works on any machine without editing hardcoded absolute paths.
#
# Directory layout expected:
#   SEC619-Threat-Detection/
#   ├── pipeline/
#   │   └── pipeline.py          ← this file
#   ├── dataset/
#   │   ├── audio/               ← WAV files to process
#   │   └── dataset_100_samples.json
#   └── results/
#       └── output_csv/          ← CSV and JSON outputs written here
# ============================================================================

# Root of the repository (one level up from pipeline/)
REPO_ROOT = Path(__file__).parent.parent

# Input: folder containing the 100 WAV audio files
INPUT_DIR = REPO_ROOT / "dataset" / "audio"

# Output: folder for CSV results and JSON bundles
OUT_DIR = REPO_ROOT / "results" / "output_csv"

# Dataset manifest: maps each audio file to its ground-truth label
MANIFEST_PATH = REPO_ROOT / "dataset" / "dataset_100_samples.json"

# Audio file extensions to process
EXTS = {".wav"}


# ============================================================================
# SECTION C — GROUND TRUTH LOADER
# ============================================================================
# Reads dataset_100_samples.json at startup to build a filename → label dict.
# Used to compute accuracy and confusion matrix during batch evaluation.
#
# Key   : "C01_VIO_S01.wav"   (filename only, no directory)
# Value : "Unsafe" | "Safe"
#
# If the manifest file is missing, evaluation is disabled silently and the
# pipeline still processes files normally (no crash).
# ============================================================================

GROUND_TRUTH: dict[str, str] = {}

if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, encoding="utf-8") as _f:
        # Each JSON entry has "id" (e.g. "C01_VIO_S01") and "ground_truth" field.
        # Append ".wav" to match the actual audio filename on disk.
        for _s in json.load(_f):
            GROUND_TRUTH[_s["id"] + ".wav"] = _s["ground_truth"]
    print(C.GREEN + f"✓ Ground truth loaded: {len(GROUND_TRUTH)} samples" + C.RESET)
else:
    print(C.YELLOW + "⚠  Manifest not found — evaluation metrics disabled." + C.RESET)
    print(C.YELLOW + f"   Expected: {MANIFEST_PATH}" + C.RESET)


# ============================================================================
# SECTION D — USER SETTINGS
# ============================================================================
# All user-configurable parameters. Edit ONLY this section to change dataset,
# server addresses, or run mode.
# ============================================================================

# ── Server configuration ──────────────────────────────────────────────────────
# Replace with your actual GPU server IPs and ports.
# Server A hosts Whisper ASR (port 8001) and SpeechBrain SER (port 9100).
# Server B hosts Qwen3Guard LLM (port 8000).

WHISPER_SERVER_IP = "YOUR_SERVER_A_IP"    # e.g. "38.224.253.168"
WHISPER_PORT      = 8001
TONE_API_URL      = f"http://{WHISPER_SERVER_IP}:9100/v1/audio/tone"
TONE_API_KEY      = ""                     # Set if SpeechBrain server requires auth

QWENGUARD_URL   = "http://YOUR_SERVER_B_IP:8000/v1/chat/completions"
QWENGUARD_MODEL = "Qwen/Qwen3Guard-Gen-8B"

# ── Model / inference settings ────────────────────────────────────────────────
LANGUAGE          = "en"     # Whisper language hint ("en", "ar", etc.)
TEMPERATURE       = 0.0      # 0.0 = deterministic / greedy decoding
MAX_TOKENS_GUARD  = 128      # Sufficient for "Safety: / Categories:" output
SAVE_JSON         = True     # Save a detailed JSON bundle per audio file
RESUME            = False    # Skip files already in the output CSV
SHOW_SAVED_RESULTS_WHEN_RESUME = True

# Cybersecurity policy:
# True  → Controversial → Unsafe  (secure default — escalate ambiguous content)
# False → Keep as Controversial   (useful for research/analysis runs)
STRICT_MODE = True

# ── A/B evaluation flag ───────────────────────────────────────────────────────
# This is the ONLY setting that changes between Run 1 and Run 2.
#
#   Run 1 → System A (baseline, text-only):
#       FUSION_ENABLED = False
#       CSV_NAME       = "multimodal_result.csv"
#
#   Run 2 → System B (full fusion pipeline):
#       FUSION_ENABLED = True
#       CSV_NAME       = "multimodal_result_fusion.csv"
FUSION_ENABLED = True

# ── Output filenames ──────────────────────────────────────────────────────────
# Change CSV_NAME between runs as described above.
CSV_NAME      = "multimodal_result_fusion.csv"
JSON_DIR_NAME = "results_json"    # Subfolder for per-file JSON bundles

# ── Console preview limits ────────────────────────────────────────────────────
# Max characters shown in terminal for transcript / guard output.
# Does NOT affect CSV/JSON output (always full text).
TRANSCRIPT_PREVIEW_CHARS = 260
GUARD_PREVIEW_CHARS      = 260

# ── Network timeouts and retry settings ──────────────────────────────────────
TONE_TIMEOUT_S        = 300     # Seconds before SpeechBrain call times out
GUARD_TIMEOUT_S       = 300     # Seconds before Qwen3Guard call times out
SLEEP_BETWEEN_FILES_S = 0.4     # Cooldown between files (reduces GPU load)
RETRIES               = 3       # Max retry attempts on transient network errors
RETRY_SLEEP_S         = 2.0     # Wait time between retries


# ============================================================================
# SECTION E — OUTPUT FOLDER SETUP
# ============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_DIR = OUT_DIR / JSON_DIR_NAME
if SAVE_JSON:
    JSON_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / CSV_NAME


# ============================================================================
# SECTION F — QWEN3GUARD OUTPUT PARSING
# ============================================================================
# Qwen3Guard returns structured plain text (not JSON):
#
#   Safety: Unsafe
#   Categories: Violent, Unethical Acts
#
# Regex patterns extract both fields from the raw response text.
# ============================================================================

# Matches "Safety: Safe|Unsafe|Controversial" (case-insensitive)
SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)

# Matches "Categories: ..." capturing everything after the colon
CATS_PATTERN = re.compile(r"Categories:\s*(.*)", re.IGNORECASE | re.DOTALL)


def parse_guard(content: str):
    """
    Parse raw Qwen3Guard output text into (safety_label, categories_list).

    Args:
        content: Raw string from Qwen3Guard message content field.

    Returns:
        tuple: (safety_label, categories_list)
            safety_label    : "Safe" | "Unsafe" | "Controversial" | None
            categories_list : list of category strings (may be empty)

    Handles comma-separated categories, newline-separated categories,
    "None" as a category string, and duplicate categories.
    """
    safety     = None
    categories = []

    # Extract safety label from "Safety: ..." line
    m = SAFE_PATTERN.search(content or "")
    if m:
        safety = m.group(1).capitalize()  # normalise capitalisation

    # Extract categories from "Categories: ..." line
    m2 = CATS_PATTERN.search(content or "")
    if m2:
        raw = (m2.group(1) or "").strip()
        if raw:
            # Support both comma-separated and newline-separated formats
            categories = (
                [p.strip() for p in raw.split(",") if p.strip()]
                if "," in raw
                else [ln.strip() for ln in raw.splitlines() if ln.strip()]
            )
        # Qwen3Guard writes "None" when no category applies → empty list
        if len(categories) == 1 and categories[0].lower() == "none":
            categories = []

    # Deduplicate while preserving insertion order
    seen, dedup = set(), []
    for c in categories:
        if c not in seen:
            seen.add(c)
            dedup.append(c)

    return safety, dedup


def normalize_safety_label(safety: str, strict_mode: bool = True) -> str:
    """
    Apply the cybersecurity safety policy to the raw Qwen3Guard verdict.

    When strict_mode=True:  "Controversial" → "Unsafe"
    When strict_mode=False: All labels returned unchanged.
    """
    if strict_mode and safety == "Controversial":
        return "Unsafe"
    return safety


# ============================================================================
# SECTION G — DISPLAY UTILITIES
# ============================================================================

def shorten(text: str, n: int) -> str:
    """
    Return a single-line, n-character preview of text (collapses newlines).
    Used for transcript and guard output previews in console and CSV.
    """
    if not text:
        return ""
    t = str(text).strip().replace("\n", " ")
    return t if len(t) <= n else t[:n] + " ..."


def hr(ch="─", n=90) -> str:
    """Return a horizontal divider line."""
    return ch * n


def banner(title: str):
    """Print a large, highly visible section header."""
    print("\n" + C.CYAN + hr("═") + C.RESET)
    print(C.BOLD + C.WHITE + title + C.RESET)
    print(C.CYAN + hr("═") + C.RESET)


def block(title: str, body: str):
    """Print a labelled content block — used for transcript/tone/guard sections."""
    print("\n" + C.DIM + hr("─") + C.RESET)
    print(C.BOLD + C.CYAN + title + C.RESET)
    print(C.DIM + hr("─") + C.RESET)
    print(body)


# Emotion label → ANSI colour mapping for colour-coded terminal output
EMOTION_COLOURS = {
    "angry":    C.RED,
    "neutral":  C.DIM,
    "cheerful": C.GREEN,
    "sad":      C.BLUE,
    "happy":    C.GREEN,   # Alias — retained for backward compatibility
}


def emotion_colour(label: str) -> str:
    """Return the ANSI colour for a given emotion label (case-insensitive)."""
    return EMOTION_COLOURS.get((label or "").lower(), C.WHITE)


def _verdict_style(safety: str):
    """Return (background_colour, icon, label_colour) for the FINAL VERDICT box."""
    s = (safety or "").upper()
    if s == "UNSAFE":        return C.BG_RED,    "🚨", C.RED
    elif s == "SAFE":        return C.BG_GREEN,  "✅", C.GREEN
    elif s == "CONTROVERSIAL": return C.BG_YELLOW, "⚠️ ", C.YELLOW
    else:                    return C.BG_GRAY,   "❓", C.WHITE


def _emotion_bar(top3: list, bar_width: int = 20) -> str:
    """
    Render an ASCII confidence bar for each emotion in the SpeechBrain top-3.
    Each bar: <emotion label>   ████████████░░░░░░░░  <probability>
    """
    lines = []
    for item in (top3 or []):
        label  = item.get("label_full", item.get("label_short", "?"))
        p      = item.get("p", 0.0)
        filled = int(round(p * bar_width))
        empty  = bar_width - filled
        bar    = "█" * filled + "░" * empty
        col    = emotion_colour(label)
        lines.append(f"  {col}{label:<12}{C.RESET}  {bar}  {p:.2f}")
    return "\n".join(lines) if lines else "  (no data)"


def print_final_verdict(
    file_name, safety, categories, tone, transcript, status,
    ground_truth, correct, t_whisper, t_tone, t_guard, total_s,
    file_index, total_files
):
    """
    Render the per-file FINAL VERDICT box in the terminal.

    Displays: safety verdict, threat categories, detected emotion,
    ground truth correctness, transcript snippet, and pipeline latency.
    """
    bg_color, icon, label_color = _verdict_style(safety)
    display_label = (safety or "UNKNOWN").upper()

    # Emotion summary from SpeechBrain output
    tone_label = tone.get("label_full", "")
    tone_p     = tone.get("top_p")
    emotion_str = f"{tone_label}  (confidence={tone_p:.2f})" if tone_label and tone_p else tone_label or "N/A"

    # Ground truth display with correctness indicator
    if ground_truth and correct == "Y":
        gt_str = C.GREEN + f"{ground_truth}   ✓ CORRECT" + C.RESET
    elif ground_truth and correct == "N":
        gt_str = C.RED + f"{ground_truth}   ✗ WRONG" + C.RESET
    else:
        gt_str = C.DIM + "N/A (manifest not loaded)" + C.RESET

    snippet    = shorten(transcript.replace("\n", " "), 70) if transcript else "(empty)"
    cat_str    = ", ".join(categories) if categories else "None"
    latency_str = (f"Whisper {t_whisper:.2f}s  |  Tone {t_tone:.2f}s  "
                   f"|  Guard {t_guard:.2f}s  |  Total {total_s:.2f}s")
    W = 88  # Inner box width

    def row(content: str) -> str:
        plain   = C.strip(content)
        padding = W - len(plain)
        return f"║  {content}{' ' * max(0, padding - 2)}║"

    top_title  = f" FINAL VERDICT — {file_name}  [{file_index} / {total_files}] "
    top_padded = top_title.ljust(W)

    print("\n")
    print(C.BOLD + label_color + "╔" + "═" * W + "╗" + C.RESET)
    print(C.BOLD + label_color + "║" + top_padded + "║" + C.RESET)
    print(C.BOLD + label_color + "╠" + "═" * W + "╣" + C.RESET)

    verdict_line = f"{bg_color}{C.BOLD}{C.WHITE}  {icon}  {display_label}  {C.RESET}"
    print(C.BOLD + label_color + "║" + C.RESET
          + f"  {verdict_line}"
          + " " * max(0, W - len(display_label) - 8)
          + C.BOLD + label_color + "║" + C.RESET)

    print(C.BOLD + label_color + "╠" + "─" * W + "╣" + C.RESET)

    ecol = emotion_colour(tone_label)
    details = [
        f"{C.BOLD}Threat Categories  {C.RESET}: {C.YELLOW}{cat_str}{C.RESET}",
        f"{C.BOLD}Detected Emotion   {C.RESET}: {ecol}{emotion_str}{C.RESET}",
        f"{C.BOLD}Ground Truth       {C.RESET}: {gt_str}",
        f"{C.BOLD}Transcript Snippet {C.RESET}: {C.DIM}{snippet}{C.RESET}",
        f"{C.BOLD}Pipeline Latency   {C.RESET}: {C.DIM}{latency_str}{C.RESET}",
        f"{C.BOLD}Status             {C.RESET}: {C.DIM}{status}{C.RESET}",
    ]
    for d in details:
        print(row(d))

    top3 = tone.get("top3", [])
    if top3:
        print(C.BOLD + label_color + "╠" + "─" * W + "╣" + C.RESET)
        print(row(f"{C.BOLD}Emotion Distribution:{C.RESET}"))
        for bar_line in _emotion_bar(top3).splitlines():
            print(row(bar_line))

    print(C.BOLD + label_color + "╚" + "═" * W + "╝" + C.RESET)


# ============================================================================
# SECTION H — TONE DISPLAY AND RESUME UTILITIES
# ============================================================================

def format_tone_card(tone: dict) -> str:
    """
    Format SpeechBrain output dict as a multi-line display string.
    Shown in the intermediate console block before the verdict box.
    Returns a fallback string if tone is empty (SpeechBrain failed).
    """
    if not tone:
        return C.DIM + "Tone: (not available / failed)" + C.RESET

    label_full  = tone.get("label_full", "")
    label_short = tone.get("label_short", "")
    top_p       = tone.get("top_p", None)
    top_p_str   = f"{top_p:.4f}" if isinstance(top_p, (int, float)) else "N/A"
    ecol        = emotion_colour(label_full)

    lines = [
        C.BOLD + "Dominant Emotion : " + C.RESET + ecol + f"{label_full} ({label_short})" + C.RESET,
        C.BOLD + "Confidence (top) : " + C.RESET + f"{top_p_str}",
        "",
        C.BOLD + "Top-3 Distribution:" + C.RESET,
    ]
    for item in (tone.get("top3", []) or []):
        lf    = item.get("label_full", item.get("label_short", ""))
        p     = item.get("p", None)
        p_str = f"{p:.4f}" if isinstance(p, (int, float)) else "N/A"
        col   = emotion_colour(lf)
        lines.append(f"  {col}{lf:<12}{C.RESET}  p={p_str}")

    return "\n".join(lines)


def print_saved_result(row: dict):
    """
    Re-display a previously saved CSV row (called during RESUME mode).
    Reconstructs the console display from stored CSV values without
    re-running the pipeline stages.
    """
    file_name = Path(str(row.get("file_path", ""))).name or "unknown"
    banner(f"[SAVED:{row.get('status','')}] {file_name}")
    print(f"Times: whisper={row.get('whisper_seconds','')}s | "
          f"tone={row.get('tone_seconds','')}s | "
          f"guard={row.get('guard_seconds','')}s | "
          f"total={row.get('total_seconds','')}s")
    block("Whisper Transcript (preview)", row.get("transcript_preview", "") or "(empty)")


# ============================================================================
# SECTION I — FILE DISCOVERY AND RESUME UTILITIES
# ============================================================================

def list_audio_files(root: Path, exts: set) -> list:
    """
    Recursively scan root for audio files matching the given extensions.
    Returns a sorted list of Paths for reproducible processing order.
    """
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_processed(csv_path: Path) -> set:
    """
    Read an existing output CSV and return the set of already-processed file paths.
    Used by RESUME mode to skip files completed in a prior run.
    """
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames and "file_path" in r.fieldnames:
            for row in r:
                fp = (row.get("file_path") or "").strip()
                if fp:
                    done.add(fp)
    return done


def load_existing_rows(csv_path: Path) -> dict:
    """Load existing CSV rows as a dict keyed by file_path (for RESUME mode)."""
    rows = {}
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fp = (row.get("file_path") or "").strip()
            if fp:
                rows[fp] = row
    return rows


def retry_call(fn, name="call"):
    """
    Execute fn() with automatic retry on transient failure.

    Attempts up to RETRIES times with RETRY_SLEEP_S wait between attempts.
    Raises RuntimeError after all attempts are exhausted.

    Args:
        fn   : Zero-argument callable to execute.
        name : Human-readable label for log messages (e.g. "Whisper", "ToneAPI").
    """
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < RETRIES:
                print(C.YELLOW + f"  [{name}] attempt {attempt}/{RETRIES} failed: {repr(e)}" + C.RESET)
                time.sleep(RETRY_SLEEP_S)
            else:
                raise RuntimeError(f"{name} failed after {RETRIES} attempts: {repr(last)}")


# ============================================================================
# SECTION J — API FUNCTIONS (one per pipeline stage)
# ============================================================================

def whisper_transcribe(audio_path: Path) -> str:
    """
    Stage 1 — ASR: Send audio to Whisper Large-v3 and return transcript text.

    Uses an OpenAI-compatible client pointing at the vLLM Whisper endpoint.
    temperature=0.0 ensures deterministic, reproducible transcription.

    Args:
        audio_path: Path to a WAV file.

    Returns:
        Plain text transcript string.
    """
    client = OpenAI(
        api_key="EMPTY",   # vLLM does not require a real API key
        base_url=f"http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1"
    )
    with open(audio_path, "rb") as f:
        out = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language=LANGUAGE,
            response_format="text",   # Return plain string, not JSON wrapper
            temperature=TEMPERATURE,
        )
    return str(out).strip()


def tone_from_api(audio_path: Path) -> dict:
    """
    Stage 2 — SER: Send audio to SpeechBrain emotion API and return result dict.

    Sends the raw WAV as multipart form-data. The API returns:
        {
            "label_short": "Ang",
            "label_full":  "Angry",
            "top_p":       0.87,
            "top3": [
                {"label_full": "Angry",   "p": 0.87},
                {"label_full": "Neutral", "p": 0.08},
                {"label_full": "Sad",     "p": 0.05}
            ]
        }

    Args:
        audio_path: Path to a WAV file.

    Returns:
        Parsed JSON response as a Python dict.
    """
    headers = {"X-API-Key": TONE_API_KEY} if TONE_API_KEY else {}
    with open(audio_path, "rb") as f:
        r = requests.post(
            TONE_API_URL,
            headers=headers,
            files={"file": (audio_path.name, f)},
            timeout=TONE_TIMEOUT_S,
        )
    r.raise_for_status()
    return r.json()


def clean_transcript_text(transcript: str) -> str:
    """
    Normalise Whisper output before passing to the guard.

    In rare cases Whisper may return a JSON-wrapped string:
        '{"text": "Hello world"}'
    This function extracts the text field in that case.
    Plain strings are returned unchanged.
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
    Stage 2→3 Bridge — Construct the text string sent to Qwen3Guard.

    System A (FUSION_ENABLED=False):
        Returns plain cleaned transcript only.

    System B (FUSION_ENABLED=True):
        Prepends a natural-language emotion context sentence.
        Example:
            [Audio context: The speaker sounds very angry
             (Angry=0.87, Neutral=0.08, Sad=0.05).]

            Transcript: I know where you live...

    Confidence qualifier mapping:
        top_p >= 0.80  → "very"
        top_p >= 0.55  → "noticeably"
        top_p < 0.55   → "somewhat"

    Falls back to plain transcript if FUSION_ENABLED=False or tone is empty.
    """
    clean_text = clean_transcript_text(transcript)

    if not FUSION_ENABLED or not tone:
        return clean_text  # System A baseline path

    label_full = tone.get("label_full", "")
    top_p      = tone.get("top_p", None)
    top3       = tone.get("top3", []) or []

    # Map confidence score to intensity qualifier
    if isinstance(top_p, (int, float)):
        confidence = "very" if top_p >= 0.80 else "noticeably" if top_p >= 0.55 else "somewhat"
    else:
        confidence = "somewhat"

    if label_full:
        emotion_sentence = f"The speaker sounds {confidence} {label_full.lower()}"
        if top3:
            score_parts = [
                f"{item.get('label_full', item.get('label_short',''))}={item.get('p', 0):.2f}"
                for item in top3[:3]
            ]
            emotion_sentence += f" ({', '.join(score_parts)})"
        emotion_sentence += "."
    else:
        emotion_sentence = "Speaker tone could not be determined."

    return f"[Audio context: {emotion_sentence}]\n\nTranscript: {clean_text}"


def qwen_guard(transcript: str, tone: dict = None):
    """
    Stage 3 — LLM Safety Classification: Send fused input to Qwen3Guard.

    Builds the fused input, sends as a chat completion request,
    and parses the structured "Safety: / Categories:" response.

    temperature=0 ensures deterministic output for reproducibility.
    max_tokens=128 is sufficient for the structured format.

    Args:
        transcript : Raw Whisper transcript text.
        tone       : SpeechBrain output dict (or None for baseline mode).

    Returns:
        tuple: (full_json, raw_text, safety_label, categories, fused_text)
    """
    fused_text = build_fused_input(transcript, tone or {})

    headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
    payload = {
        "model"      : QWENGUARD_MODEL,
        "messages"   : [{"role": "user", "content": fused_text}],
        "temperature": 0,
        "max_tokens" : MAX_TOKENS_GUARD,
    }

    r = requests.post(QWENGUARD_URL, headers=headers, json=payload, timeout=GUARD_TIMEOUT_S)
    r.raise_for_status()

    data = r.json()
    raw  = data["choices"][0]["message"]["content"].strip()
    safety, categories = parse_guard(raw)
    safety = normalize_safety_label(safety, strict_mode=STRICT_MODE)

    return data, raw, safety, categories, fused_text


# ============================================================================
# SECTION K — SESSION SUMMARY TABLE
# ============================================================================

def print_session_summary(results: list):
    """
    Print a compact summary table for all files processed in this session.
    Columns: # | File | Safety | GT | OK? | Emotion | Categories | Time(s)
    Footer shows aggregate counts and live accuracy.
    """
    print("\n\n" + C.BOLD + C.CYAN + "═" * 100 + C.RESET)
    print(C.BOLD + C.WHITE + " SESSION SUMMARY — All Processed Files".center(100) + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 100 + C.RESET)

    col_w = [4, 28, 14, 10, 5, 12, 22, 8]
    header = (f"{'#':<{col_w[0]}} {'File':<{col_w[1]}} {'Safety':<{col_w[2]}} "
              f"{'GT':<{col_w[3]}} {'OK?':<{col_w[4]}} {'Emotion':<{col_w[5]}} "
              f"{'Categories':<{col_w[6]}} {'Time(s)':<{col_w[7]}}")
    print(C.BOLD + C.DIM + header + C.RESET)
    print(C.DIM + "─" * 100 + C.RESET)

    safe_c = unsafe_c = warn_c = fail_c = correct_c = wrong_c = 0

    for i, r in enumerate(results, 1):
        safety  = r.get("safety",  "") or "N/A"
        gt      = r.get("ground_truth", "") or "—"
        correct = r.get("correct", "")
        file_n  = r.get("file_name", "")[:col_w[1]]
        emotion = r.get("emotion", "N/A")
        cats    = r.get("categories", "None")[:col_w[6]]
        total_s = r.get("total_s", 0.0)

        s_upper = safety.upper()
        if s_upper == "UNSAFE":
            safety_col = C.RED + C.BOLD + f"{'🚨 ' + safety:<{col_w[2]}}" + C.RESET; unsafe_c += 1
        elif s_upper == "SAFE":
            safety_col = C.GREEN + f"{'✅ ' + safety:<{col_w[2]}}" + C.RESET; safe_c += 1
        elif s_upper == "CONTROVERSIAL":
            safety_col = C.YELLOW + f"{'⚠  ' + safety:<{col_w[2]}}" + C.RESET; warn_c += 1
        else:
            safety_col = C.DIM + f"{'❓ ' + safety:<{col_w[2]}}" + C.RESET; fail_c += 1

        if correct == "Y":
            correct_col = C.GREEN + C.BOLD + "✓" + C.RESET; correct_c += 1
        elif correct == "N":
            correct_col = C.RED + C.BOLD + "✗" + C.RESET; wrong_c += 1
        else:
            correct_col = C.DIM + "—" + C.RESET

        ecol        = emotion_colour(emotion)
        emotion_col = ecol + emotion[:col_w[5]] + C.RESET

        print(f"{str(i):<{col_w[0]}} {file_n:<{col_w[1]}} "
              + safety_col + " " + f"{gt:<{col_w[3]}} "
              + correct_col + "    " + emotion_col
              + " " * max(0, col_w[5] - len(emotion)) + " "
              + f"{cats:<{col_w[6]}} {total_s:<{col_w[7]}.2f}")

    print(C.DIM + "─" * 100 + C.RESET)
    total_eval = correct_c + wrong_c
    acc_str = (f"  Accuracy: {correct_c}/{total_eval} = {correct_c/total_eval*100:.1f}%"
               if total_eval else "")
    print(C.GREEN  + f"  ✅ Safe: {safe_c}   " + C.RESET +
          C.RED    + f"🚨 Unsafe: {unsafe_c}   " + C.RESET +
          C.YELLOW + f"⚠  Controversial: {warn_c}   " + C.RESET +
          C.DIM    + f"❌ Failed: {fail_c}   " + C.RESET +
          C.BOLD   + acc_str + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 100 + C.RESET)


# ============================================================================
# SECTION L — MAIN BATCH EXECUTION
# ============================================================================
# Top-level control flow:
#   1. Print run configuration header
#   2. Discover audio files in INPUT_DIR
#   3. Iterate files — run all 3 pipeline stages per file
#   4. Write CSV row and JSON bundle after each file (flush immediately)
#   5. Print session summary and performance metrics at the end
# ============================================================================

mode_label = "System B — Full Fusion" if FUSION_ENABLED else "System A — Baseline (text-only)"
banner(f"100-Sample Evaluation  |  {mode_label}")
print(f"{C.BOLD}Dataset  :{C.RESET} 100 samples — 50 Unsafe · 50 Safe")
print(f"{C.BOLD}Input    :{C.RESET} {INPUT_DIR}")
print(f"{C.BOLD}Output   :{C.RESET} {OUT_DIR}")
print(f"{C.BOLD}Whisper  :{C.RESET} http://{WHISPER_SERVER_IP}:{WHISPER_PORT}/v1")
print(f"{C.BOLD}Tone     :{C.RESET} {TONE_API_URL}")
print(f"{C.BOLD}Guard    :{C.RESET} {QWENGUARD_URL}  (model={QWENGUARD_MODEL})")
print(f"{C.BOLD}Policy   :{C.RESET} STRICT_MODE={STRICT_MODE}  FUSION_ENABLED={FUSION_ENABLED}")

files = list_audio_files(INPUT_DIR, EXTS)
print(f"\n{C.BOLD}Found files: {len(files)}{C.RESET}")

processed  = load_processed(CSV_PATH) if RESUME else set()
saved_rows = load_existing_rows(CSV_PATH) if RESUME else {}
if RESUME:
    print(f"{C.YELLOW}Resume: {len(processed)} file(s) already in CSV{C.RESET}")

# CSV column schema — 23 columns written for every processed file
fieldnames = [
    "timestamp", "file_path", "file_size_bytes", "language",
    "whisper_seconds", "tone_seconds", "guard_seconds", "total_seconds",
    "transcript_preview", "transcript_len",
    "tone_label_short", "tone_label_full", "tone_top_p", "tone_top3_json",
    "clean_transcript_preview",
    "safety", "categories",
    "guard_raw_preview", "guard_raw_full",
    "status", "error",
    "ground_truth", "correct",
]

csv_exists      = CSV_PATH.exists()
session_results = []

with open(CSV_PATH, "a", encoding="utf-8", newline="") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    # Run-level counters
    ok = warn = fail = saved_displayed = 0
    tp = tn = fp = fn = 0   # Confusion matrix counters

    for file_index, p in enumerate(tqdm(files, desc="Processing"), start=1):

        # Resume: skip already-processed files
        if RESUME and str(p) in processed:
            if SHOW_SAVED_RESULTS_WHEN_RESUME:
                row = saved_rows.get(str(p))
                if row:
                    print_saved_result(row)
                    saved_displayed += 1
            continue

        # Per-file initialisation
        ts         = datetime.now().isoformat(timespec="seconds")
        size_bytes = p.stat().st_size
        status     = "OK"
        err        = ""

        transcript = ""; clean_text = ""; tone = {}
        safety = ""; categories = []; guard_raw = ""; guard_json = None
        t_whisper = t_tone = t_guard = 0.0
        t0_total  = time.time()

        # ── Stage 1: Whisper ASR ──────────────────────────────────────────────
        try:
            t0         = time.time()
            transcript = retry_call(lambda: whisper_transcribe(p), name="Whisper")
            t_whisper  = time.time() - t0
            print(C.DIM + f"\n  [{p.name}] Whisper {t_whisper:.2f}s — {len(transcript)} chars" + C.RESET)
        except Exception as e:
            status = "FAIL_WHISPER"; err = str(e)
            print(C.RED + f"\n  [{p.name}] Whisper FAILED: {err}" + C.RESET)

        # ── Stage 2: SpeechBrain Emotion Recognition ──────────────────────────
        # Non-fatal: pipeline continues with empty tone dict on failure
        if transcript.strip():
            try:
                t0     = time.time()
                tone   = retry_call(lambda: tone_from_api(p), name="ToneAPI")
                t_tone = time.time() - t0
                print(C.DIM + f"  [{p.name}] Tone {t_tone:.2f}s — emotion={tone.get('label_full','?')}" + C.RESET)
            except Exception as e:
                status = "WARN_TONE_FAILED" if status == "OK" else status
                err = str(e); tone = {}
                print(C.YELLOW + f"  [{p.name}] Tone WARN: {err}" + C.RESET)

        # ── Stage 3: Qwen3Guard Safety Classification ─────────────────────────
        if transcript.strip():
            try:
                t0 = time.time()
                guard_json, guard_raw, safety, categories, clean_text = retry_call(
                    lambda: qwen_guard(transcript, tone), name="QwenGuard"
                )
                t_guard = time.time() - t0
                print(C.DIM + f"  [{p.name}] Guard {t_guard:.2f}s — safety={safety}" + C.RESET)
            except Exception as e:
                status = "FAIL_GUARD"; err = str(e)
                print(C.RED + f"  [{p.name}] Guard FAILED: {err}" + C.RESET)
        else:
            if status == "OK":
                status = "SKIP_EMPTY_TRANSCRIPT"
                err    = "Whisper returned empty transcript."

        total_s = time.time() - t0_total

        # ── Ground truth comparison ───────────────────────────────────────────
        ground_truth = GROUND_TRUTH.get(p.name, "")
        if ground_truth and safety:
            correct = "Y" if safety.lower() == ground_truth.lower() else "N"
        else:
            correct = ""

        if correct == "Y":
            print(C.GREEN + f"  ✓ CORRECT  (GT={ground_truth}, Pred={safety})" + C.RESET)
        elif correct == "N":
            print(C.RED   + f"  ✗ WRONG    (GT={ground_truth}, Pred={safety})" + C.RESET)

        # Update confusion matrix counters
        if ground_truth and safety:
            g, pred = ground_truth.lower(), safety.lower()
            if   g == "unsafe" and pred == "unsafe": tp += 1
            elif g == "safe"   and pred == "safe":   tn += 1
            elif g == "safe"   and pred == "unsafe": fp += 1
            elif g == "unsafe" and pred == "safe":   fn += 1

        # Flatten tone fields for CSV
        tone_label_short = str(tone.get("label_short") or "")
        tone_label_full  = str(tone.get("label_full")  or "")
        tone_top_p       = tone.get("top_p")
        tone_top3_json   = json.dumps(tone.get("top3", []), ensure_ascii=False) if tone else ""

        # Console output blocks
        block("Raw Whisper Transcript (preview)",
              shorten(transcript, TRANSCRIPT_PREVIEW_CHARS) or C.DIM + "(empty)" + C.RESET)
        block("Fused Input Sent to Guard (preview)",
              shorten(clean_text, TRANSCRIPT_PREVIEW_CHARS) or C.DIM + "(empty)" + C.RESET)
        block("SpeechBrain Tone Analysis", format_tone_card(tone))
        block("Qwen3Guard Raw Output",
              shorten(guard_raw, GUARD_PREVIEW_CHARS) or C.DIM + "(empty)" + C.RESET)
        if err:
            block(C.YELLOW + "⚠  Pipeline Note" + C.RESET, C.YELLOW + err + C.RESET)

        print_final_verdict(
            p.name, safety, categories, tone, transcript, status,
            ground_truth, correct, t_whisper, t_tone, t_guard, total_s,
            file_index, len(files),
        )

        # Save per-file JSON bundle (complete pipeline inputs + outputs)
        if SAVE_JSON:
            bundle = {
                "timestamp"        : ts,
                "audio_path"       : str(p),
                "file_size_bytes"  : size_bytes,
                "language"         : LANGUAGE,
                "fusion_enabled"   : FUSION_ENABLED,
                "latency_seconds"  : {"whisper": round(t_whisper,4), "tone": round(t_tone,4),
                                      "qwenguard": round(t_guard,4), "total": round(total_s,4)},
                "whisper_transcript_raw": transcript,
                "guard_clean_text"      : clean_text,
                "tone_summary"          : tone if tone else None,
                "guard_parsed"          : {"safety": safety, "categories": categories},
                "guard_raw"             : guard_raw,
                "guard_full_json"       : guard_json,
                "status"                : status,
                "error"                 : err,
                "ground_truth"          : ground_truth,
                "correct"               : correct,
            }
            out_json = JSON_DIR / (p.stem.replace(" ", "_")[:60] + f"__{abs(hash(str(p))) % 10**8}.json")
            with open(out_json, "w", encoding="utf-8") as jf:
                json.dump(bundle, jf, ensure_ascii=False, indent=2)

        # Write CSV row — flushed immediately to prevent data loss on crash
        writer.writerow({
            "timestamp"               : ts,
            "file_path"               : str(p),
            "file_size_bytes"         : size_bytes,
            "language"                : LANGUAGE,
            "whisper_seconds"         : round(t_whisper, 2) if t_whisper else "",
            "tone_seconds"            : round(t_tone, 2)    if t_tone    else "",
            "guard_seconds"           : round(t_guard, 2)   if t_guard   else "",
            "total_seconds"           : round(total_s, 2),
            "transcript_preview"      : shorten(transcript, TRANSCRIPT_PREVIEW_CHARS),
            "transcript_len"          : len(transcript or ""),
            "tone_label_short"        : tone_label_short,
            "tone_label_full"         : tone_label_full,
            "tone_top_p"              : tone_top_p if tone_top_p is not None else "",
            "tone_top3_json"          : tone_top3_json,
            "clean_transcript_preview": shorten(clean_text, TRANSCRIPT_PREVIEW_CHARS),
            "safety"                  : safety or "",
            "categories"              : ", ".join(categories) if categories else "",
            "guard_raw_preview"       : shorten(guard_raw, GUARD_PREVIEW_CHARS),
            "guard_raw_full"          : guard_raw,
            "status"                  : status,
            "error"                   : err,
            "ground_truth"            : ground_truth,
            "correct"                 : correct,
        })
        fcsv.flush()   # Flush after every row — preserves partial results if run is interrupted

        session_results.append({
            "file_name"   : p.name,
            "safety"      : safety or "N/A",
            "ground_truth": ground_truth,
            "correct"     : correct,
            "emotion"     : tone_label_full or "N/A",
            "categories"  : ", ".join(categories) if categories else "None",
            "total_s"     : total_s,
            "status"      : status,
        })

        if status == "OK":    ok += 1
        elif status.startswith("WARN"): warn += 1
        else:                           fail += 1

        if SLEEP_BETWEEN_FILES_S > 0:
            time.sleep(SLEEP_BETWEEN_FILES_S)


# ============================================================================
# SECTION M — RUN COMPLETE: SUMMARY AND PERFORMANCE METRICS
# ============================================================================

if session_results:
    print_session_summary(session_results)

banner(f"Run Complete — {mode_label}")
print(f"{C.GREEN}  ✅ OK          : {ok}{C.RESET}")
print(f"{C.YELLOW}  ⚠  Warnings    : {warn}{C.RESET}")
print(f"{C.RED}  ❌ Failures    : {fail}{C.RESET}")
print(f"{C.DIM}  ↩  Resumed     : {saved_displayed}{C.RESET}")
print(f"{C.BOLD}  📄 CSV saved at: {CSV_PATH}{C.RESET}")

# Compute and print performance metrics (requires ground truth labels)
total_eval = tp + tn + fp + fn
if total_eval > 0:
    acc    = (tp + tn) / total_eval * 100
    prec   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_s = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0.0
    rec_s  = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
    macro  = (f1 + f1_s) / 2

    print("\n" + C.BOLD + C.CYAN + "═" * 65 + C.RESET)
    print(C.BOLD + C.WHITE + "  PERFORMANCE EVALUATION — 100-Sample Dataset".center(65) + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 65 + C.RESET)
    print(f"  Mode       : {mode_label}")
    print(f"  Evaluated  : {total_eval}  |  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(C.CYAN + "  " + "─" * 63 + C.RESET)
    print(f"  {C.BOLD}Accuracy          : {acc:.1f}%{C.RESET}")
    print(f"  {C.BOLD}Precision (Unsafe): {prec:.1f}%{C.RESET}")
    print(f"  {C.BOLD}Recall    (Unsafe): {rec:.1f}%{C.RESET}")
    print(f"  {C.BOLD}F1        (Unsafe): {f1:.1f}%{C.RESET}")
    print(f"  {C.BOLD}Precision (Safe)  : {prec_s:.1f}%{C.RESET}")
    print(f"  {C.BOLD}Recall    (Safe)  : {rec_s:.1f}%{C.RESET}")
    print(f"  {C.BOLD}F1        (Safe)  : {f1_s:.1f}%{C.RESET}")
    print(C.CYAN + "  " + "─" * 63 + C.RESET)
    print(C.BOLD + f"  Macro-avg F1      : {macro:.1f}%" + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 65 + C.RESET)

    if fn > 0:
        print(C.RED + C.BOLD + f"\n  ⚠  WARNING: {fn} missed threat(s) (FN). Review these cases." + C.RESET)
    if fp > 0:
        print(C.YELLOW + f"  ℹ  {fp} false alarm(s) (FP). Safe content flagged as Unsafe." + C.RESET)
else:
    print(C.YELLOW + "\n  No ground truth — evaluation skipped." + C.RESET)
    print(C.YELLOW + "  Set MANIFEST_PATH to dataset_100_samples.json to enable." + C.RESET)
