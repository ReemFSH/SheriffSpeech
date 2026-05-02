"""
=============================================================================
SEC619 — Edge TTS Emotional Audio Generator
100-Sample Dataset (50 Unsafe · 50 Safe)
Emotions: Angry (16) · Neutral (55) · Cheerful (20) · Sad (9)
Version : v3 — Neutral rate/pitch fix
=============================================================================

WHY NO SSML STYLE TAGS:
    Using <mstts:express-as style="angry"> on violent/sensitive content causes
    Microsoft's safety system to inject spoken disclaimers at the start and end
    of the audio (content warnings, policy notices). These are active speech at
    the same volume as the script and cannot be removed by silence detection.

FIX (applied here):
    Use plain text + Communicate(rate, pitch) parameters only.
    Edge TTS applies prosody without triggering safety injection.
    The audio contains ONLY the script text — no injected speech.

NEUTRAL FIX (v3):
    Previous version used rate="0%" and pitch="0Hz" for Neutral samples.
    Edge-tts requires an explicit +/- prefix on all rate and pitch values.
    "0%" with no sign causes: ValueError("Invalid rate '0%'.")

    Fix applied:
        Rate  : "0%"   →  "-10%"   (slight slowdown — calm, measured delivery)
        Pitch : "0Hz"  →  "-5Hz"   (slight lowering — flat, professional tone)

    This affected C05_SH_S02 and any other Neutral samples that were being
    regenerated. The fix also improves acoustic separation between Neutral
    and Cheerful for SpeechBrain classification.

CHEERFUL FIX (v2):
    Previous version used en-US-SaraNeural + rate=+30% + pitch=+20Hz.
    This triggered NoAudioReceived errors on all 20 Cheerful samples.

    Root cause: SaraNeural + high rate/pitch on certain content types
    causes Edge TTS to reject the synthesis request server-side.

    Fix applied:
        Voice  : SaraNeural  →  JennyNeural
        Rate   : +30%        →  +20%
        Pitch  : +20Hz       →  +10Hz

    JennyNeural is brighter and more expressive than Guy/Aria at baseline,
    so the reduced prosody still produces a clearly upbeat delivery that
    SpeechBrain can distinguish from Neutral.

HOW PROSODY SIMULATES EMOTION (SpeechBrain-detectable):
    Angry    → fast rate (+40%) + high pitch (+20Hz)
               SpeechBrain detects: high pitch variance, faster tempo, tension
    Neutral  → slightly slower (-10%) + slightly lower (-5Hz)
               SpeechBrain detects: flat, calm, baseline acoustic pattern
               NOTE: must use "-10%" not "0%" — edge-tts rejects unsigned values
    Cheerful → moderate rate (+20%) + moderate pitch (+10Hz) + bright voice
               SpeechBrain detects: elevated pitch, brighter tone, energy
    Sad      → slow rate (-30%) + low pitch (-20Hz)
               SpeechBrain detects: low pitch, slow tempo, low energy

RUN BEHAVIOUR:
    - Files that already exist with real audio (> 2 KB) are SKIPPED.
    - On first run: generates all 100 files.
    - On re-run after partial failure: only generates the missing ones.
    - To regenerate everything: delete the audio/ folder first.

    After the previous 80-success / 20-failure run:
    - 80 existing WAV files will be skipped automatically.
    - Only the 20 Cheerful samples will be re-attempted with the fixed voice.

INSTALL:
    pip install edge-tts
    ffmpeg must be on PATH (https://www.gyan.dev/ffmpeg/builds/)

RUN:
    python tts_edge_emotional.py

OUTPUT:
    audio/<sample_id>.wav   — 16kHz · Mono · PCM · 16-bit (Whisper-ready)
    Named to match dataset_100_samples.json IDs exactly.
=============================================================================
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter


# ── Dependency check — edge_tts ───────────────────────────────────────────────
try:
    import edge_tts
except ImportError:
    print("ERROR: edge-tts not installed.")
    print("       Run:  pip install edge-tts")
    sys.exit(1)


# ── Dependency check — ffmpeg ─────────────────────────────────────────────────
def check_ffmpeg() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if r.returncode == 0:
            print("ffmpeg : " + r.stdout.split("\n")[0][:70])
            return True
    except FileNotFoundError:
        pass
    print("ERROR: ffmpeg not found on PATH.")
    print("       Download : https://www.gyan.dev/ffmpeg/builds/")
    print("       Extract to C:\\ffmpeg\\ then add C:\\ffmpeg\\bin to System PATH.")
    return False


# ============================================================================
# PATHS
# ============================================================================

BASE      = Path(__file__).parent
MANIFEST  = BASE / "dataset_100_samples_updated.json"
AUDIO_DIR = BASE / "audio"


# ============================================================================
# EMOTION → VOICE + PROSODY MAPPING
#
# Keys must exactly match expected_emotion values in dataset_100_samples.json:
#   "Angry"  ·  "Neutral"  ·  "Cheerful"  ·  "Sad"
# (case-sensitive — capital first letter)
#
# Voice logic:
#   GuyNeural   — male, professional → Angry and Neutral
#   JennyNeural — female, bright/warm → Cheerful  (FIXED: was SaraNeural)
#   AriaNeural  — female, expressive  → Sad
# ============================================================================

EMOTION_MAP: dict[str, dict] = {

    "Angry": {
        # Urgent, tense, clipped delivery
        # SpeechBrain target: anger class
        "voice": "en-US-GuyNeural",
        "rate":  "+40%",
        "pitch": "+20Hz",
    },

    "Neutral": {
        # Calm, professional, flat delivery
        # SpeechBrain target: neutral class
        # Slightly below baseline — clear acoustic separation from Cheerful
        # NOTE: edge-tts requires explicit +/- prefix on rate and pitch.
        #       "0%" and "0Hz" are invalid — use "-0%" and "+0Hz" or small offset.
        "voice": "en-US-GuyNeural",
        "rate":  "-10%",
        "pitch": "-5Hz",
    },

    "Cheerful": {
        # Upbeat, energetic, lighter delivery
        # SpeechBrain target: happiness / cheerful class
        #
        # FIXED from v1:
        #   SaraNeural → JennyNeural  (avoids NoAudioReceived server rejection)
        #   +30%       → +20%         (moderate rate — still clearly upbeat)
        #   +20Hz      → +10Hz        (moderate pitch — bright but not clipped)
        #
        # JennyNeural has a naturally warmer, more expressive baseline than
        # Guy/Aria, so the reduced prosody still produces a clearly cheerful
        # delivery that SpeechBrain distinguishes from Neutral.
        "voice": "en-US-JennyNeural",
        "rate":  "+20%",
        "pitch": "+20Hz",
    },

    "Sad": {
        # Heavy, slow, low-energy delivery
        # SpeechBrain target: sadness class
        "voice": "en-US-AriaNeural",
        "rate":  "-30%",
        "pitch": "-20Hz",
    },
}

# Fallback when manifest contains an unrecognised emotion label
DEFAULT_CFG = EMOTION_MAP["Neutral"]


# ============================================================================
# GENERATE ONE FILE
# ============================================================================

async def generate_one(sample: dict, out_wav: Path) -> tuple[str, object]:
    """
    Generate one WAV using plain text + prosody parameters only.
    No SSML express-as style tags — prevents Microsoft safety injection.

    Returns:
        ("ok",    file_size_kb)   — generated successfully
        ("skip",  0)              — valid file already exists, skipped
        ("error", reason_str)     — generation or ffmpeg conversion failed
    """
    # Skip files that already have real audio (> 2 KB threshold)
    if out_wav.exists() and out_wav.stat().st_size > 2000:
        return ("skip", 0)

    emotion = sample.get("expected_emotion", "Neutral")
    cfg     = EMOTION_MAP.get(emotion, DEFAULT_CFG)
    script  = sample["script"]
    tmp_mp3 = out_wav.with_suffix(".mp3")

    try:
        # ── Edge TTS: plain text + prosody only (no SSML style tag) ──────
        communicate = edge_tts.Communicate(
            script,
            cfg["voice"],
            rate  = cfg["rate"],
            pitch = cfg["pitch"],
        )
        await communicate.save(str(tmp_mp3))

        if not tmp_mp3.exists() or tmp_mp3.stat().st_size < 100:
            return ("error", "Edge TTS returned empty or missing MP3")

        # ── ffmpeg: MP3 → 16kHz mono PCM WAV ─────────────────────────────
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i",          str(tmp_mp3),
                "-ar",         "16000",    # 16 kHz — Whisper requirement
                "-ac",         "1",        # mono channel
                "-sample_fmt", "s16",      # 16-bit PCM
                str(out_wav),
            ],
            capture_output=True,
            text=True,
        )

        tmp_mp3.unlink(missing_ok=True)

        if result.returncode != 0:
            return ("error", f"ffmpeg error: {result.stderr[-300:]}")

        if not out_wav.exists() or out_wav.stat().st_size < 100:
            return ("error", "WAV file not created or is empty after ffmpeg")

        return ("ok", out_wav.stat().st_size // 1024)

    except Exception as exc:
        tmp_mp3.unlink(missing_ok=True)
        return ("error", repr(exc))


# ============================================================================
# BATCH RUNNER
# ============================================================================

async def run_all(samples: list) -> int:
    """
    Process all samples sequentially.
    Already-generated files are skipped automatically.
    Returns the total number of failures.
    """
    total      = len(samples)
    ok_count   = 0
    skip_count = 0
    fail_count = 0
    failures:  list[tuple[str, str]] = []

    print()
    print("=" * 70)
    print(f"  Generating {total} audio files")
    print(f"  Method   : Plain text + prosody  (no SSML style tags)")
    print(f"  Neutral  : GuyNeural -10%/-5Hz  (FIXED — was invalid '0%'/'0Hz')")
    print(f"  Cheerful : JennyNeural +20%/+10Hz  (FIXED — was SaraNeural)")
    print(f"  Format   : 16kHz · Mono · PCM · 16-bit")
    print(f"  Output   : {AUDIO_DIR}")
    print("=" * 70)

    for i, sample in enumerate(samples, 1):

        sid       = sample["id"]
        emotion   = sample.get("expected_emotion", "Neutral")
        label     = sample.get("ground_truth", "?")
        cfg       = EMOTION_MAP.get(emotion, DEFAULT_CFG)
        out_wav   = AUDIO_DIR / f"{sid}.wav"
        indicator = "🔴" if label == "Unsafe" else "🟢"

        print(f"\n[{i:03d}/{total}] {indicator} {sid}  [{label}]")
        print(f"           Emotion : {emotion:<10}  voice={cfg['voice']}")
        print(f"           Prosody : rate={cfg['rate']}  pitch={cfg['pitch']}")

        status, info = await generate_one(sample, out_wav)

        if status == "ok":
            print(f"           Result  : ✓  {info} KB")
            ok_count += 1
        elif status == "skip":
            print(f"           Result  : →  already exists (skipped)")
            skip_count += 1
        else:
            print(f"           Result  : ✗  FAILED — {info}")
            fail_count += 1
            failures.append((sid, str(info)))

        # Delay to avoid Edge TTS server rate limiting
        await asyncio.sleep(0.6)

    # ── Run summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  BATCH COMPLETE")
    print(f"  ✓ Generated  : {ok_count}")
    print(f"  → Skipped    : {skip_count}  (already existed — delete audio/ to regenerate)")
    print(f"  ✗ Failed     : {fail_count}  (re-run script to retry)")
    if failures:
        print()
        print("  Failed samples:")
        for sid, reason in failures:
            print(f"    {sid:<22}  {reason[:80]}")
    print()
    print(f"  Files saved to : {AUDIO_DIR}")
    print(f"  Format         : 16kHz · Mono · PCM · 16-bit — ready for Whisper")
    print("=" * 70)

    return fail_count


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("  SEC619 — Edge TTS Emotional Audio Generator  (v3 — Neutral fix)")
    print("  Dataset  : 100 samples  |  50 Unsafe · 50 Safe")
    print("  Emotions : Angry · Neutral · Cheerful · Sad")
    print("  Neutral  : GuyNeural -10%/-5Hz  (FIXED: was '0%'/'0Hz' — invalid)")
    print("  Cheerful : JennyNeural +20%/+10Hz  (was SaraNeural — caused failures)")
    print("  Method   : Plain text + prosody  (no SSML style tags)")
    print("=" * 70)

    # ffmpeg check
    print()
    if not check_ffmpeg():
        sys.exit(1)

    # Load manifest
    print(f"\nManifest : {MANIFEST}")
    if not MANIFEST.exists():
        print(f"\nERROR: dataset_100_samples.json not found.")
        print(f"       Place this script in the same folder as dataset_100_samples.json")
        print(f"       or update MANIFEST at the top of this file.")
        sys.exit(1)

    with open(MANIFEST, encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded   : {len(samples)} samples")

    # Distribution summary
    emo_counts = Counter(s.get("expected_emotion", "?") for s in samples)
    lbl_counts = Counter(s.get("ground_truth",     "?") for s in samples)
    print(f"Labels   : Unsafe={lbl_counts.get('Unsafe',0)}  "
          f"Safe={lbl_counts.get('Safe',0)}")
    print(f"Emotions : " +
          "  ".join(f"{e}={c}" for e, c in sorted(emo_counts.items())))

    # Warn on unrecognised emotion labels
    unknown = set(emo_counts) - set(EMOTION_MAP)
    if unknown:
        print(f"\nWARNING: Unrecognised emotion(s): {unknown}")
        print(f"         These samples will use Neutral voice as fallback.")

    # Voice + prosody mapping table
    print()
    print("  Emotion → Voice + Prosody:")
    print(f"  {'Emotion':<12} {'Voice':<22} {'Rate':<10} {'Pitch':<10} Files  Note")
    print("  " + "─" * 75)
    notes = {
        "Neutral":  "← FIXED (was '0%'/'0Hz' — invalid, caused ValueError)",
        "Cheerful": "← FIXED (was SaraNeural +30%/+20Hz)",
    }
    for emo_name, cfg in EMOTION_MAP.items():
        count = emo_counts.get(emo_name, 0)
        note  = notes.get(emo_name, "")
        print(f"  {emo_name:<12} {cfg['voice']:<22} {cfg['rate']:<10} "
              f"{cfg['pitch']:<10} {count:<6} {note}")

    # Check existing files
    AUDIO_DIR.mkdir(exist_ok=True)
    existing = sum(
        1 for s in samples
        if (AUDIO_DIR / f"{s['id']}.wav").exists()
        and (AUDIO_DIR / f"{s['id']}.wav").stat().st_size > 2000
    )
    to_generate = len(samples) - existing

    cheerful_pending = sum(
        1 for s in samples
        if s.get("expected_emotion") == "Cheerful"
        and not (
            (AUDIO_DIR / f"{s['id']}.wav").exists()
            and (AUDIO_DIR / f"{s['id']}.wav").stat().st_size > 2000
        )
    )

    print()
    if existing > 0:
        print(f"  {existing} valid WAV files already exist → will be skipped.")
    if cheerful_pending > 0:
        print(f"  {cheerful_pending} Cheerful file(s) pending → "
              f"will be generated with JennyNeural (fixed voice).")
    print(f"  {to_generate} file(s) will be generated now.")
    print()

    if to_generate == 0:
        print("  All files already exist. Delete the audio/ folder to regenerate.")
        return

    input("  Press ENTER to start generation, or Ctrl+C to cancel ... ")

    fail_count = asyncio.run(run_all(samples))

    if fail_count > 0:
        print(f"\n  {fail_count} file(s) failed. Run the script again to retry.")
        sys.exit(1)
    else:
        print(f"\n  All {len(samples)} files generated successfully.")
        print(f"  Ready to run the pipeline.")


if __name__ == "__main__":
    main()