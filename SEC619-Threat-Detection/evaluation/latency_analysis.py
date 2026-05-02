# ============================================================================
# SEC619 — Latency Analysis Script
# File    : evaluation/latency_analysis.py
# Author  : Reem Fuad Shareef
# Supervisor: Dr. Waleed Algobi
# ============================================================================
#
# PURPOSE
# -------
# Reads the pipeline output CSVs and prints the per-stage latency breakdown
# table required for the project report (Section 7.5, Table 7).
#
# Produces for each system:
#   - Mean, Std, Min, Max latency per stage
#   - Percentage of total end-to-end time
#   - List of the 3 slowest audio files
#
# USAGE
# -----
#   python evaluation/latency_analysis.py
#
# REQUIREMENTS
# ------------
#   pip install pandas
# ============================================================================

import pandas as pd
from pathlib import Path

# ── Path configuration (relative to repo root) ───────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "results" / "output_csv"

CSV_FUSION   = RESULTS_DIR / "multimodal_result_fusion.csv"   # System B
CSV_BASELINE = RESULTS_DIR / "multimodal_result.csv"          # System A


def analyse(csv_path: Path, label: str):
    """
    Load a pipeline result CSV and print the latency breakdown table.

    Filters to status=="OK" rows only so failed/incomplete files don't
    skew the latency statistics.

    Columns expected in the CSV:
        whisper_seconds, tone_seconds, guard_seconds, total_seconds

    Args:
        csv_path : Path to the pipeline output CSV file.
        label    : Human-readable label for the system (printed in header).
    """
    if not csv_path.exists():
        print(f"  ⚠  Not found: {csv_path}")
        print(f"     Run pipeline/pipeline.py first to generate this file.")
        return

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()    # Only use successfully processed files

    if df.empty:
        print(f"  ⚠  No OK rows in {csv_path.name}")
        return

    # Total mean used to compute percentage column
    total_mean = df["total_seconds"].dropna().astype(float).mean()

    print(f"\n  Latency Report — {label}")
    print(f"  {'Stage':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'% Total':>9}")
    print("  " + "-" * 73)

    # Pipeline stages in processing order
    stages = [
        ("whisper_seconds", "ASR — Whisper Large-v3"),
        ("tone_seconds",    "SER — SpeechBrain Emotion"),
        ("guard_seconds",   "LLM — Qwen3Guard-Gen-8B"),
        ("total_seconds",   "End-to-End Total"),
    ]
    for col, name in stages:
        if col not in df.columns:
            print(f"  {name:<30} column not found in CSV")
            continue
        s   = df[col].dropna().astype(float)
        pct = (s.mean() / total_mean * 100) if col != "total_seconds" else 100.0
        print(f"  {name:<30} {s.mean():>8.2f} {s.std():>8.2f} "
              f"{s.min():>8.2f} {s.max():>8.2f} {pct:>8.1f}%")

    print(f"\n  Files processed (OK): {len(df)}")

    # Show the 3 slowest files — useful for identifying bottlenecks
    slow = df.nlargest(3, "total_seconds")[["file_path", "total_seconds"]]
    print("  Slowest 3 files:")
    for _, row in slow.iterrows():
        name = Path(str(row["file_path"])).name
        print(f"    {name:<28} {row['total_seconds']:.2f}s")


# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 75)
print("  SEC619 — Pipeline Latency Breakdown")
print("  Dataset: 100 audio samples (50 Unsafe · 50 Safe)")
print("=" * 75)

analyse(CSV_FUSION,   "System B — Full Fusion Pipeline (Whisper + SpeechBrain + Qwen3Guard)")
analyse(CSV_BASELINE, "System A — Text-only Baseline (Whisper + Qwen3Guard)")

print()
print("  Note: For the thesis latency table, use the CORRECTED totals from")
print("  evaluation/compare_systems.py — this script uses raw total_seconds")
print("  which may be confounded by server warmup between runs.")
