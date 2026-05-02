# ============================================================================
# SEC619 — System A vs System B Comparison Analysis
# File    : evaluation/compare_systems.py
# Author  : Reem Fuad Shareef
# Supervisor: Dr. Waleed Algobi
# ============================================================================
#
# PURPOSE
# -------
# Reads the two pipeline CSV outputs (System A and System B) and produces:
#   1. comparison_sysA_vs_sysB.csv       — merged per-file comparison table
#   2. disagreement_cases.csv            — only files where systems disagree
#   3. performance_metrics_comparison.csv — all metrics side-by-side
#   4. comparison_summary_report.txt     — human-readable text report
#
# LATENCY CORRECTION METHODOLOGY
# --------------------------------
# Both System A and System B call all three pipeline stages in the notebook,
# because SpeechBrain runs unconditionally. The difference is only in what
# is SENT to Qwen3Guard:
#
#   System A → Qwen3Guard receives: <transcript>   (tone discarded)
#   System B → Qwen3Guard receives: [emotion] + <transcript>
#
# Because the two runs executed at different times on shared GPU servers,
# raw total_seconds is confounded by server warmup and network jitter.
# This script corrects for that by rebuilding the end-to-end total from
# individual stage times:
#
#   System A corrected total = whisper_seconds + guard_seconds
#   System B corrected total = whisper_seconds + tone_seconds + guard_seconds
#
# USAGE
# -----
#   python evaluation/compare_systems.py
#
# REQUIREMENTS
# ------------
#   pip install pandas numpy
# ============================================================================

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Paths are relative to the repo root (two levels up from this file).
# ============================================================================

REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "results" / "output_csv"

# Input: pipeline CSV outputs from the two runs
CSV_BASELINE = RESULTS_DIR / "multimodal_result.csv"          # System A
CSV_FUSION   = RESULTS_DIR / "multimodal_result_fusion.csv"   # System B

# Output: comparison reports and CSVs written to the same folder
COMPARISON_CSV   = RESULTS_DIR / "comparison_sysA_vs_sysB.csv"
DISAGREEMENT_CSV = RESULTS_DIR / "disagreement_cases.csv"
METRICS_CSV      = RESULTS_DIR / "performance_metrics_comparison.csv"
SUMMARY_REPORT   = RESULTS_DIR / "comparison_summary_report.txt"


# ============================================================================
# COLOUR CODES (terminal output only)
# ============================================================================

class C:
    RESET   = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
    RED     = "\033[91m"; GREEN = "\033[92m";  YELLOW= "\033[93m"
    CYAN    = "\033[96m"; WHITE = "\033[97m"


# ============================================================================
# HELPERS
# ============================================================================

def hr(ch="=", n=65):
    """Return a horizontal divider string."""
    return ch * n


def safe_float(val, default=0.0) -> float:
    """Convert a value to float safely, returning default on failure."""
    try:
        return float(val) if val not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def compute_corrected_latency(df: pd.DataFrame, is_fusion: bool) -> dict:
    """
    Compute corrected per-stage latency statistics.

    KEY CORRECTION:
    ───────────────
    System A (is_fusion=False):
        corrected_total = whisper_seconds + guard_seconds
        tone_seconds excluded — SpeechBrain ran but output was not fused.
        Including it would inflate System A's budget with unused work.

    System B (is_fusion=True):
        corrected_total = whisper_seconds + tone_seconds + guard_seconds
        All three stages contribute to the final classification.

    Stage percentages are recomputed from the corrected total so they sum
    to exactly 100%.

    Args:
        df        : DataFrame of pipeline results (status == "OK" rows only).
        is_fusion : True for System B, False for System A.

    Returns:
        dict mapping stage name → dict of {mean, std, min, max, pct, n}.
    """
    def col_stats(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "pct": 0.0, "n": 0}
        return {"mean": s.mean(), "std": s.std(ddof=1), "min": s.min(),
                "max": s.max(), "pct": 0.0, "n": len(s)}

    whisper = col_stats(df.get("whisper_seconds", pd.Series(dtype=float)))
    tone    = col_stats(df.get("tone_seconds",    pd.Series(dtype=float)))
    guard   = col_stats(df.get("guard_seconds",   pd.Series(dtype=float)))

    if is_fusion:
        # System B: all three stages count
        w = pd.to_numeric(df.get("whisper_seconds", 0), errors="coerce").fillna(0)
        t = pd.to_numeric(df.get("tone_seconds",    0), errors="coerce").fillna(0)
        g = pd.to_numeric(df.get("guard_seconds",   0), errors="coerce").fillna(0)
        corrected   = w + t + g
        total_mean  = corrected.mean()
        total_stats = {"mean": corrected.mean(), "std": corrected.std(ddof=1),
                       "min": corrected.min(), "max": corrected.max(), "pct": 100.0, "n": len(corrected)}
        whisper["pct"] = (whisper["mean"] / total_mean * 100) if total_mean > 0 else 0
        tone["pct"]    = (tone["mean"]    / total_mean * 100) if total_mean > 0 else 0
        guard["pct"]   = (guard["mean"]   / total_mean * 100) if total_mean > 0 else 0
        return {"ASR Whisper": whisper, "SER SpeechBrain": tone,
                "LLM Qwen3Guard": guard, "End-to-End Total": total_stats}
    else:
        # System A: whisper + guard only (tone excluded)
        w = pd.to_numeric(df.get("whisper_seconds", 0), errors="coerce").fillna(0)
        g = pd.to_numeric(df.get("guard_seconds",   0), errors="coerce").fillna(0)
        corrected   = w + g
        total_mean  = corrected.mean()
        total_stats = {"mean": corrected.mean(), "std": corrected.std(ddof=1),
                       "min": corrected.min(), "max": corrected.max(), "pct": 100.0, "n": len(corrected)}
        tone_excl = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                     "pct": 0.0, "n": 0, "note": "excluded (not fused)"}
        whisper["pct"] = (whisper["mean"] / total_mean * 100) if total_mean > 0 else 0
        guard["pct"]   = (guard["mean"]   / total_mean * 100) if total_mean > 0 else 0
        return {"ASR Whisper": whisper, "SER SpeechBrain": tone_excl,
                "LLM Qwen3Guard": guard, "End-to-End Total": total_stats}


def evaluate(csv_path: Path, is_fusion: bool) -> dict:
    """
    Load a pipeline result CSV and compute all evaluation metrics.

    Filters to status=="OK" rows only, then computes:
    - Confusion matrix: TP, TN, FP, FN
    - Classification metrics: accuracy, precision, recall, F1 (both classes), macro F1
    - Corrected latency statistics per stage
    - Safety label and threat category distributions
    - Lists of FN (missed threats) and FP (false alarms)

    Args:
        csv_path  : Path to the pipeline output CSV.
        is_fusion : True for System B (include tone in latency), False for System A.

    Returns:
        dict with all metrics, dataframes, and distribution counters.
        Returns empty dict if the file does not exist.
    """
    if not csv_path.exists():
        print(C.RED + f"File not found: {csv_path}" + C.RESET)
        return {}

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()

    if "ground_truth" not in df.columns or "safety" not in df.columns:
        print(C.YELLOW + f"Missing required columns in {csv_path.name}" + C.RESET)
        return {}

    df["pred"]  = df["safety"].str.strip().str.lower()
    df["truth"] = df["ground_truth"].str.strip().str.lower()
    df          = df.dropna(subset=["pred", "truth"])
    df          = df[df["truth"].isin(["unsafe", "safe"])]

    # Binary classification confusion matrix
    tp = int(((df.pred == "unsafe") & (df.truth == "unsafe")).sum())
    tn = int(((df.pred == "safe")   & (df.truth == "safe"  )).sum())
    fp = int(((df.pred == "unsafe") & (df.truth == "safe"  )).sum())
    fn = int(((df.pred == "safe")   & (df.truth == "unsafe")).sum())
    total = tp + tn + fp + fn

    # Classification metrics
    acc    = (tp + tn) / total * 100 if total > 0 else 0
    prec   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    prec_s = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
    rec_s  = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
    macro  = (f1 + f1_s) / 2

    # Corrected latency breakdown
    lat = compute_corrected_latency(df, is_fusion=is_fusion)

    # Safety label distribution
    safety_dist = dict(df["safety"].str.strip().value_counts())

    # Threat category distribution
    cat_counter: Counter = Counter()
    for cats in df["categories"].dropna():
        for cat in str(cats).split(","):
            cat = cat.strip()
            if cat and cat.lower() != "none":
                cat_counter[cat] += 1

    fn_rows = df[(df.pred == "safe")   & (df.truth == "unsafe")]
    fp_rows = df[(df.pred == "unsafe") & (df.truth == "safe")]

    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn, total=total,
        acc=acc, prec=prec, rec=rec, f1=f1,
        prec_s=prec_s, rec_s=rec_s, f1_s=f1_s, macro=macro,
        lat=lat, fn_rows=fn_rows, fp_rows=fp_rows,
        safety_dist=safety_dist, cat_counter=cat_counter,
        df=df, is_fusion=is_fusion,
    )


# ============================================================================
# CONSOLE PRINTERS
# ============================================================================

def print_confusion_matrix(label: str, r: dict):
    """Print a formatted confusion matrix to the console."""
    print(f"\n  Confusion Matrix — {label}")
    print(f"  {'':20s} {'Pred: UNSAFE':>14} {'Pred: SAFE':>12}")
    print(f"  {'Actual: UNSAFE':20s} {'TP = ' + str(r['tp']):>14} {'FN = ' + str(r['fn']):>12}  ← missed threats")
    print(f"  {'Actual: SAFE':20s} {'FP = ' + str(r['fp']):>14} {'TN = ' + str(r['tn']):>12}")


def print_metrics_table(A: dict, B: dict):
    """
    Print a side-by-side metrics comparison table (System A vs System B).
    Delta is colour-coded green (improvement) or red (regression).
    """
    rows = [
        ("Accuracy (%)",            "acc"),
        ("Precision — Unsafe (%)",  "prec"),
        ("Recall — Unsafe (%)",     "rec"),
        ("F1 Score — Unsafe (%)",   "f1"),
        ("Precision — Safe (%)",    "prec_s"),
        ("Recall — Safe (%)",       "rec_s"),
        ("F1 Score — Safe (%)",     "f1_s"),
        ("Macro-avg F1 (%)",        "macro"),
    ]
    print(f"\n  {'Metric':<26} {'System A':>10} {'System B':>10} {'Delta':>9}")
    print("  " + "-" * 57)
    for label, key in rows:
        a_val = A[key]; b_val = B[key]; delta = b_val - a_val
        col = C.GREEN if delta > 0 else (C.RED if delta < 0 else "")
        print(f"  {label:<26} {a_val:>10.1f} {b_val:>10.1f} "
              f"{col}{'+' if delta > 0 else ''}{delta:>8.1f}{C.RESET}")
    print("  " + "-" * 57)
    for label, key in [("TP","tp"),("TN","tn"),("FP","fp"),("FN","fn")]:
        a_val = A[key]; b_val = B[key]; delta = b_val - a_val
        col = (C.GREEN if delta < 0 else C.RED if delta > 0 else "") if key in ("fp","fn") \
              else (C.GREEN if delta > 0 else C.RED if delta < 0 else "")
        print(f"  {label:<26} {a_val:>10d} {b_val:>10d} "
              f"{col}{'+' if delta > 0 else ''}{delta:>8d}{C.RESET}")


def print_latency_table(A: dict, B: dict):
    """
    Print corrected latency comparison table with methodology footnote.

    System A tone row shows "—" because SpeechBrain ran but output was
    NOT used in the Qwen3Guard prompt for System A.
    """
    print(f"\n  {'Stage':<22} {'SysA Mean':>10} {'SysB Mean':>10} "
          f"{'Overhead':>10} {'SysB Std':>9} {'SysB%':>7}")
    print("  " + "-" * 72)

    stages = ["ASR Whisper", "SER SpeechBrain", "LLM Qwen3Guard", "End-to-End Total"]
    for stage in stages:
        a_s   = A["lat"].get(stage, {})
        b_s   = B["lat"].get(stage, {})
        a_m   = a_s.get("mean", 0)
        b_m   = b_s.get("mean", 0)
        b_std = b_s.get("std", 0)
        b_pct = b_s.get("pct", 0)

        if stage == "SER SpeechBrain":
            overhead = b_m; a_str = "—  (excl.)"; ov_str = f"+{overhead:.2f}s"; ov_col = C.YELLOW
        elif stage == "End-to-End Total":
            overhead = b_m - a_m; a_str = f"{a_m:.2f}s"
            ov_str = f"{'+' if overhead >= 0 else ''}{overhead:.2f}s"
            ov_col = C.YELLOW if overhead > 0 else C.GREEN
        else:
            overhead = b_m - a_m; a_str = f"{a_m:.2f}s" if a_m else "   —  "
            ov_str = f"{'+' if overhead >= 0 else ''}{overhead:.2f}s"; ov_col = C.GREEN if overhead <= 0 else ""

        sep = "─" * 72 if stage == "End-to-End Total" else ""
        if sep:
            print("  " + sep)
        print(f"  {stage:<22} {a_str:>10} {b_m:>9.2f}s "
              f"{ov_col}{ov_str:>10}{C.RESET} {b_std:>8.2f}s {b_pct:>6.1f}%")

    b_total  = B["lat"]["End-to-End Total"]["mean"]
    a_total  = A["lat"]["End-to-End Total"]["mean"]
    ser_cost = B["lat"]["SER SpeechBrain"]["mean"]
    print(f"\n  {C.CYAN}SpeechBrain fusion overhead  : +{ser_cost:.2f}s per sample{C.RESET}")
    print(f"  {C.CYAN}Net latency increase (B vs A): +{b_total - a_total:.2f}s "
          f"(+{(b_total - a_total)/a_total*100:.1f}%){C.RESET}")
    print(f"\n  {C.DIM}Methodology: System A total = Whisper + Guard (SpeechBrain excluded){C.RESET}")
    print(f"  {C.DIM}             System B total = Whisper + SpeechBrain + Guard{C.RESET}")
    print(f"  {C.DIM}Raw total_seconds NOT used — confounded by server warmup differences.{C.RESET}")


def print_agreement_section(A_df: pd.DataFrame, B_df: pd.DataFrame):
    """
    Print agreement analysis: how often System A and System B agree per file.
    Returns (merged_df, disagree_df) for downstream export.
    """
    a = A_df[["file_path","safety","truth"]].copy()
    b = B_df[["file_path","safety"]].copy()
    a["fname"] = a["file_path"].apply(lambda x: Path(x).name)
    b["fname"] = b["file_path"].apply(lambda x: Path(x).name)
    merged = a.merge(b, on="fname", suffixes=("_A","_B"))
    merged["agree"] = merged["safety_A"].str.lower() == merged["safety_B"].str.lower()
    disagree = merged[~merged["agree"]]
    total = len(merged); agree_n = merged["agree"].sum()

    print(f"\n  Total compared  : {total}")
    print(f"  Agree           : {agree_n} ({agree_n/total*100:.1f}%)")
    print(f"  Disagree        : {len(disagree)} ({len(disagree)/total*100:.1f}%)")

    if len(disagree):
        a2b = disagree[(disagree.safety_A.str.lower()=="safe") & (disagree.safety_B.str.lower()=="unsafe")]
        b2a = disagree[(disagree.safety_A.str.lower()=="unsafe") & (disagree.safety_B.str.lower()=="safe")]
        print(f"\n  Safe→Unsafe  (Fusion escalated new threat): {len(a2b)}")
        print(f"  Unsafe→Safe  (Fusion reduced alert level) : {len(b2a)}")
        print(f"\n  Disagreement cases:")
        for _, row in disagree.iterrows():
            print(f"    • {row['fname']:<28} SysA={row['safety_A']:<8} "
                  f"SysB={row['safety_B']:<8} GT={row['truth'].upper()}")

    return merged, disagree


# ============================================================================
# CSV / TXT EXPORTERS
# ============================================================================

def export_merged_csv(A_df: pd.DataFrame, B_df: pd.DataFrame, out_path: Path):
    """
    Export merged per-file comparison CSV.
    Adds columns: agreement, safety_change, sysA_correct, sysB_correct, improvement.
    """
    a = A_df.copy(); b = B_df.copy()
    a["fname"] = a["file_path"].apply(lambda x: Path(x).name)
    b["fname"] = b["file_path"].apply(lambda x: Path(x).name)
    b_sel = ["fname","safety","categories","guard_seconds","total_seconds"]
    for c in ["tone_label_full","tone_top_p"]:
        if c in b.columns: b_sel.append(c)
    merged = a.merge(b[b_sel], on="fname", suffixes=("_A","_B"))
    merged["agreement"] = merged.apply(
        lambda r: "AGREE" if r.safety_A.lower()==r.safety_B.lower() else "DISAGREE", axis=1)
    merged["safety_change"] = merged.apply(lambda r: (
        "No Change" if r.safety_A.lower()==r.safety_B.lower()
        else f"{r.safety_A} → {r.safety_B} (Fusion detected threat)" if r.safety_A.lower()=="safe"
        else f"{r.safety_A} → {r.safety_B} (Fusion reduced alert)"), axis=1)
    merged["sysA_correct"] = merged.apply(
        lambda r: "Y" if r.safety_A.lower()==r.truth.lower() else "N", axis=1)
    merged["sysB_correct"] = merged.apply(
        lambda r: "Y" if r.safety_B.lower()==r.truth.lower() else "N", axis=1)
    merged["improvement"] = merged.apply(lambda r: (
        "Fusion FIXED" if r.sysB_correct=="Y" and r.sysA_correct=="N"
        else "Fusion BROKE" if r.sysB_correct=="N" and r.sysA_correct=="Y"
        else "Same"), axis=1)
    out_cols = ["fname","ground_truth",
                "safety_A","sysA_correct","categories_A","guard_seconds_A","total_seconds_A",
                "safety_B","sysB_correct","categories_B","guard_seconds_B","total_seconds_B",
                "agreement","safety_change","improvement","transcript_preview",
                "tone_label_full","tone_top_p"]
    merged.rename(columns={"fname":"file_name"}, inplace=True)
    out_cols[0] = "file_name"
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(C.GREEN + f"  ✅ Merged CSV       : {out_path}" + C.RESET)
    return merged


def export_disagreement_csv(merged: pd.DataFrame, out_path: Path):
    """Export only the disagreement cases to a separate CSV."""
    dis = merged[merged["agreement"]=="DISAGREE"]
    if dis.empty:
        print(C.YELLOW + "  No disagreement cases." + C.RESET); return
    cols = ["file_name","ground_truth","safety_A","sysA_correct",
            "safety_B","sysB_correct","categories_A","categories_B",
            "tone_label_full","tone_top_p","transcript_preview","improvement"]
    cols = [c for c in cols if c in dis.columns]
    dis[cols].to_csv(out_path, index=False, encoding="utf-8")
    print(C.GREEN + f"  ✅ Disagreement CSV : {out_path}" + C.RESET)


def export_metrics_csv(A: dict, B: dict, out_path: Path):
    """Export all performance and latency metrics to a CSV for easy reporting."""
    rows_def = [("Accuracy","acc"),("Precision — Unsafe","prec"),("Recall — Unsafe","rec"),
                ("F1 — Unsafe","f1"),("Precision — Safe","prec_s"),("Recall — Safe","rec_s"),
                ("F1 — Safe","f1_s"),("Macro-avg F1","macro")]
    rows = []
    for label, key in rows_def:
        a_val=A[key]; b_val=B[key]; diff=b_val-a_val
        rows.append({"Metric":label, "System_A (%)":f"{a_val:.2f}", "System_B (%)":f"{b_val:.2f}",
                     "Delta":f"{'+' if diff>=0 else ''}{diff:.2f}",
                     "Winner":"System B" if diff>0 else ("System A" if diff<0 else "Tie")})
    for label, key in [("TP","tp"),("TN","tn"),("FP","fp"),("FN","fn")]:
        a_val=A[key]; b_val=B[key]; diff=b_val-a_val; lower_is_better = key in ("fp","fn")
        rows.append({"Metric":label,"System_A (%)":str(a_val),"System_B (%)":str(b_val),
                     "Delta":f"{'+' if diff>=0 else ''}{diff}",
                     "Winner":("System B" if diff<0 else "System A" if diff>0 else "Tie")
                              if lower_is_better
                              else ("System B" if diff>0 else "System A" if diff<0 else "Tie")})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(C.GREEN + f"  ✅ Metrics CSV      : {out_path}" + C.RESET)


def export_summary_txt(A: dict, B: dict, merged_df: pd.DataFrame, out_path: Path):
    """
    Write a comprehensive plain-text summary report.
    Sections: Overview, Metrics, Confusion Matrices, Agreement, Latency,
    Missed Threats, False Alarms, Key Insights.
    """
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def section(title):
        lines.extend(["", hr("="), f"  {title}", hr("=")])

    lines += [hr("█"), "  SYSTEM A vs SYSTEM B — COMPARISON ANALYSIS REPORT (CORRECTED LATENCY)",
              "  Dataset : 100 audio samples (50 Unsafe / 50 Safe)",
              f"  Generated: {ts}", hr("█")]

    section("1. OVERVIEW")
    lines += [
        "  System A (Baseline) : Whisper + Qwen3Guard (text-only, no SpeechBrain fusion)",
        "  System B (Fusion)   : Whisper + SpeechBrain + Fusion Layer + Qwen3Guard",
        f"  Total samples       : {A['total']}",
        "",
        "  LATENCY CORRECTION:",
        "  System A corrected total = Whisper + Qwen3Guard (SpeechBrain excluded)",
        "  System B corrected total = Whisper + SpeechBrain + Qwen3Guard",
    ]

    section("2. PERFORMANCE METRICS COMPARISON")
    rows_def = [("Accuracy","acc"),("Precision — Unsafe","prec"),("Recall — Unsafe","rec"),
                ("F1 — Unsafe","f1"),("Precision — Safe","prec_s"),("Recall — Safe","rec_s"),
                ("F1 — Safe","f1_s"),("Macro-avg F1","macro")]
    lines.append(f"\n  {'Metric':<24} {'System A':>10} {'System B':>10} {'Delta':>8}  Result")
    lines.append("  " + "-" * 65)
    for label, key in rows_def:
        a_val=A[key]; b_val=B[key]; delta=b_val-a_val
        result="▲ Better" if delta>0 else ("▼ Worse" if delta<0 else "Tie")
        lines.append(f"  {label:<24} {a_val:>10.1f} {b_val:>10.1f} "
                     f"{'+' if delta>=0 else ''}{delta:>7.1f}  {result}")
    lines.append("  " + "-" * 65)
    for label, key in [("TP","tp"),("TN","tn"),("FP","fp"),("FN","fn")]:
        a_val=A[key]; b_val=B[key]; delta=b_val-a_val
        lines.append(f"  {label:<24} {a_val:>10} {b_val:>10} {'+' if delta>=0 else ''}{delta:>8}")

    section("3. CONFUSION MATRICES")
    for label, r in [("System A — Baseline", A), ("System B — Fusion Pipeline", B)]:
        lines += [f"\n  {label}",
                  f"  {'':20s} {'Pred: UNSAFE':>14} {'Pred: SAFE':>12}",
                  f"  {'Actual: UNSAFE':20s} {'TP = '+str(r['tp']):>14} {'FN = '+str(r['fn']):>12}  ← missed threats",
                  f"  {'Actual: SAFE':20s} {'FP = '+str(r['fp']):>14} {'TN = '+str(r['tn']):>12}"]

    section("4. AGREEMENT ANALYSIS")
    total=len(merged_df); agree=(merged_df["agreement"]=="AGREE").sum(); disagr=total-agree
    lines += [f"  Total compared  : {total}",
              f"  Agree           : {agree} ({agree/total*100:.1f}%)",
              f"  Disagree        : {disagr} ({disagr/total*100:.1f}%)"]
    if disagr:
        for _, row in merged_df[merged_df["agreement"]=="DISAGREE"].iterrows():
            lines.append(f"    • {row['file_name']:<28} SysA={row['safety_A']:<8} "
                         f"SysB={row['safety_B']:<8} GT={row['ground_truth'].upper()}")

    section("5. LATENCY COMPARISON (corrected — seconds)")
    a_total = A["lat"]["End-to-End Total"]["mean"]; b_total = B["lat"]["End-to-End Total"]["mean"]
    ser_cost = B["lat"]["SER SpeechBrain"]["mean"]; overhead_pct = (b_total-a_total)/a_total*100 if a_total>0 else 0
    lines.append(f"\n  {'Stage':<22} {'SysA Mean':>12} {'SysB Mean':>12} {'Overhead':>10} {'SysB Std':>10} {'SysB%':>7}")
    lines.append("  " + "-" * 76)
    for stage in ["ASR Whisper","SER SpeechBrain","LLM Qwen3Guard","End-to-End Total"]:
        a_s=A["lat"].get(stage,{}); b_s=B["lat"].get(stage,{})
        a_m=a_s.get("mean",0); b_m=b_s.get("mean",0); b_std=b_s.get("std",0); b_pct=b_s.get("pct",0)
        a_str = "0.00s (excl.)" if stage=="SER SpeechBrain" else (f"{a_m:.2f}s" if a_m else "   —  ")
        ov = f"+{b_m:.2f}s" if stage=="SER SpeechBrain" else f"{b_m-a_m:+.2f}s"
        sep = "  "+"-"*76+"\n" if stage=="End-to-End Total" else ""
        lines.append(sep+f"  {stage:<22} {a_str:>12} {b_m:>11.2f}s {ov:>10} {b_std:>9.2f}s {b_pct:>6.1f}%")
    lines += ["", f"  SpeechBrain fusion overhead  : +{ser_cost:.2f}s per sample (mean)",
              f"  Net latency increase (B vs A) : +{b_total-a_total:.2f}s (+{overhead_pct:.1f}%)"]

    section("6. MISSED THREATS — SYSTEM A (FN)")
    if len(A["fn_rows"]):
        for _,row in A["fn_rows"].iterrows():
            lines.append(f"  • {Path(str(row.get('file_path',''))).name:<28} "
                         f"Transcript: {str(row.get('transcript_preview',''))[:100]}...")
    else:
        lines.append("  None — System A caught all threats.")

    section("7. MISSED THREATS — SYSTEM B (FN)")
    if len(B["fn_rows"]):
        for _,row in B["fn_rows"].iterrows():
            lines.append(f"  • {Path(str(row.get('file_path',''))).name:<28} "
                         f"Transcript: {str(row.get('transcript_preview',''))[:100]}...")
    else:
        lines.append("  None — System B caught all threats.")

    section("8. FALSE ALARMS — SYSTEM B (FP)")
    if len(B["fp_rows"]):
        for _,row in B["fp_rows"].iterrows():
            lines.append(f"  • {Path(str(row.get('file_path',''))).name:<28} "
                         f"Transcript: {str(row.get('transcript_preview',''))[:100]}...")
    else:
        lines.append("  None — System B generated zero false alarms.")

    section("9. KEY INSIGHTS")
    acc_diff=B["acc"]-A["acc"]; f1_diff=B["macro"]-A["macro"]
    if acc_diff>0: lines.append(f"  ★ Fusion IMPROVES accuracy by +{acc_diff:.1f}%")
    if f1_diff>0:  lines.append(f"  ★ Fusion IMPROVES Macro-avg F1 by +{f1_diff:.1f}%")
    if B["fp"]==0: lines.append("  ✓ Zero false alarms in System B — perfect Unsafe precision")
    lines += [f"  • SpeechBrain adds +{ser_cost:.2f}s per sample ({overhead_pct:.1f}% overhead)",
              "  • Hardest cases: empathetic/ambiguous language (self-harm as support)",
              "  • Future: dedicated self-harm classifiers, ASR normalization, streaming"]
    lines += ["", hr("─"), "  END OF REPORT", hr("─")]

    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(C.GREEN + f"  ✅ Summary report   : {out_path}" + C.RESET)
    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + C.BOLD + C.CYAN + hr() + C.RESET)
    print(C.BOLD + "  SEC619 — System A vs B Comparison (Corrected Latency)".center(65) + C.RESET)
    print(C.BOLD + C.CYAN + hr() + C.RESET)

    print(C.CYAN + "\n  Loading CSV files..." + C.RESET)
    A = evaluate(CSV_BASELINE, is_fusion=False)   # System A: whisper + guard
    B = evaluate(CSV_FUSION,   is_fusion=True)    # System B: whisper + tone + guard

    if not A or not B:
        print(C.RED + "  ERROR: Could not load both CSV files. Check paths in SECTION 'PATH CONFIGURATION'." + C.RESET)
        return

    print(C.GREEN + f"  ✅ System A loaded  : {A['total']} samples" + C.RESET)
    print(C.GREEN + f"  ✅ System B loaded  : {B['total']} samples" + C.RESET)
    print(C.YELLOW + "\n  ⚠  LATENCY CORRECTION APPLIED" + C.RESET)
    print(C.DIM + "  A total = Whisper + Guard  |  B total = Whisper + SpeechBrain + Guard" + C.RESET)

    print("\n" + C.BOLD + C.CYAN + hr("-") + C.RESET)
    print(C.BOLD + "  PERFORMANCE METRICS" + C.RESET)
    print_metrics_table(A, B)

    print("\n" + C.BOLD + C.CYAN + hr("-") + C.RESET)
    print(C.BOLD + "  CONFUSION MATRICES" + C.RESET)
    print_confusion_matrix("System A — Text-only Baseline", A)
    print_confusion_matrix("System B — Full Fusion Pipeline", B)

    print("\n" + C.BOLD + C.CYAN + hr("-") + C.RESET)
    print(C.BOLD + "  LATENCY BREAKDOWN  (corrected)" + C.RESET)
    print_latency_table(A, B)

    print("\n" + C.BOLD + C.CYAN + hr("-") + C.RESET)
    print(C.BOLD + "  AGREEMENT ANALYSIS" + C.RESET)
    merged_df, _ = print_agreement_section(A["df"], B["df"])

    print("\n" + C.BOLD + C.CYAN + hr("-") + C.RESET)
    print(C.BOLD + "  EXPORTING OUTPUT FILES" + C.RESET + "\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_full = export_merged_csv(A["df"], B["df"], COMPARISON_CSV)
    export_disagreement_csv(merged_full, DISAGREEMENT_CSV)
    export_metrics_csv(A, B, METRICS_CSV)
    export_summary_txt(A, B, merged_full, SUMMARY_REPORT)

    a_e2e = A["lat"]["End-to-End Total"]["mean"]
    b_e2e = B["lat"]["End-to-End Total"]["mean"]
    ser   = B["lat"]["SER SpeechBrain"]["mean"]

    print("\n" + C.BOLD + C.CYAN + hr() + C.RESET)
    print(C.BOLD + "  KEY VALUES FOR YOUR REPORT" + C.RESET)
    print(f"    Macro-avg F1     : SysA={A['macro']:.1f}%   SysB={B['macro']:.1f}%")
    print(f"    Corrected E2E    : SysA={a_e2e:.2f}s  SysB={b_e2e:.2f}s")
    print(f"    SpeechBrain cost : +{ser:.2f}s per sample")
    print(f"    Net overhead     : +{b_e2e-a_e2e:.2f}s (+{(b_e2e-a_e2e)/a_e2e*100:.1f}%)")
    print(f"    FP: {A['fp']} → {B['fp']}   FN: {A['fn']} → {B['fn']}")
    print("\n" + C.BOLD + C.CYAN + hr() + C.RESET + "\n")


if __name__ == "__main__":
    main()
