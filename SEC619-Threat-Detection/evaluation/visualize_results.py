# ============================================================================
# SEC619 — Results Visualization Script
# File    : evaluation/visualize_results.py
# Author  : Reem Fuad Shareef
# Supervisor: Dr. Waleed Algobi
# ============================================================================
#
# PURPOSE
# -------
# Generates publication-ready comparison charts for the thesis:
#   "From Speech to Safety: LLM-Driven Digital Threat Detection"
#
# Charts produced (saved to results/charts/):
#   01_safety_distribution.png   — Safe vs Unsafe count comparison
#   02_agreement_pie.png         — Pie chart: how often both systems agree
#   03_disagreement_breakdown.png — Four-bucket outcome analysis
#   04_latency_comparison.png    — Per-stage latency: System A vs B
#   05_category_distribution.png — Top threat categories
#   06_emotion_impact.png        — How emotion affects classification changes
#   07_performance_metrics.png   — Accuracy, Precision, Recall, F1 comparison
#   08_confusion_matrices.png    — Side-by-side confusion matrix heatmaps
#   09_improvement_breakdown.png — Predictions fixed vs broken by fusion
#
# USAGE
# -----
#   python evaluation/visualize_results.py
#
# REQUIREMENTS
# ------------
#   pip install matplotlib seaborn pandas numpy
# ============================================================================

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ── Third-party ──────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    sys.exit("ERROR: Run  pip install matplotlib numpy  first.")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False
    print("WARNING: seaborn not found — confusion matrices use plain matplotlib.")


# ============================================================================
# PATH CONFIGURATION (relative to repo root)
# ============================================================================

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "output_csv"
CHARTS_DIR  = REPO_ROOT / "results" / "charts"

BASELINE_CSV = RESULTS_DIR / "multimodal_result.csv"          # System A
FUSION_CSV   = RESULTS_DIR / "multimodal_result_fusion.csv"   # System B


# ============================================================================
# VISUAL SETTINGS
# ============================================================================

DPI        = 150             # Output resolution (150 dpi = print quality)
FIG_STD    = (10, 5.5)      # Standard chart dimensions
FIG_WIDE   = (12, 5.5)      # Wide chart (horizontal bars, many categories)
FIG_SQUARE = (7, 7)         # Square chart (pie charts)
FIG_DOUBLE = (13, 5.5)      # Double-wide (side-by-side confusion matrices)

# Brand colour palette
CLR_A        = "#2E7DB4"    # Steel blue  — System A (baseline)
CLR_B        = "#C94F2A"    # Burnt coral — System B (fusion)
CLR_SAFE     = "#2E8B5A"    # Green       — Safe label
CLR_UNSAFE   = "#B83232"    # Red         — Unsafe label
CLR_AGREE    = "#2E8B5A"    # Green       — Agreement sections
CLR_DISAGREE = "#C94F2A"    # Coral       — Disagreement sections
CLR_FIXED    = "#2E7DB4"    # Blue        — Fusion fixed a prediction
CLR_BROKE    = "#C94F2A"    # Coral       — Fusion broke a prediction
CLR_BOTH_OK  = "#4A7C3F"    # Dark green  — Both systems correct
CLR_BOTH_NG  = "#8A8A8A"    # Grey        — Both systems wrong

EMOTION_COLORS = {
    "agree":          "#4A7C3F",   # Agree (both classify same)
    "safe_to_unsafe": "#C94F2A",   # Fusion escalated Safe → Unsafe
    "unsafe_to_safe": "#2E7DB4",   # Fusion downgraded Unsafe → Safe
}


# ============================================================================
# HELPERS
# ============================================================================

def _load(path: Path) -> dict[str, dict]:
    """
    Load a pipeline CSV and return {filename: row_dict}.
    Key is the audio filename only (e.g. "C01_VIO_S01.wav").
    Returns empty dict if file does not exist.
    """
    if not path.exists():
        print(f"  WARNING — file not found: {path}")
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fp = (row.get("file_path") or "").strip()
            if fp:
                out[Path(fp).name] = row
    return out


def _flt(val, default: float = 0.0) -> float:
    """Safely convert a CSV value to float."""
    try:
        return float(val) if val not in (None, "", "None") else default
    except (TypeError, ValueError):
        return default


def _label(ax, bars, fmt="{:.0f}", va="bottom", pad=2, **kw):
    """Annotate bar heights with formatted value labels."""
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, pad), textcoords="offset points",
                    ha="center", va=va, fontsize=9, fontweight="bold", **kw)


def _legend(ax, labels: list[str], colors: list[str], **kw):
    """Add a patch legend to the axis."""
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, frameon=False, fontsize=9, **kw)


def _style(ax):
    """Apply a clean, minimal grid style (removes top/right spines)."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path):
    """Save a figure to disk and close it."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _compute_metrics(data: dict) -> Optional[dict]:
    """
    Compute binary classification metrics from the embedded ground_truth column.
    Returns None if no ground truth data is available.
    """
    tp = fp = tn = fn = 0
    for row in data.values():
        pred    = (row.get("safety") or "").strip().lower()
        gt      = (row.get("ground_truth") or "").strip().lower()
        if not pred or not gt:
            continue
        pred_pos = pred in ("unsafe", "controversial")
        gt_pos   = gt   in ("unsafe", "controversial")
        if   pred_pos and gt_pos:  tp += 1
        elif pred_pos:             fp += 1
        elif gt_pos:               fn += 1
        else:                      tn += 1
    total = tp + fp + tn + fn
    if total == 0:
        return None
    acc  = (tp + tn) / total * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1_u = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ps   = tn / (tn + fn) * 100 if (tn + fn) else 0.0
    rs   = tn / (tn + fp) * 100 if (tn + fp) else 0.0
    f1_s = 2 * ps * rs / (ps + rs) if (ps + rs) else 0.0
    mac  = (f1_u + f1_s) / 2
    return dict(accuracy=acc, precision=prec, recall=rec,
                f1_unsafe=f1_u, f1_safe=f1_s, macro_f1=mac,
                tp=tp, fp=fp, tn=tn, fn=fn)


def _count_categories(data: dict) -> Counter:
    """Count threat category occurrences across all CSV rows."""
    c: Counter = Counter()
    for row in data.values():
        cats = (row.get("categories") or "").strip()
        if not cats:
            continue
        for cat in cats.split(","):
            cat = cat.strip()
            if cat and cat.lower() != "none":
                c[cat] += 1
    return c


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def chart_safety_distribution(baseline: dict, fusion: dict, out: Path):
    """
    Chart 01 — Side-by-side bar chart of Safe vs Unsafe prediction counts.
    Shows whether fusion changes the overall label distribution.
    """
    bc = Counter(r.get("safety", "") for r in baseline.values())
    fc = Counter(r.get("safety", "") for r in fusion.values())
    labels = ["Safe", "Unsafe"]
    bv = [bc.get(l, 0) for l in labels]
    fv = [fc.get(l, 0) for l in labels]

    fig, ax = plt.subplots(figsize=FIG_STD)
    x, w = np.arange(len(labels)), 0.35
    b1 = ax.bar(x - w/2, bv, w, color=CLR_A, label="System A", zorder=3)
    b2 = ax.bar(x + w/2, fv, w, color=CLR_B, label="System B", zorder=3)
    _label(ax, b1); _label(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Samples"); ax.set_title("Safety label distribution", fontsize=13, fontweight="bold")
    _style(ax); _legend(ax, ["System A (baseline)", "System B (fusion)"], [CLR_A, CLR_B])
    fig.tight_layout(); _save(fig, out)


def chart_agreement_pie(baseline: dict, fusion: dict, out: Path):
    """
    Chart 02 — Pie chart showing the percentage of files where both systems agree.
    """
    common   = set(baseline) & set(fusion)
    agree    = sum(1 for f in common if baseline[f].get("safety") == fusion[f].get("safety"))
    disagree = len(common) - agree

    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    wedges, texts, pcts = ax.pie(
        [agree, disagree],
        labels=[f"Agree\n({agree})", f"Disagree\n({disagree})"],
        colors=[CLR_AGREE, CLR_DISAGREE],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    for p in pcts:
        p.set_fontweight("bold"); p.set_fontsize(13)
    ax.set_title("System A vs B — prediction agreement", fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout(); _save(fig, out)


def chart_disagreement_breakdown(baseline: dict, fusion: dict, out: Path):
    """
    Chart 03 — Four-bucket bar chart classifying all file outcomes:
    Both Safe, Both Unsafe, Safe→Unsafe (fusion escalated), Unsafe→Safe (fusion cleared).
    """
    common = set(baseline) & set(fusion)
    counts = {"Both safe": 0, "Both unsafe": 0,
              "Safe → Unsafe\n(fusion escalated)": 0,
              "Unsafe → Safe\n(fusion cleared)": 0}
    for fp in common:
        b = baseline[fp].get("safety", ""); f = fusion[fp].get("safety", "")
        if   b == "Safe"   and f == "Safe":   counts["Both safe"] += 1
        elif b == "Unsafe" and f == "Unsafe": counts["Both unsafe"] += 1
        elif b == "Safe"   and f == "Unsafe": counts["Safe → Unsafe\n(fusion escalated)"] += 1
        elif b == "Unsafe" and f == "Safe":   counts["Unsafe → Safe\n(fusion cleared)"] += 1

    fig, ax = plt.subplots(figsize=FIG_STD)
    bars = ax.bar(list(counts.keys()), list(counts.values()), color=[CLR_SAFE, CLR_UNSAFE, CLR_B, CLR_A], zorder=3)
    _label(ax, bars)
    ax.set_ylabel("Samples"); ax.set_title("Classification outcome breakdown", fontsize=13, fontweight="bold")
    _style(ax); fig.tight_layout(); _save(fig, out)


def chart_latency_comparison(baseline: dict, fusion: dict, out: Path):
    """
    Chart 04 — Grouped bar chart of mean latency per pipeline stage.
    Shows the overhead added by SpeechBrain in System B.
    Note: Uses raw CSV totals. For corrected totals, see compare_systems.py.
    """
    common = set(baseline) & set(fusion)
    stages = ["whisper_seconds", "tone_seconds", "guard_seconds", "total_seconds"]
    labels = ["Whisper (ASR)", "SpeechBrain (SER)", "Qwen3Guard (LLM)", "Total"]

    bav = [np.mean([_flt(baseline[f].get(s)) for f in common]) for s in stages]
    fav = [np.mean([_flt(fusion[f].get(s))   for f in common]) for s in stages]

    fig, ax = plt.subplots(figsize=FIG_STD)
    x, w = np.arange(len(labels)), 0.35
    b1 = ax.bar(x - w/2, bav, w, color=CLR_A, label="System A", zorder=3)
    b2 = ax.bar(x + w/2, fav, w, color=CLR_B, label="System B", zorder=3)
    _label(ax, b1, fmt="{:.2f}s", pad=3); _label(ax, b2, fmt="{:.2f}s", pad=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Average latency (s)"); ax.set_title("Pipeline latency — System A vs B", fontsize=13, fontweight="bold")
    _style(ax); _legend(ax, ["System A (baseline)", "System B (fusion)"], [CLR_A, CLR_B])
    fig.tight_layout(); _save(fig, out)


def chart_category_distribution(baseline: dict, fusion: dict, out: Path):
    """
    Chart 05 — Horizontal grouped bar chart of the top 8 threat categories.
    Shows which categories are most commonly detected by each system.
    """
    bc = _count_categories(baseline); fc = _count_categories(fusion)
    top = [cat for cat, _ in (bc + fc).most_common(8)]
    if not top:
        print("  WARNING — no categories found, skipping chart 05"); return

    bv = [bc.get(c, 0) for c in top]; fv = [fc.get(c, 0) for c in top]
    y, h = np.arange(len(top)), 0.35

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    b1 = ax.barh(y - h/2, bv, h, color=CLR_A, label="System A", zorder=3)
    b2 = ax.barh(y + h/2, fv, h, color=CLR_B, label="System B", zorder=3)
    for bar in (*b1, *b2):
        w = bar.get_width()
        ax.annotate(f"{int(w)}", xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(top); ax.invert_yaxis()
    ax.set_xlabel("Count"); ax.set_title("Top threat categories — System A vs B", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color="#EBEBEB", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    _legend(ax, ["System A", "System B"], [CLR_A, CLR_B], loc="lower right")
    fig.tight_layout(); _save(fig, out)


def chart_emotion_impact(baseline: dict, fusion: dict, out: Path):
    """
    Chart 06 — Grouped bar chart showing how each emotion class affects
    the classification change between System A and System B.
    """
    common = set(baseline) & set(fusion)
    buckets: dict[str, dict] = {}
    for fp in common:
        emo = (fusion[fp].get("tone_label_full") or "").strip()
        if not emo: continue
        b_s = baseline[fp].get("safety", ""); f_s = fusion[fp].get("safety", "")
        if emo not in buckets:
            buckets[emo] = {"agree": 0, "safe_to_unsafe": 0, "unsafe_to_safe": 0, "total": 0}
        buckets[emo]["total"] += 1
        if b_s == f_s: buckets[emo]["agree"] += 1
        elif b_s == "Safe"   and f_s == "Unsafe": buckets[emo]["safe_to_unsafe"] += 1
        elif b_s == "Unsafe" and f_s == "Safe":   buckets[emo]["unsafe_to_safe"] += 1

    emotions = [e for e, d in buckets.items() if d["total"] >= 2]
    if not emotions:
        print("  WARNING — insufficient emotion data, skipping chart 06"); return

    x, w = np.arange(len(emotions)), 0.25
    ag  = [buckets[e]["agree"]          for e in emotions]
    s2u = [buckets[e]["safe_to_unsafe"] for e in emotions]
    u2s = [buckets[e]["unsafe_to_safe"] for e in emotions]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    b1 = ax.bar(x - w, ag,  w, color=EMOTION_COLORS["agree"],          label="Agree",                zorder=3)
    b2 = ax.bar(x,     s2u, w, color=EMOTION_COLORS["safe_to_unsafe"], label="Safe → Unsafe (fusion)",zorder=3)
    b3 = ax.bar(x + w, u2s, w, color=EMOTION_COLORS["unsafe_to_safe"], label="Unsafe → Safe (fusion)",zorder=3)
    for bars in (b1, b2, b3): _label(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(emotions, rotation=30, ha="right")
    ax.set_ylabel("Samples"); ax.set_title("Emotion impact on safety classification", fontsize=13, fontweight="bold")
    _style(ax)
    _legend(ax, ["Agree", "Safe → Unsafe", "Unsafe → Safe"], list(EMOTION_COLORS.values()), loc="upper right")
    fig.tight_layout(); _save(fig, out)


def chart_performance_metrics(baseline: dict, fusion: dict, out: Path) -> tuple[Optional[dict], Optional[dict]]:
    """
    Chart 07 — Grouped bar chart of all classification metrics.
    Annotates each pair with the delta (+ green, − red).
    Requires the CSV to contain a 'ground_truth' column.
    """
    bm = _compute_metrics(baseline); fm = _compute_metrics(fusion)
    if not bm or not fm:
        print("  WARNING — no ground truth found, skipping chart 07"); return None, None

    keys  = ["accuracy", "precision", "recall", "f1_unsafe", "f1_safe", "macro_f1"]
    names = ["Accuracy", "Precision\n(unsafe)", "Recall\n(unsafe)",
             "F1 (unsafe)", "F1 (safe)", "Macro F1"]
    bv = [bm[k] for k in keys]; fv = [fm[k] for k in keys]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    x, w = np.arange(len(names)), 0.35
    b1 = ax.bar(x - w/2, bv, w, color=CLR_A, label="System A", zorder=3)
    b2 = ax.bar(x + w/2, fv, w, color=CLR_B, label="System B", zorder=3)
    _label(ax, b1, fmt="{:.1f}%"); _label(ax, b2, fmt="{:.1f}%")

    # Delta annotation above each bar pair
    for i, (b, f) in enumerate(zip(bv, fv)):
        diff = f - b
        color = "#2E8B5A" if diff > 0 else "#B83232" if diff < 0 else "#888888"
        sign  = "+" if diff >= 0 else ""
        ax.annotate(f"{sign}{diff:.1f}%", xy=(i, max(b, f) + 5),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)"); ax.set_title("Performance metrics — System A vs B", fontsize=13, fontweight="bold")
    _style(ax); _legend(ax, ["System A (baseline)", "System B (fusion)"], [CLR_A, CLR_B], loc="lower right")
    fig.tight_layout(); _save(fig, out)
    return bm, fm


def chart_confusion_matrices(bm: dict, fm: dict, out: Path):
    """
    Chart 08 — Side-by-side confusion matrix heatmaps.
    Rows = Actual class, Columns = Predicted class.
    Uses seaborn if available, otherwise plain matplotlib.
    """
    def _cm(m): return [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = ["Safe", "Unsafe"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    for ax, cm, title, cmap in zip(
        axes, [_cm(bm), _cm(fm)],
        ["System A (baseline)", "System B (fusion)"],
        ["Blues", "Oranges"],
    ):
        cm_arr = np.array(cm)
        if _SNS:
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap=cmap,
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        annot_kws={"size": 16, "weight": "bold"}, cbar=False,
                        linewidths=1, linecolor="white")
        else:
            ax.imshow(cm_arr, cmap=cmap)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(labels); ax.set_yticklabels(labels)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm_arr[i,j]), ha="center", va="center",
                            fontsize=16, fontweight="bold", color="black")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion matrices — Safe/Unsafe classification", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); _save(fig, out)


def chart_improvement_breakdown(baseline: dict, fusion: dict, out: Path):
    """
    Chart 09 — Bar chart showing how many predictions fusion fixed vs broke.
    Buckets: Both correct, Both wrong, Fusion fixed (N→Y), Fusion broke (Y→N).
    Requires the CSV to contain a 'correct' column.
    """
    common = set(baseline) & set(fusion)
    both_ok = both_ng = fixed = broke = no_gt = 0
    for fp in common:
        bc = (baseline[fp].get("correct") or "").strip().upper()
        fc = (fusion[fp].get("correct")   or "").strip().upper()
        if not bc or not fc: no_gt += 1; continue
        if   bc == "Y" and fc == "Y": both_ok += 1
        elif bc == "N" and fc == "N": both_ng += 1
        elif bc == "N" and fc == "Y": fixed   += 1
        elif bc == "Y" and fc == "N": broke   += 1

    if both_ok + both_ng + fixed + broke == 0:
        print("  WARNING — no 'correct' column found, skipping chart 09"); return

    net = fixed - broke; sign = "+" if net >= 0 else ""
    labels = ["Both correct", "Both wrong", "Fusion fixed", "Fusion broke"]
    values = [both_ok, both_ng, fixed, broke]
    colors = [CLR_BOTH_OK, CLR_BOTH_NG, CLR_FIXED, CLR_BROKE]

    fig, ax = plt.subplots(figsize=FIG_STD)
    bars = ax.bar(labels, values, color=colors, zorder=3)
    _label(ax, bars)
    ax.set_ylabel("Samples")
    ax.set_title("Fusion impact — fixed vs broken predictions", fontsize=13, fontweight="bold")
    _style(ax)
    color_net = "#2E8B5A" if net >= 0 else "#B83232"
    ax.text(0.98, 0.96, f"Net change: {sign}{net} samples",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=color_net)
    fig.tight_layout(); _save(fig, out)


# ============================================================================
# SUMMARY CONSOLE OUTPUT
# ============================================================================

def print_summary(bm: Optional[dict], fm: Optional[dict]):
    """Print a compact metrics summary table to the console."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("  PERFORMANCE SUMMARY")
    print(sep)
    if not bm or not fm:
        print("  (ground_truth column missing — metrics unavailable)"); return
    rows = [("Accuracy", bm["accuracy"], fm["accuracy"]),
            ("Precision", bm["precision"], fm["precision"]),
            ("Recall",    bm["recall"],    fm["recall"]),
            ("F1 (unsafe)", bm["f1_unsafe"], fm["f1_unsafe"]),
            ("F1 (safe)",   bm["f1_safe"],   fm["f1_safe"]),
            ("Macro F1",    bm["macro_f1"],  fm["macro_f1"])]
    print(f"  {'Metric':<16} {'System A':>10} {'System B':>10} {'Delta':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for name, bv, fv in rows:
        delta = fv - bv; sign = "+" if delta >= 0 else ""
        print(f"  {name:<16} {bv:>9.1f}% {fv:>9.1f}% {sign}{delta:>8.1f}%")
    print(f"\n  Confusion matrices")
    print(f"  System A  —  TP:{bm['tp']}  FP:{bm['fp']}  TN:{bm['tn']}  FN:{bm['fn']}")
    print(f"  System B  —  TP:{fm['tp']}  FP:{fm['fp']}  TN:{fm['tn']}  FN:{fm['fn']}")
    print(sep)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 64)
    print("  SEC619 — Visualization Generator")
    print("  Generates 9 publication-ready comparison charts")
    print("=" * 64)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading CSVs...")
    baseline = _load(BASELINE_CSV)
    fusion   = _load(FUSION_CSV)

    if not baseline:
        sys.exit(f"ERROR: Baseline CSV not found:\n  {BASELINE_CSV}\n  Run pipeline/pipeline.py first.")
    if not fusion:
        sys.exit(f"ERROR: Fusion CSV not found:\n  {FUSION_CSV}\n  Run pipeline/pipeline.py with FUSION_ENABLED=True.")

    print(f"  System A: {len(baseline)} samples")
    print(f"  System B: {len(fusion)} samples")
    print(f"\nGenerating charts → {CHARTS_DIR}\n")

    chart_safety_distribution(    baseline, fusion, CHARTS_DIR / "01_safety_distribution.png")
    chart_agreement_pie(          baseline, fusion, CHARTS_DIR / "02_agreement_pie.png")
    chart_disagreement_breakdown( baseline, fusion, CHARTS_DIR / "03_disagreement_breakdown.png")
    chart_latency_comparison(     baseline, fusion, CHARTS_DIR / "04_latency_comparison.png")
    chart_category_distribution(  baseline, fusion, CHARTS_DIR / "05_category_distribution.png")
    chart_emotion_impact(         baseline, fusion, CHARTS_DIR / "06_emotion_impact.png")

    bm, fm = chart_performance_metrics(baseline, fusion, CHARTS_DIR / "07_performance_metrics.png")
    if bm and fm:
        chart_confusion_matrices(bm, fm, CHARTS_DIR / "08_confusion_matrices.png")

    chart_improvement_breakdown(  baseline, fusion, CHARTS_DIR / "09_improvement_breakdown.png")

    print_summary(bm, fm)

    saved = len(list(CHARTS_DIR.glob("*.png")))
    print(f"\nDone. {saved} charts saved to:\n  {CHARTS_DIR}\n")


if __name__ == "__main__":
    main()
