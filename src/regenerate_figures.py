"""
Regenerate Manuscript Plots (Figures 4, 5, 6, 7, and 8)
from saved results and metrics CSV files.

Manuscript: "Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"
Authors: Ari Muzakir, Usman Ependi, Suyanto (2026)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Academic styling settings
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
mpl.rcParams["axes.edgecolor"] = "#222222"
mpl.rcParams["axes.linewidth"] = 0.8

BASE_BLUE = "#1B365D"
BASE_RED = "#B22222"
BASE_GREEN = "#2E7D32"
BASE_ORANGE = "#D95F02"
BASE_PURPLE = "#7570B3"
BASE_GRAY = "#555555"
BASE_DARK = "#111111"
BASE_LIGHT = "#F8F9FA"
BASE_CYAN = "#00838F"
BASE_LINE = "#222222"
BASE_GRID = "#CCCCCC"


def generate_figure_4():
    """Fig. 4: Held-out comment model comparison."""
    csv_path = RESULTS_DIR / "comment_metrics.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 4: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)

    models = [
        ("tfidf_word_char_lr", "Word/char TF-IDF + LR", BASE_GRAY),
        ("indobert_p2", "IndoBERT-base-p2", BASE_BLUE),
        ("indobertweet", "IndoBERTweet", BASE_PURPLE),
        ("mbert", "Multilingual BERT", BASE_ORANGE),
        ("equal_text_ensemble", "Equal Text Ensemble", BASE_GREEN),
    ]

    data = []
    for k, name, col in models:
        sub = df[df["model"] == k]
        if len(sub):
            r = sub.iloc[0]
            data.append({"name": name, "f1": r["macro_f1"], "color": col})

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(data))
    for i, d in enumerate(data):
        ax.plot([0.90, d["f1"]], [i, i], color="#DDDDDD", linewidth=1.5, zorder=1)
        ax.scatter(d["f1"], i, s=140, color=d["color"], edgecolors=BASE_DARK, linewidth=1.0, zorder=3)
        ax.text(d["f1"] + 0.0015, i, f"{d['f1']:.4f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color=BASE_DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d["name"] for d in data], fontsize=10, fontweight="medium")
    ax.set_xlim(0.925, 0.975)
    ax.set_xlabel("Held-Out Macro F1 (Seed 42)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_title("Fig. 4: Held-Out Comment Model Comparison (N = 1,226 test comments)", fontsize=11.5, fontweight="bold", loc="left", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_04_comment_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated: {out.name}")


def generate_figure_5():
    """Fig. 5: Source-paired differences in video macro F1 (Forest plot)."""
    csv_path = RESULTS_DIR / "paired_statistics/video_equal_vat_paired_group_statistics.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 5: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)

    baseline_info = {
        "text": ("Extracted text", BASE_BLUE),
        "visual": ("Visual only", BASE_RED),
        "siglip2_vta_probe": ("SigLIP 2 VTA probe", BASE_ORANGE),
        "healnet_adapted_2024": ("HEALNet adaptation", BASE_GREEN),
    }

    rows = []
    for _, r in df.iterrows():
        b = r["baseline"]
        if b in baseline_info:
            name, col = baseline_info[b]
            rows.append({
                "name": name,
                "color": col,
                "delta": r["delta_macro_f1"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p_holm_11": r["group_permutation_p_holm_extended_11"],
            })
    rows.sort(key=lambda x: x["delta"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(rows))
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.2, zorder=1)

    for i, r in enumerate(rows):
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=BASE_DARK, linewidth=2.0, zorder=2)
        ax.plot([r["ci_low"], r["ci_low"]], [i - 0.1, i + 0.1], color=BASE_DARK, linewidth=1.8, zorder=2)
        ax.plot([r["ci_high"], r["ci_high"]], [i - 0.1, i + 0.1], color=BASE_DARK, linewidth=1.8, zorder=2)
        ax.scatter(r["delta"], i, s=140, marker="D", color=r["color"], edgecolors=BASE_DARK, linewidth=1.0, zorder=3)
        stat = f"$\\Delta$ = {r['delta']:+.4f}; p = {r['p_holm_11']:.4f}"
        ax.text(0.045, i - 0.25, stat, fontsize=9.5, fontweight="bold", color=BASE_DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=10, fontweight="medium")
    ax.set_xlim(-0.025, 0.10)
    ax.set_xlabel("Macro F1 Difference (Equal Late Fusion minus Baseline)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_title("Fig. 5: Source-Paired Differences in Video Macro F1", fontsize=11.5, fontweight="bold", loc="left", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_05_fusion_differences.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated: {out.name}")


def generate_figure_6():
    """Fig. 6: Video macro F1 across three training seeds."""
    csv_path = RESULTS_DIR / "video_seed_metrics.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 6: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)

    models = [
        ("text", "Extracted text", BASE_BLUE),
        ("equal_vat", "Equal late fusion", BASE_GREEN),
        ("vat_coverage", "Availability-aware fusion", BASE_RED),
    ]
    seeds = [42, 2026, 2027]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for key, name, col in models:
        sub = df[df["model"] == key]
        vals = [sub[sub["seed"] == s]["macro_f1"].values[0] for s in seeds]
        ax.plot(seeds, vals, marker="o", markersize=8, linewidth=2.0, label=name, color=col)
        for s, v in zip(seeds, vals):
            ax.text(s, v + 0.0012, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold", color=col)

    ax.set_xticks(seeds)
    ax.set_xticklabels(["Seed 42", "Seed 2026", "Seed 2027"], fontsize=10)
    ax.set_ylabel("Held-Out Video Macro F1", fontsize=10.5, fontweight="bold")
    ax.set_title("Fig. 6: Video Macro F1 across Three Training Seeds", fontsize=11.5, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_06_seed_stability.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated: {out.name}")


def generate_figure_7():
    """Fig. 7: OCR / STT availability regimes with segment counts."""
    csv_path = RESULTS_DIR / "video_platform_and_coverage_slices.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 7: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    cov = df[df["slice_type"] == "coverage"]

    slice_specs = [
        ("(np.int64(0), np.int64(0))", "No Extraction\n(n = 685, 17.3%)"),
        ("(np.int64(0), np.int64(1))", "STT Only\n(n = 1,231, 31.2%)"),
        ("(np.int64(1), np.int64(0))", "OCR Only\n(n = 760, 19.2%)"),
        ("(np.int64(1), np.int64(1))", "Both Available\n(n = 1,275, 32.3%)"),
    ]

    models_to_compare = [
        ("equal_vat", "Equal Late Fusion (VAT)", BASE_GREEN),
        ("text", "Unimodal Text", BASE_BLUE),
        ("siglip2_vta_probe", "SigLIP 2 VTA Probe", BASE_ORANGE),
        ("visual", "Unimodal Visual", BASE_RED),
        ("audio", "Unimodal Audio", BASE_GRAY),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 5.0), sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(models_to_compare))

    for col_idx, (slice_key, header_title) in enumerate(slice_specs):
        ax = axes[col_idx]
        ax.set_facecolor("#FFFFFF")
        sub = cov[cov["slice"] == slice_key]

        ax.set_title(header_title, fontsize=10.5, fontweight="bold", pad=12, color=BASE_DARK)

        scores = []
        for m_key, m_name, col in models_to_compare:
            row = sub[sub["model"] == m_key]
            score = row.iloc[0]["macro_f1"] if not row.empty else 0.0
            scores.append((m_name, score, col))

        for i, (m_name, score, col) in enumerate(scores):
            ax.barh(i, score, color=col, height=0.55, edgecolor=BASE_DARK, linewidth=1.0, zorder=3)
            ax.text(score + 0.006, i, f"{score:.4f}", va="center", ha="left", fontsize=8.8, fontweight="bold", color=BASE_DARK)

        ax.set_yticks(y_pos)
        if col_idx == 0:
            ax.set_yticklabels([m[1] for m in models_to_compare], fontsize=10, fontweight="semibold")
        else:
            ax.tick_params(left=False)

        ax.set_xlim(0.40, 0.63)
        ax.set_xlabel("Macro F1", fontsize=10, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BASE_LINE)
        ax.spines["bottom"].set_color(BASE_LINE)
        ax.xaxis.grid(True, linestyle=":", alpha=0.7, color=BASE_GRID)
        ax.set_axisbelow(True)

    fig.suptitle("Fig. 7: Video Performance by Extraction Availability (N = 3,951)", fontsize=12.5, fontweight="bold", x=0.08, y=0.98, ha="left", color=BASE_DARK)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.16, right=0.96, wspace=0.18)
    out = FIGURES_DIR / "fig_07_extraction_availability.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated: {out.name}")


def generate_figure_8():
    """Fig. 8: Decision threshold comparison (0.50 vs calibrated)."""
    csv_path = RESULTS_DIR / "video_seed_metrics.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 8: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    s42 = df[df["seed"] == 42]

    models_to_compare = [
        ("equal_vat", "Equal Late Fusion (VAT)", BASE_GREEN),
        ("text", "Unimodal Text", BASE_BLUE),
        ("siglip2_vta_probe", "SigLIP 2 VTA Probe", BASE_ORANGE),
        ("visual", "Unimodal Visual", BASE_RED),
        ("equal_va", "Equal VA (Visual + Audio)", BASE_PURPLE),
        ("va_lr", "VA Logistic Regression", BASE_CYAN),
        ("vat_flags", "VAT Modality Flags", "#E65100"),
        ("vat_lr", "VAT Logistic Regression", "#1B5E20"),
        ("vat_coverage", "Availability-Aware Fusion", "#C2185B"),
        ("audio", "Unimodal Audio", BASE_GRAY),
    ]

    paired_data = []
    for m_key, m_name, col in models_to_compare:
        row = s42[s42["model"] == m_key]
        if not row.empty:
            r = row.iloc[0]
            f1_050 = r["macro_f1_fixed_050"]
            f1_sel = r["macro_f1"]
            delta = f1_sel - f1_050
            paired_data.append({
                "key": m_key,
                "name": m_name,
                "color": col,
                "f1_050": f1_050,
                "f1_sel": f1_sel,
                "delta": delta,
            })

    paired_data.sort(key=lambda x: x["f1_sel"])

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(paired_data))

    for i, d in enumerate(paired_data):
        y = i
        f0 = d["f1_050"]
        f1 = d["f1_sel"]

        line_color = BASE_GREEN if d["delta"] > 0 else BASE_RED
        ax.plot([f0, f1], [y, y], color=line_color, linewidth=2.0, zorder=2)

        if abs(d["delta"]) > 0.015:
            ax.annotate("", xy=(f1, y), xytext=(f0, y),
                        arrowprops=dict(arrowstyle="->", color=line_color, lw=1.6, mutation_scale=12),
                        zorder=2)

        ax.scatter(f0, y, s=120, color=BASE_GRAY, edgecolors=BASE_DARK, linewidth=1.0, zorder=3)
        ax.scatter(f1, y, s=150, color=d["color"], edgecolors=BASE_DARK, linewidth=1.2, zorder=4)

        delta_str = f"$\\Delta$ {d['delta']:+.4f}" if abs(d["delta"]) >= 0.001 else "$\\Delta$ ~0.000"
        delta_color = BASE_GREEN if d["delta"] > 0 else BASE_RED
        ax.text(max(f0, f1) + 0.012, y, delta_str, va="center", ha="left", fontsize=9.0, fontweight="bold", color=delta_color)

    y_labels = [d["name"] for d in paired_data]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="semibold")

    ax.set_xlim(0.33, 0.66)
    ax.set_ylim(-0.6, len(paired_data) - 0.4)
    ax.set_xlabel("Macro F1 Score (N = 3,951 Video Segments)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Fig. 8: Decision Threshold Comparison (Default 0.50 vs. Selected $\\tau^*$)", fontsize=12.5, fontweight="bold", loc="left", pad=14, color=BASE_DARK)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_GRAY, markeredgecolor=BASE_DARK, markersize=9, label="Default Threshold (t = 0.50)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_GREEN, markeredgecolor=BASE_DARK, markersize=10, label="Selected Threshold (t = $\\tau^*$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", alpha=0.7, color=BASE_GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out = FIGURES_DIR / "fig_08_threshold_sensitivity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated: {out.name}")


if __name__ == "__main__":
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_figure_7()
    generate_figure_8()
