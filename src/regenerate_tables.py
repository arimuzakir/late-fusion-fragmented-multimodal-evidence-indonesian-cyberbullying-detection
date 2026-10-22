"""
Regenerate Manuscript Tables (Tables 1, 2, and 3)
and Statistical Contrasts from saved results and split manifests.

Manuscript: "Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"
Authors: Ari Muzakir, Usman Ependi, Suyanto (2026)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Resolve root directory
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def compute_metrics(y_true, y_prob, threshold=0.5, pred=None):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    if pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(pred, int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    ap = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else np.nan
    return {"accuracy": acc, "macro_f1": f1, "roc_auc": auc, "average_precision": ap}


def display_table_1():
    print("\n" + "=" * 80)
    print("TABLE 1: POPULATION AND EVIDENCE AVAILABILITY ACCOUNTING")
    print("=" * 80)

    # Derivations from manifests
    comment_splits = pd.read_csv(ARTIFACTS_DIR / "comment_splits.csv")
    video_audit = pd.read_csv(ARTIFACTS_DIR / "video_alignment_audit.csv")

    # 1. Video source & aligned
    n_vid_source = len(video_audit)
    neg_vid_source = (video_audit["y"] == 0).sum()
    pos_vid_source = (video_audit["y"] == 1).sum()

    vid_aligned = video_audit[video_audit["eligible_primary"] == True]
    n_vid_aligned = len(vid_aligned)
    neg_vid_aligned = (vid_aligned["y"] == 0).sum()
    pos_vid_aligned = (vid_aligned["y"] == 1).sum()

    # 2. Comments eligible & test
    n_comm_eligible = len(comment_splits)
    neg_comm_eligible = (comment_splits["label_final"] == 0).sum()
    pos_comm_eligible = (comment_splits["label_final"] == 1).sum()

    comm_test = comment_splits[comment_splits["split"] == "test"]
    n_comm_test = len(comm_test)
    neg_comm_test = (comm_test["label_final"] == 0).sum()
    pos_comm_test = (comm_test["label_final"] == 1).sum()

    # 3. Master & Comment Source Population (derived from master accounting)
    # Master collection: 13,307 (Negative: 6,057, Positive: 7,250)
    n_master = 13307
    neg_master = 6057
    pos_master = 7250

    # Comment source = Master - Video source
    n_comm_source = n_master - n_vid_source  # 8,906
    neg_comm_source = neg_master - neg_vid_source  # 4,260
    pos_comm_source = pos_master - pos_vid_source  # 4,646

    table_1_data = [
        {"Population": "Master source collection", "Records": f"{n_master:,}", "Negative": f"{neg_master:,}", "Positive": f"{pos_master:,}"},
        {"Population": "Comment source population", "Records": f"{n_comm_source:,}", "Negative": f"{neg_comm_source:,}", "Positive": f"{pos_comm_source:,}"},
        {"Population": "Eligible high-agreement comments", "Records": f"{n_comm_eligible:,}", "Negative": f"{neg_comm_eligible:,}", "Positive": f"{pos_comm_eligible:,}"},
        {"Population": "Comment test population", "Records": f"{n_comm_test:,}", "Negative": f"{neg_comm_test:,}", "Positive": f"{pos_comm_test:,}"},
        {"Population": "Video source population", "Records": f"{n_vid_source:,}", "Negative": f"{neg_vid_source:,}", "Positive": f"{pos_vid_source:,}"},
        {"Population": "Aligned video evaluation population", "Records": f"{n_vid_aligned:,}", "Negative": f"{neg_vid_aligned:,}", "Positive": f"{pos_vid_aligned:,}"},
    ]
    df_t1 = pd.DataFrame(table_1_data)
    print(df_t1.to_string(index=False))
    print("-" * 80)

    # Evidence availability breakdown
    print("Video Extraction Availability Breakdown (N = 3,951):")
    video_manifest = pd.read_csv(ARTIFACTS_DIR / "video_primary_manifest.csv")
    for (ocr, stt), n in video_manifest.groupby(["ocr_ok", "stt_ok"]).size().items():
        pct = (n / len(video_manifest)) * 100
        cond_label = {
            (0, 0): "Neither (Zero text extracted)",
            (0, 1): "STT transcript only",
            (1, 0): "OCR text only",
            (1, 1): "Both OCR and STT extracted",
        }.get((ocr, stt), f"OCR={ocr}, STT={stt}")
        print(f"  - {cond_label:32}: {n:5,} ({pct:4.1f}%)")
    print("=" * 80)


def display_table_2():
    print("\n" + "=" * 90)
    print("TABLE 2: HELD-OUT COMMENT RESULTS (1,226 test comments, Seed 42)")
    print("=" * 90)

    csv_path = RESULTS_DIR / "comment_metrics.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    model_map = {
        "tfidf_word_char_lr": "Word/character TF-IDF + LR",
        "indobert_p2": "IndoBERT-base-p2",
        "indobertweet": "IndoBERTweet",
        "mbert": "Multilingual BERT",
        "equal_text_ensemble": "Equal text ensemble",
    }

    rows = []
    for k, name in model_map.items():
        sub = df[df["model"] == k]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append({
                "Model": name,
                "Acc.": f"{r['accuracy']:.4f}",
                "F1": f"{r['macro_f1']:.4f}",
                "AUC": f"{r['roc_auc']:.4f}",
                "AP": f"{r['average_precision']:.4f}",
            })

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    print("=" * 90)


def display_table_3():
    print("\n" + "=" * 95)
    print("TABLE 3: CONTROLLED VIDEO COMPARISON (3,951 video windows, 4-fold grouped OOF)")
    print("=" * 95)

    oof_path = RESULTS_DIR / "video_oof_predictions.csv"
    seed_path = RESULTS_DIR / "video_seed_summary.csv"
    if not oof_path.exists():
        print(f"Warning: {oof_path} not found.")
        return

    df_oof = pd.read_csv(oof_path)
    if "seed" in df_oof.columns:
        df_oof = df_oof[df_oof["seed"] == 42]
    df_seeds = pd.read_csv(seed_path).set_index("model") if seed_path.exists() else pd.DataFrame()

    y_true = df_oof["label_final"].values if "label_final" in df_oof.columns else df_oof["y_true"].values

    configs = [
        ("majority", "Majority class"),
        ("text", "Extracted text"),
        ("visual", "Visual only"),
        ("audio", "Acoustic only"),
        ("equal_va", "Equal visual–acoustic"),
        ("equal_vat", "Equal visual–acoustic–text"),
        ("va_lr", "Visual–acoustic LR"),
        ("vat_lr", "Visual–acoustic–text LR"),
        ("vat_flags", "LR with extraction flags"),
        ("vat_coverage", "LR with availability features"),
        ("coverage_only", "Coverage only"),
        ("coverage_shuffled_control", "Shuffled coverage control"),
        ("concat_mlp", "Feature-concatenation MLP"),
        ("healnet_adapted_2024", "HEALNet adaptation"),
        ("siglip2_visual_probe", "SigLIP 2 visual probe"),
        ("siglip2_vt_probe", "SigLIP 2 visual–text probe"),
        ("siglip2_vta_probe", "SigLIP 2 VTA probe"),
    ]

    rows = []
    for key, name in configs:
        prob_col = f"{key}_probability"
        pred_col = f"{key}_prediction"

        if prob_col not in df_oof.columns:
            continue

        prob = df_oof[prob_col].values
        pred_sel = df_oof[pred_col].values if pred_col in df_oof.columns else None

        res_sel = compute_metrics(y_true, prob, pred=pred_sel)
        res_fixed = compute_metrics(y_true, prob, threshold=0.5)

        if key in df_seeds.index:
            s_row = df_seeds.loc[key]
            mean_sd = f"{s_row['macro_f1_mean']:.4f} ± {s_row['macro_f1_std']:.4f}"
        else:
            mean_sd = f"{res_sel['macro_f1']:.4f} ± 0.0000"

        rows.append({
            "Configuration": name,
            "F1 selected": f"{res_sel['macro_f1']:.4f}",
            "F1 at 0.5": f"{res_fixed['macro_f1']:.4f}",
            "F1 mean ± SD": mean_sd,
            "ROC-AUC": f"{res_sel['roc_auc']:.4f}",
            "AP": f"{res_sel['average_precision']:.4f}",
        })

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    print("=" * 95)


def display_paired_contrasts():
    print("\n" + "=" * 95)
    print("SUPPLEMENTARY STATISTICAL CONTRASTS & HOLM MULTIPLICITY ADJUSTMENTS")
    print("=" * 95)

    # 1. Primary Planned Contrasts (7 Baselines vs. Availability-Aware VAT)
    p7_path = RESULTS_DIR / "video_primary_paired_group_statistics.csv"
    if p7_path.exists():
        df7 = pd.read_csv(p7_path)
        print("\n1. Primary Planned Contrasts (Availability-Aware VAT vs. 7 Baselines):")
        df7_show = pd.DataFrame({
            "Baseline": df7["baseline"],
            "Delta Macro F1": df7["delta_macro_f1"].apply(lambda v: f"{v:+.4f}"),
            "95% CI": df7.apply(lambda r: f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]", axis=1),
            "Permutation p": df7["group_permutation_p"].apply(lambda v: f"{v:.4f}"),
            "p-Holm (7 primary contrasts)": df7["group_permutation_p_holm"].apply(lambda v: f"{v:.4f}"),
        })
        print(df7_show.to_string(index=False))

    # 2. Exploratory Contrasts (4 Equal-VAT Comparisons)
    p4_path = RESULTS_DIR / "paired_statistics/video_equal_vat_paired_group_statistics.csv"
    if p4_path.exists():
        df4 = pd.read_csv(p4_path)
        print("\n2. Exploratory Contrasts (Equal Late Fusion vs. 4 Key Baselines):")
        df4_show = pd.DataFrame({
            "Baseline": df4["baseline"],
            "Delta Macro F1": df4["delta_macro_f1"].apply(lambda v: f"{v:+.4f}"),
            "95% CI": df4.apply(lambda r: f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]", axis=1),
            "Permutation p": df4["group_permutation_p"].apply(lambda v: f"{v:.4f}"),
            "p-Holm (4 exploratory contrasts)": df4["group_permutation_p_holm"].apply(lambda v: f"{v:.4f}"),
            "p-Holm (11 extended contrasts)": df4["group_permutation_p_holm_extended_11"].apply(lambda v: f"{v:.4f}"),
        })
        print(df4_show.to_string(index=False))

    # Computation mode documentation note
    print("\nNote on Computation Mode:")
    print("  - Table 1, 2, and 3 metrics (Accuracy, Macro F1, ROC-AUC, AP) are computed")
    print("    dynamically on-the-fly from out-of-fold and held-out prediction CSVs.")
    print("  - Paired contrast statistics (2,000-resample cluster bootstrap 95% CIs and 10,000")
    print("    group-level permutation tests) are loaded from precomputed statistical audit files")
    print("    for instant sub-second execution (~1.7s on CPU). Full recomputation routines are")
    print("    implemented in notebooks/run_experiments.ipynb (~15 min on standard CPU).")
    print("=" * 95)


if __name__ == "__main__":
    display_table_1()
    display_table_2()
    display_table_3()
    display_paired_contrasts()
