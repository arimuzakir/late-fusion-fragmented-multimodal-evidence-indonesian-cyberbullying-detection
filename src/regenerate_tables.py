"""
Regenerate Manuscript Tables (Tables 1, 2, and 3)
from saved prediction and manifest artifacts.

Manuscript: "Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"
Authors: Ari Muzakir, Usman Ependi, Suyanto (2026)
"""

import sys
from pathlib import Path
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
    print("\n" + "=" * 90)
    print("TABLE 1: POPULATION AND EVIDENCE AVAILABILITY ACCOUNTING")
    print("=" * 90)
    
    # Text comments
    comment_splits = pd.read_csv(ARTIFACTS_DIR / "comment_splits.csv")
    print("1. Social Comments Corpus:")
    print(f"   Total canonical comments: {len(comment_splits):,}")
    for split, count in comment_splits["split"].value_counts().items():
        sub = comment_splits[comment_splits["split"] == split]
        pos = (sub["label_final"] == 1).sum()
        neg = (sub["label_final"] == 0).sum()
        print(f"   - {split:10}: {count:,} comments ({neg:,} non-bullying, {pos:,} bullying)")
        
    # Video segments
    video_manifest = pd.read_csv(ARTIFACTS_DIR / "video_primary_manifest.csv")
    print("\n2. Video Segments Sample (Eligible Primary, N = 3,951):")
    total_vid = len(video_manifest)
    pos_vid = (video_manifest["label_final"] == 1).sum()
    neg_vid = (video_manifest["label_final"] == 0).sum()
    print(f"   Total eligible segments: {total_vid:,} ({neg_vid:,} non-bullying, {pos_vid:,} bullying)")
    print("   Evidence Availability Breakdown:")
    for (ocr, stt), n in video_manifest.groupby(["ocr_ok", "stt_ok"]).size().items():
        status = "Both OCR + STT" if ocr and stt else ("OCR only" if ocr else ("STT only" if stt else "Neither (Zero text)"))
        pct = (n / total_vid) * 100
        print(f"   - {status:25}: {n:5,d} ({pct:5.1f}%)")
    print("=" * 90)


def display_table_2():
    print("\n" + "=" * 90)
    print("TABLE 2: HELD-OUT COMMENT RESULTS (1,226 test comments, Seed 42)")
    print("=" * 90)
    
    metrics_path = RESULTS_DIR / "comment_metrics.csv"
    if not metrics_path.exists():
        print(f"Missing metrics file: {metrics_path}")
        return
        
    df = pd.read_csv(metrics_path)
    
    models = [
        ("tfidf_word_char_lr", "Word/character TF-IDF + LR"),
        ("indobert_p2", "IndoBERT-base-p2"),
        ("indobertweet", "IndoBERTweet"),
        ("mbert", "Multilingual BERT"),
        ("equal_text_ensemble", "Equal text ensemble")
    ]
    
    rows = []
    for model_key, model_name in models:
        sub = df[df["model"] == model_key]
        if len(sub) == 0:
            continue
        # Take seed 42 row (first occurrence if multiple seeds)
        r = sub.iloc[0]
        rows.append({
            "Model": model_name,
            "Acc.": f"{r['accuracy']:.4f}",
            "F1": f"{r['macro_f1']:.4f}",
            "AUC": f"{r['roc_auc']:.4f}",
            "AP": f"{r['average_precision']:.4f}"
        })
        
    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    print("=" * 90)


def display_table_3():
    print("\n" + "=" * 95)
    print("TABLE 3: CONTROLLED VIDEO COMPARISON (3,951 video windows, 4-fold grouped OOF)")
    print("=" * 95)
    
    oof_path = RESULTS_DIR / "video_oof_predictions.csv"
    seed_summary_path = RESULTS_DIR / "video_seed_summary.csv"
    
    if not oof_path.exists() or not seed_summary_path.exists():
        print("Missing required video results files.")
        return
        
    df_oof = pd.read_csv(oof_path)
    df_seeds = pd.read_csv(seed_summary_path).set_index("model")
    
    # Primary seed 42 OOF predictions
    df_s42 = df_oof[df_oof["seed"] == 42].sort_values("master_id")
    
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
        ("siglip2_vta_probe", "SigLIP 2 VTA probe")
    ]
    
    rows = []
    for key, name in configs:
        prob_col = f"{key}_probability"
        pred_col = f"{key}_prediction"
        
        if prob_col not in df_s42.columns:
            continue
            
        prob = df_s42[prob_col].to_numpy()
        pred = df_s42[pred_col].to_numpy()
        y_true = df_s42["label_final"].to_numpy()
        
        # Selected cutoff metrics
        res_sel = compute_metrics(y_true, prob, pred=pred)
        # Fixed 0.50 cutoff metrics
        res_fixed = compute_metrics(y_true, prob, threshold=0.5)
        
        # 3-seed Mean +- SD
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
            "AP": f"{res_sel['average_precision']:.4f}"
        })
        
    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    print("=" * 95)


def display_paired_contrasts():
    print("\n" + "=" * 90)
    print("SUPPLEMENTARY: PAIRED DIFFERENCES & HOLM MULTIPLICITY CORRECTION")
    print("=" * 90)
    p_path = RESULTS_DIR / "paired_statistics/video_equal_vat_paired_group_statistics.csv"
    if p_path.exists():
        df = pd.read_csv(p_path)
        cols = ["baseline", "delta_macro_f1", "ci_low", "ci_high", "group_permutation_p", "group_permutation_p_holm", "group_permutation_p_holm_extended_11"]
        print(df[cols].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    display_table_1()
    display_table_2()
    display_table_3()
    display_paired_contrasts()
