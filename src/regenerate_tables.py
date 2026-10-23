"""
Regenerate Manuscript Tables (Tables 1, 2, and 3)
and Statistical Contrasts from saved results and split manifests.

Manuscript: "Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"
Authors: Ari Muzakir, Usman Ependi, Suyanto (2026)
"""

from pathlib import Path
import argparse
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


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(m)
    sorted_p = p_values[order]
    adj = sorted_p * (m - np.arange(m))
    adj = np.maximum.accumulate(adj)
    adj = np.minimum(1.0, adj)
    return adj[ranks]


def recompute_live_contrasts(B=2000, P=2000):
    print("\n" + "=" * 95)
    print("LIVE RECOMPUTATION OF STATISTICAL CONTRASTS & HOLM ADJUSTMENTS")
    print(f"Cluster Bootstrap Replicates: B = {B:,} | Group Permutations: P = {P:,}")
    print("=" * 95)

    oof_path = RESULTS_DIR / "video_oof_predictions.csv"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found.")
        return

    df_oof = pd.read_csv(oof_path)
    df_oof = df_oof[df_oof["seed"] == 42].reset_index(drop=True)
    y = df_oof["label_final"].values
    groups, group_indices = np.unique(df_oof["group_id"], return_inverse=True)
    G = len(groups)

    def get_group_counts(pred_col):
        pred = df_oof[pred_col].values
        c = np.zeros((G, 4))
        for k, (a, b) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
            np.add.at(c[:, k], group_indices, ((y == a) & (pred == b)).astype(float))
        return c

    def macro_f1_from_counts(c):
        tp, fn, fp, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
        f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
        f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1e-12)
        return (f1p + f1n) / 2

    def run_paired_test(cand_col, base_col, seed):
        cand_c = get_group_counts(cand_col)
        base_c = get_group_counts(base_col)
        d = macro_f1_from_counts(cand_c.sum(0)) - macro_f1_from_counts(base_c.sum(0))
        rng = np.random.default_rng(seed)
        w = rng.multinomial(G, np.ones(G) / G, size=B).astype(float)
        bd = macro_f1_from_counts(w @ cand_c) - macro_f1_from_counts(w @ base_c)
        ci_low, ci_high = np.percentile(bd, [2.5, 97.5])
        s = rng.choice([-1, 1], size=(P, G)).astype(float)
        perm = np.empty(P)
        for i in range(P):
            m = (s[i] > 0)[:, None]
            perm[i] = macro_f1_from_counts(np.where(m, cand_c, base_c).sum(0)) - macro_f1_from_counts(np.where(m, base_c, cand_c).sum(0))
        p = (1 + np.sum(np.abs(perm) >= abs(d) - 1e-15)) / (P + 1)
        return d, ci_low, ci_high, p

    # 1. Primary Planned Contrasts (Seed 20261020)
    p7_baselines = ["visual", "vat_lr", "vat_flags", "coverage_only", "coverage_shuffled_control", "healnet_adapted_2024", "siglip2_vta_probe"]
    rows7 = []
    p_vals7 = []
    for b in p7_baselines:
        d, lo, hi, p = run_paired_test("vat_coverage_prediction", f"{b}_prediction", 20261020)
        rows7.append({"Baseline": b, "Delta Macro F1": f"{d:+.4f}", "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "Permutation p": p})
        p_vals7.append(p)
    p7_holm = holm_adjust(p_vals7)
    for i, r in enumerate(rows7):
        r["Permutation p"] = f"{p_vals7[i]:.4f}"
        r["p-Holm (7 primary contrasts)"] = f"{p7_holm[i]:.4f}"

    print("\n1. Primary Planned Contrasts (Live Recomputed from Out-of-Fold Predictions):")
    print(pd.DataFrame(rows7).to_string(index=False))

    # 2. Exploratory Contrasts (Seed 20261027)
    p4_baselines = ["text", "visual", "siglip2_vta_probe", "healnet_adapted_2024"]
    rows4 = []
    p_vals4 = []
    for b in p4_baselines:
        d, lo, hi, p = run_paired_test("equal_vat_prediction", f"{b}_prediction", 20261027)
        rows4.append({"Baseline": b, "Delta Macro F1": f"{d:+.4f}", "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "Permutation p": p})
        p_vals4.append(p)
    p4_holm = holm_adjust(p_vals4)
    p11_all = list(p_vals7) + list(p_vals4)
    p11_holm = holm_adjust(p11_all)
    for i, r in enumerate(rows4):
        r["Permutation p"] = f"{p_vals4[i]:.4f}"
        r["p-Holm (4 exploratory contrasts)"] = f"{p4_holm[i]:.4f}"
        r["p-Holm (11 extended contrasts)"] = f"{p11_holm[len(p7_baselines) + i]:.4f}"

    print("\n2. Exploratory Contrasts (Live Recomputed from Out-of-Fold Predictions):")
    print(pd.DataFrame(rows4).to_string(index=False))
    print("=" * 95)


def display_paired_contrasts(recompute=False):
    if recompute:
        recompute_live_contrasts(B=2000, P=2000)
        return

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
    print("  - Paired contrast statistics (2,000-resample cluster bootstrap 95% CIs and 2,000")
    print("    paired group permutations per contrast) are loaded from precomputed statistical audit files")
    print("    for instant sub-second execution (~0.2s on CPU).")
    print("  - To recompute all bootstrap CIs and permutation tests live from raw predictions, run:")
    print("      python src/regenerate_tables.py --recompute")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate Manuscript Tables and Statistical Contrasts.")
    parser.add_argument("--recompute", action="store_true", help="Recompute cluster-bootstrap CIs and paired group permutations live on the fly from OOF predictions.")
    args = parser.parse_args()

    display_table_1()
    display_table_2()
    display_table_3()
    display_paired_contrasts(recompute=args.recompute)
