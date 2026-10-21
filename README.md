# Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

This repository contains the official benchmark artifacts, out-of-fold predictions, evaluation manifests, and reproduction scripts for the manuscript:

> **"Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"**  
> *Ari Muzakir, Usman Ependi, Suyanto* (2026)  
> Universitas Bina Darma, Palembang, Indonesia  
> Corresponding author: `arimuzakir@binadarma.ac.id`

---

## 1. Reproducibility Scope and Tiers

To ensure complete scientific transparency, we distinguish between two reproduction tiers:

### Level 1: Metric, Table, and Figure Reproduction (Supported Out-of-the-Box)
- **Scope**: Re-evaluates all performance metrics, confusion matrices, source-group cluster-bootstrap confidence intervals, paired permutation hypothesis tests with Holm multiplicity adjustments, and publication figures directly from stored, frozen out-of-fold predictions and split manifests.
- **Hardware Requirement**: Standard CPU (laptop/workstation).
- **Runtime**: < 30 seconds.
- **Command**:
  ```bash
  python src/regenerate_tables.py
  python src/regenerate_figures.py
  ```

### Level 2: End-to-End Retraining from Raw Evidence (Compute & Storage Intensive)
- **Scope**: Re-extracts multimodal keyframes, fine-tunes text and visual transformer encoders (`IndoBERT-base-p2`, `IndoBERTweet`, `mBERT`, and `EfficientNet-B0`) per outer fold and training seed, fits meta-learners, and conducts full hyperparameter sweeps.
- **Hardware Requirement**: NVIDIA GPU with >= 24GB VRAM (e.g., NVIDIA L4 or A100).
- **Raw Data Prerequisites**: Raw video media files, full extracted audio WAVs, raw frame caches, and base pretrained checkpoints from Hugging Face Hub.
- **Entrypoint**: `notebooks/run_experiments.ipynb` (runnable on Google Colab or on-premise GPU cluster).

---

## 2. Dataset Population and Evidence Availability (Table 1)

The evaluation spans two distinct Indonesian corpora under strict source-disjoint partitioning:

| Corpus / Split | Observations | Non-Bullying | Bullying | Evidence Availability Conditions |
|---|---|---|---|---|
| **Social Comments (Train)** | 5,741 | 2,668 | 3,073 | 100% Unimodal text |
| **Social Comments (Validation)** | 1,229 | 576 | 653 | 100% Unimodal text |
| **Social Comments (Held-out Test)** | 1,226 | 555 | 671 | 100% Unimodal text |
| **Video Windows (Eligible Primary)** | 3,951 | 1,627 | 2,324 | Multimodal video segments (10s window, stride 3s) |
| *— Both OCR and STT extracted* | 1,275 (32.3%) | — | — | Keyframe + Audio vector + OCR text + STT text |
| *— STT transcript only* | 1,231 (31.2%) | — | — | Keyframe + Audio vector + STT text (no visual text) |
| *— OCR text only* | 760 (19.2%) | — | — | Keyframe + Audio vector + OCR text (no audio transcript) |
| *— Neither (Zero text extracted)* | 685 (17.3%) | — | — | Keyframe + Audio vector only (neutral fallback text) |

### Annotation Protocol and Governance
- **Segment Labeling**: Five independent native Indonesian annotators labeled all text comments and 10-second video segments. Final labels reflect majority consensus ($\ge 3/5$). Annotator agreement achieved high reliability (Fleiss' $\kappa = 0.8853$; Cohen's $\kappa = 0.8851$).
- **Expert Video Adjudication**: Independent expert adjudication evaluated contextual bullying incidents at the root video level. Expert review served as contextual oversight for parent sources and is not conflated with automated re-annotation of all sliding windows.

---

## 3. Disjoint Partitioning Roles Across Video Folds

To eliminate data leakage, video segments are grouped by canonical URL, author, and perceptual image hash. Within each training partition, observations are strictly partitioned into four non-overlapping roles:

| Outer Fold | Fit Role | Early Stop Role | Meta-Fit Role | Calibration Role | Held-out Test Role | Total Windows |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 1,621 | 266 | 404 | 423 | 1,237 | 3,951 |
| **Fold 1** | 1,889 | 328 | 466 | 495 | 773 | 3,951 |
| **Fold 2** | 1,890 | 314 | 466 | 476 | 805 | 3,951 |
| **Fold 3** | 1,677 | 293 | 420 | 425 | 1,136 | 3,951 |
| **Total** | 7,077 | 1,201 | 1,756 | 1,819 | **3,951 (OOF)** | 15,804 |

> **Technical Note on Threshold Selection & Meta-Fit Predictions**:  
> In accordance with rigorous scientific reporting, decision cutoffs (`thresholds.json`) were determined exclusively on the reserved `calibration` role. Predictions on the `meta_fit` and `calibration` roles were reconstructed directly from the frozen branch representations and fitted meta-models without retraining, and verified to match the recorded thresholds across all 12 seed-fold combinations.

---

## 4. Mapping of Artifacts to Manuscript Tables and Figures

### Manuscript Tables
- **Table 1 (Population Accounting)**: Re-generated by `python src/regenerate_tables.py` using `artifacts/comment_splits.csv` and `artifacts/video_primary_manifest.csv`.
- **Table 2 (Held-Out Comment Results)**: Re-generated from `results/comment_metrics.csv` (TF-IDF + LR, IndoBERT-p2, IndoBERTweet, mBERT, and Equal Ensemble on 1,226 test comments).
- **Table 3 (Controlled Video Comparison)**: Re-generated from `results/video_oof_predictions.csv` and `results/video_seed_summary.csv` (17 configurations across Seeds 42, 2026, and 2027).
- **Supplementary Text Contrasts**: Re-generated from `results/paired_statistics/video_equal_vat_paired_group_statistics.csv` (Holm-adjusted permutation hypothesis tests across 4-contrast and extended 11-contrast families).

### Manuscript Figures
All eight figures in `figures/` match the final published manuscript V6:
- **`figures/fig_01_dataset_composition.png`** (Fig. 1): Data composition and evidence availability flowchart.
- **`figures/fig_02_fusion_architecture.png`** (Fig. 2): Trimodal video branches and decision-level fusion architecture.
- **`figures/fig_03_evaluation_protocol.png`** (Fig. 3): Source-disjoint fitting, selection, calibration, and test roles.
- **`figures/fig_04_comment_comparison.png`** (Fig. 4): Held-out comment model comparison dot plot.
- **`figures/fig_05_fusion_differences.png`** (Fig. 5): Source-paired differences in video macro F1 (Forest plot with 11-contrast Holm adjustments: HEALNet $p=0.0165$, visual $p=0.2599$, SigLIP 2 VTA $p=0.3463$, text $p=1.0000$).
- **`figures/fig_06_seed_stability.png`** (Fig. 6): Video macro F1 trajectory across three training seeds (42, 2026, 2027).
- **`figures/fig_07_extraction_availability.png`** (Fig. 7): Video performance broken down by OCR and STT availability strata.
- **`figures/fig_08_threshold_sensitivity.png`** (Fig. 8): Video macro F1 comparison at fixed (0.50) versus selected decision cutoffs.

---

## 5. Repository Structure

```
├── .gitignore
├── LICENSE                          # MIT License (Ari Muzakir, Usman Ependi, Suyanto)
├── CITATION.cff                     # Citation Metadata
├── README.md                        # Documentation and replication guide
├── requirements.txt                 # Pinned Python package dependencies
├── configs/
│   ├── experiment_config.json       # Hyperparameters, seeds, batch sizes, threshold grid
│   ├── environment.json             # Execution environment, CUDA/GPU specifications
│   ├── model_revisions.json         # Pinned Hugging Face checkpoint commit SHAs
│   ├── healnet_provenance.json      # Official HEALNet commit hash and adaptation specs
│   └── siglip2_provenance.json      # SigLIP 2 model revision and extraction protocol
├── artifacts/
│   ├── comment_splits.csv           # Text comment split allocations (8,196 comments)
│   ├── video_roles.csv              # 4-fold disjoint role assignments (15,804 records)
│   ├── video_primary_manifest.csv   # Metadata for 3,951 eligible video windows
│   ├── video_input_lock.json        # Integrity hash locks for input features
│   ├── video_alignment_audit.csv    # Alignment audit across 4,401 candidate windows
│   ├── video_excluded_alignment.csv # Explicit records of 450 excluded misaligned windows
│   └── siglip2_frozen_features.npz  # Pre-computed SigLIP 2 features (anonymized IDs)
├── notebooks/
│   └── run_experiments.ipynb        # Cleaned experiment workflow notebook
├── src/
│   ├── regenerate_tables.py         # Verification and reproduction of Tables 1, 2, and 3
│   └── regenerate_figures.py        # Programmatic reproduction of publication figures
├── results/
│   ├── comment_predictions.csv      # Test predictions for all text models
│   ├── comment_metrics.csv          # Aggregate metrics on held-out text comments
│   ├── comment_cluster_ci.csv       # 95% Cluster-bootstrap CIs for text models
│   ├── video_oof_predictions.csv    # Out-of-fold sample predictions across all video models
│   ├── video_seed_metrics.csv       # Out-of-fold metrics across seeds 42, 2026, and 2027
│   ├── video_seed_summary.csv       # Mean ± SD across seeds
│   ├── video_cluster_ci.csv         # 95% Cluster-bootstrap CIs for video configurations
│   ├── paired_statistics/           # Paired permutation tests and Holm adjustments
│   ├── text/                        # TF-IDF model and text predictions
│   └── video/                       # Stored fold-level outputs, thresholds, and models
├── figures/                         # Eight publication figures matching the manuscript
└── docs/
    ├── annotation_guidelines.docx   # Complete data labeling guidelines
    ├── gold_standard_phase_1.pdf    # Phase 1 annotation gold standard documentation
    ├── gold_standard_phase_2.pdf    # Phase 2 annotation gold standard documentation
    ├── statistical_addendum.md      # Methodological notes on Holm multiplicity testing
    └── methods_and_claim_boundaries.md
```

---

## 6. Quickstart: Replicating Tables and Figures

### 1. Clone the repository and setup environment
```bash
git clone https://github.com/arimuzakir/late-fusion-fragmented-multimodal-evidence-indonesian-cyberbullying-detection.git
cd late-fusion-fragmented-multimodal-evidence-indonesian-cyberbullying-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Re-compute Tables 1, 2, and 3
```bash
python src/regenerate_tables.py
```

### 3. Re-generate Figures 4 through 8
```bash
python src/regenerate_figures.py
```

---

## 7. License and Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Third-party components (`HEALNet`, `SigLIP 2`, `IndoBERT`, `IndoBERTweet`) retain their respective author licenses as noted in `configs/`.
