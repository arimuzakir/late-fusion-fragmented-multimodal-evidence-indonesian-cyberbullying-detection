# Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Official reproduction package and benchmark artifacts for the manuscript:

> **Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection**  
> Ari Muzakir, Usman Ependi, Suyanto (2026)  
> Universitas Bina Darma, Palembang, Indonesia  
> Contact: `arimuzakir@binadarma.ac.id`

---

## 1. Overview and Reproducibility Tiers

This repository provides evaluation manifests, out-of-fold predictions, statistical tests, and scripts to reproduce all findings reported in the manuscript. We distinguish two reproducibility tiers:

- **Level 1 (Evaluation and Metric Reproduction — CPU)**  
  Recomputes all metrics, confusion matrices, cluster-bootstrap confidence intervals (95% CI), hypothesis tests, Tables 1–3, and empirical Figures 4–8 directly from stored out-of-fold predictions.  
  *Execution time: ~3.4 seconds total on a standard CPU (verified in a clean clone environment).*  
  *Scope note: Figures 1–3 are conceptual method and architecture diagrams bundled as publication assets from manuscript OpenXML media, whereas Figures 4–8 are programmatically regenerated from data.*

- **Level 2 (End-to-End Retraining — GPU Required)**  
  Retrains unimodal encoders and late fusion models from source data using `notebooks/run_experiments.ipynb`. Due to privacy protections on human subjects and repository size limits, raw social media video files, raw audio WAVs, and large pretrained checkpoints (475 MB per model) are maintained separately and not bundled into this Git release.

---

## 2. Quickstart

### Setup
```bash
git clone https://github.com/arimuzakir/late-fusion-fragmented-multimodal-evidence-indonesian-cyberbullying-detection.git
cd late-fusion-fragmented-multimodal-evidence-indonesian-cyberbullying-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce Tables (Tables 1, 2, and 3)
```bash
python src/regenerate_tables.py
```

### Reproduce Figures (Figures 4 through 8)
```bash
python src/regenerate_figures.py
```

---

## 3. Dataset Summary and Protocol

### Population Accounting (Table 1)
| Population | Records | Negative (Non-Bullying) | Positive (Bullying) |
|---|:---:|:---:|:---:|
| **Master source collection** | 13,307 | 6,057 | 7,250 |
| **Comment source population** | 8,906 | 4,260 | 4,646 |
| **Eligible high-agreement comments** | 8,196 | 3,799 | 4,397 |
| **Comment test population** | 1,226 | 555 | 671 |
| **Video source population** | 4,401 | 1,797 | 2,604 |
| **Aligned video evaluation population** | 3,951 | 1,627 | 2,324 |

*Video Extraction Availability Strata (N = 3,951 eligible 10s windows, stride 3s)*:
- Both OCR and STT extracted: 1,275 (32.3%)
- STT transcript only: 1,231 (31.2%)
- OCR text only: 760 (19.2%)
- Zero text extracted (Neither): 685 (17.3%)

### Annotation Protocol and Reliability
- **Fleiss' Kappa ($\kappa = 0.8853$)**: Evaluated across the entire master collection of $N = 13,307$ records by five independent human annotators (`anotator_1` through `anotator_5`), demonstrating almost perfect agreement under Landis & Koch (1977).
- **Annotator Pairwise Cohen's Kappa (mean $\kappa = 0.8851$)**: Unweighted mean of all 10 bilateral annotator pairs across the same master corpus ($N = 13,307$).
- **Inter-Expert Agreement ($\kappa = 0.9075$)**: Bilateral Cohen's kappa between two independent domain experts (Expert 1: Indonesian Forensic Linguist; Expert 2: Child Developmental Psychologist) evaluated on a stratified gold-standard subsample of $n = 1,331$ records (10.0% of the corpus). This metric reflects inter-expert agreement on the audit sample and is not a post-adjudication score for the full segment corpus.
- **Documentary Evidence**: Annotation guidelines and phase-specific gold standards are archived in `docs/annotation_guidelines.docx`, `docs/gold_standard_phase_1.pdf`, and `docs/gold_standard_phase_2.pdf`. Master segment labels reflect majority consensus ($\ge 3/5$) without circular performance-guided relabeling.

### Disjoint Partitioning and Role Accounting
Within each 4-fold group partition, observations strictly avoid leakage across roles:
| Outer Fold | Fit Role | Early Stop Role | Meta-Fit Role | Calibration Role | Test Role (OOF) | Total Windows |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 1,621 | 266 | 404 | 423 | 1,237 | 3,951 |
| **Fold 1** | 1,889 | 328 | 466 | 495 | 773 | 3,951 |
| **Fold 2** | 1,890 | 314 | 466 | 476 | 805 | 3,951 |
| **Fold 3** | 1,677 | 293 | 420 | 425 | 1,136 | 3,951 |
| **Total** | **7,077** | **1,201** | **1,756** | **1,819** | **3,951** | **15,804** |

Decision thresholds (`thresholds.json`) were selected exclusively on the `calibration` split. Intermediate `meta_fit` and `calibration` predictions were reconstructed from frozen representations and fitted meta-models without retraining. Full verification across all 360 model-seed-fold-role configurations is documented in [`artifacts/recovery_validation.csv`](artifacts/recovery_validation.csv) with zero threshold discrepancy.

---

## 4. Manuscript Mapping

- **Table 1**: Population and evidence availability accounting (`artifacts/comment_splits.csv`, `artifacts/video_alignment_audit.csv`, `artifacts/video_primary_manifest.csv`).
- **Table 2**: Held-out comment model comparison (`results/comment_metrics.csv`).
- **Table 3**: Controlled multimodal video comparison across 17 models and 3 random seeds (`results/video_oof_predictions.csv`, `results/video_seed_summary.csv`).
- **Supplementary Statistical Contrasts**:
  - *Primary Planned Contrasts (7 Baselines vs. Availability-Aware VAT)*: `results/video_primary_paired_group_statistics.csv` with `p-Holm (7 primary contrasts)`.
  - *Exploratory Contrasts (Equal Late Fusion vs. 4 Key Baselines)*: `results/paired_statistics/video_equal_vat_paired_group_statistics.csv` with `p-Holm (4 exploratory contrasts)` and `p-Holm (11 extended contrasts)`.
- **Figures 1–3 (Method Assets)**: System architecture, corpus creation, and protocol diagrams (`figures/fig_01_*` to `fig_03_*`).
- **Figures 4–8 (Regenerated Empirical Plots)**: Comment comparison, fusion differences forest plot, seed stability, extraction availability, and threshold sensitivity (`figures/fig_04_*` to `fig_08_*`).

---

## 5. Repository Structure

```
├── configs/               # Experiment parameters, environment info, model revisions
├── artifacts/             # Manifest splits, video roles, recovery validation, feature embeddings
├── notebooks/             # Workflow notebook for training and experiment execution
├── src/                   # Table and figure regeneration scripts, vendor modules
├── results/               # Out-of-fold sample predictions, aggregate metrics, paired stats
├── figures/               # Eight publication figures matching manuscript V6
├── docs/                  # Labeling guidelines, gold standards, statistical addenda
├── LICENSE                # MIT License and third-party notices
├── CITATION.cff           # Citation metadata
└── requirements.txt       # Python dependencies
```

---

## 6. Citation and License

If you use this benchmark, code, or experimental artifacts, please cite:

```bibtex
@article{muzakir2026latefusion,
  title={Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection},
  author={Muzakir, Ari and Ependi, Usman and Suyanto},
  year={2026}
}
```

This software is released under the MIT License. Adapted components from HEALNet retain their Apache 2.0 license (see `src/vendor/HEALNET_LICENSE`).
