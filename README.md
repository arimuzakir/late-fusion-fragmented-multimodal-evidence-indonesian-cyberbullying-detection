# Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Official reproduction package and benchmark artifacts for the manuscript:

> **Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection**  
> Ari Muzakir, Usman Ependi, Suyanto (2026)  
> Universitas Bina Darma, Palembang, Indonesia  
> Contact: `arimuzakir@binadarma.ac.id`

---

## 1. Overview and Reproducibility Tiers

This repository provides evaluation manifests, out-of-fold predictions, statistical tests, and scripts to reproduce all findings reported in the manuscript. We distinguish two reproducibility tiers:

- **Level 1 (Evaluation and Reproduction — CPU, Self-Contained)**  
  Recomputes all metrics, confusion matrices, cluster-bootstrap confidence intervals (95% CI), hypothesis tests, Tables 1–3, and Figures 4–8 directly from stored out-of-fold predictions.  
  *Runtime: < 30 seconds on a standard CPU.*

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
- **Social Comments**: 8,196 comments (Train: 5,741; Validation: 1,229; Test: 1,226).
- **Video Windows (N = 3,951 eligible 10s segments, stride 3s)**:
  - Both OCR and STT: 1,275 (32.3%)
  - STT transcript only: 1,231 (31.2%)
  - OCR text only: 760 (19.2%)
  - Zero text extracted (Neither): 685 (17.3%)

### Labeling and Annotation
- **Segment Labeling**: Five independent annotators labeled each 10-second window (Fleiss' $\kappa = 0.8853$; pairwise Cohen's $\kappa = 0.8851$). Final labels reflect majority consensus ($\ge 3/5$).
- **Expert Adjudication**: Domain experts evaluated incident-level parent video context rather than performing an exhaustive independent validation of every individual segment. Post-adjudication agreement reached $\kappa = 0.9075$.

### Disjoint Partitioning Roles
Within each 4-fold group partition, observations strictly avoid leakage across roles:
| Outer Fold | Fit | Early Stop | Meta-Fit | Calibration | Test (OOF) | Total |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 1,621 | 266 | 404 | 423 | 1,237 | 3,951 |
| **Fold 1** | 1,889 | 328 | 466 | 495 | 773 | 3,951 |
| **Fold 2** | 1,890 | 314 | 466 | 476 | 805 | 3,951 |
| **Fold 3** | 1,677 | 293 | 420 | 425 | 1,136 | 3,951 |
| **Total** | **7,077** | **1,201** | **1,756** | **1,819** | **3,951** | **15,804** |

Decision thresholds (`thresholds.json`) were selected exclusively on the `calibration` split. Intermediate `meta_fit` and `calibration` predictions were reconstructed from frozen representations and fitted meta-models, reproducing all recorded thresholds without retraining.

---

## 4. Manuscript Mapping

- **Table 1**: Population and evidence availability accounting (`artifacts/comment_splits.csv`, `artifacts/video_primary_manifest.csv`).
- **Table 2**: Held-out comment model comparison (`results/comment_metrics.csv`).
- **Table 3**: Controlled multimodal video comparison across 17 models and 3 random seeds (`results/video_oof_predictions.csv`, `results/video_seed_summary.csv`).
- **Supplementary Statistical Contrasts**: Paired group permutation tests with 7-contrast and 11-contrast Holm adjustments (`results/paired_statistics/video_equal_vat_paired_group_statistics.csv`).
- **Figures 1–3**: System architecture, corpus creation, and protocol diagrams (`figures/fig_01_*` to `fig_03_*`).
- **Figures 4–8**: Comment comparison, fusion differences forest plot, seed stability, extraction availability, and threshold sensitivity (`figures/fig_04_*` to `fig_08_*`).

---

## 5. Repository Structure

```
├── configs/               # Experiment parameters, environment info, model revisions
├── artifacts/             # Manifest splits, video roles, anonymized feature embeddings
├── notebooks/             # Workflow notebook for training and experiment execution
├── src/                   # Table and figure regeneration scripts, vendor modules
├── results/               # Out-of-fold sample predictions, aggregate metrics, paired stats
├── figures/               # Final publication figures matching manuscript V6
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
