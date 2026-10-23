# Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Research repository for the manuscript “Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection.”

## Overview

This project evaluates Indonesian cyberbullying detection under fragmented multimodal evidence across social media comments and video windows. The study benchmarks unimodal text, visual, and acoustic representations against late fusion architectures, including equal fusion, availability-aware fusion with modality flags, HEALNet adaptation (NeurIPS 2024), and SigLIP 2 vision-language probes (Google 2025).

The primary evaluation protocol spans two corpora under strict source-disjoint partitioning:
- 8,196 eligible text comments (Train: 5,741; Validation: 1,229; Test: 1,226).
- 3,951 aligned video segments evaluated through 4-fold grouped nested cross-validation with non-overlapping fit, early stopping, meta-fitting, calibration, and test roles across three training seeds (42, 2026, 2027).

## Experimental configurations

### Unimodal baselines
- Majority class
- Extracted text (IndoBERT-base-p2 fine-tuned per fold)
- Visual only (EfficientNet-B0 fine-tuned per fold)
- Acoustic only (Ridge classifier on audio feature vectors)

### Late fusion and control models
- Equal visual–acoustic (Equal VA)
- Equal visual–acoustic–text (Equal VAT)
- Visual–acoustic logistic regression (VA LR)
- Visual–acoustic–text logistic regression (VAT LR)
- LR with extraction flags (VAT + binary OCR/STT flags)
- LR with availability features (Availability-aware VAT)
- Coverage-only control (Modality presence features without content)
- Shuffled coverage control (Permuted coverage features)
- Feature-concatenation MLP (Concatenated branch features)

### Representation and architecture baselines
- HEALNet adaptation (Official NeurIPS 2024 architecture adapted to frozen representations)
- SigLIP 2 visual probe (Pretrained vision encoder + linear probe)
- SigLIP 2 visual–text probe (Multimodal vision-language probe)
- SigLIP 2 VTA probe (Vision-text probe with acoustic feature augmentation)

## Final reported results

### Table 1: Population and evidence accounting
| Population | Records | Negative (Non-Bullying) | Positive (Bullying) |
|---|:---:|:---:|:---:|
| Master source collection | 13,307 | 6,057 | 7,250 |
| Comment source population | 8,906 | 4,260 | 4,646 |
| Eligible high-agreement comments | 8,196 | 3,799 | 4,397 |
| Comment test population | 1,226 | 555 | 671 |
| Video source population | 4,401 | 1,797 | 2,604 |
| Aligned video evaluation population | 3,951 | 1,627 | 2,324 |

*Video extraction availability breakdown (N = 3,951)*: Both OCR and STT: 1,275 (32.3%); STT only: 1,231 (31.2%); OCR only: 760 (19.2%); Zero text extracted: 685 (17.3%).

### Table 2: Held-out comment results (1,226 test comments, Seed 42)
| Model | Accuracy | Macro F1 | ROC-AUC | Average Precision |
|---|:---:|:---:|:---:|:---:|
| Word/character TF-IDF + LR | 0.9364 | 0.9359 | 0.9870 | 0.9904 |
| IndoBERT-base-p2 | 0.9608 | 0.9604 | 0.9920 | 0.9944 |
| IndoBERTweet | 0.9527 | 0.9524 | 0.9849 | 0.9807 |
| Multilingual BERT | 0.9600 | 0.9598 | 0.9880 | 0.9923 |
| Equal text ensemble | 0.9666 | 0.9663 | 0.9932 | 0.9955 |

### Table 3: Controlled video comparison (3,951 video windows, 4-fold grouped OOF)
| Configuration | F1 selected | F1 at 0.50 | F1 mean ± SD | ROC-AUC | AP |
|---|:---:|:---:|:---:|:---:|:---:|
| Majority class | 0.3704 | 0.3704 | 0.3704 ± 0.0000 | 0.4701 | 0.5721 |
| Extracted text | 0.5589 | 0.5615 | 0.5560 ± 0.0067 | 0.5862 | 0.6610 |
| Visual only | 0.5281 | 0.4911 | 0.5228 ± 0.0141 | 0.5276 | 0.6082 |
| Acoustic only | 0.5073 | 0.3964 | 0.5073 ± 0.0000 | 0.5082 | 0.6010 |
| Equal visual–acoustic | 0.5228 | 0.4036 | 0.5174 ± 0.0073 | 0.5232 | 0.6032 |
| Equal visual–acoustic–text | 0.5615 | 0.5176 | 0.5559 ± 0.0085 | 0.5842 | 0.6553 |
| Visual–acoustic LR | 0.5164 | 0.3734 | 0.5081 ± 0.0104 | 0.5258 | 0.6063 |
| Visual–acoustic–text LR | 0.5120 | 0.3770 | 0.5084 ± 0.0059 | 0.5173 | 0.5993 |
| LR with extraction flags | 0.5135 | 0.3858 | 0.5142 ± 0.0127 | 0.5210 | 0.6060 |
| LR with availability features | 0.5081 | 0.4187 | 0.5174 ± 0.0103 | 0.5169 | 0.6026 |
| Coverage only | 0.4939 | 0.3704 | 0.4939 ± 0.0000 | 0.5081 | 0.5979 |
| Shuffled coverage control | 0.5097 | 0.3952 | 0.5060 ± 0.0034 | 0.5205 | 0.6048 |
| Feature-concatenation MLP | 0.5067 | 0.5065 | 0.5159 ± 0.0089 | 0.5079 | 0.5957 |
| HEALNet adaptation | 0.5076 | 0.5054 | 0.5159 ± 0.0088 | 0.5186 | 0.6055 |
| SigLIP 2 visual probe | 0.5302 | 0.5269 | 0.5302 ± 0.0000 | 0.5445 | 0.6369 |
| SigLIP 2 visual–text probe | 0.5278 | 0.5381 | 0.5278 ± 0.0000 | 0.5598 | 0.6465 |
| SigLIP 2 VTA probe | 0.5304 | 0.5415 | 0.5304 ± 0.0000 | 0.5632 | 0.6476 |

Equal late fusion (VAT) achieves 0.5615 macro F1 at its selected cutoff ($\tau^* = 0.575$ on Seed 42). When tested under source-paired group permutations, equal late fusion shows significant improvement over HEALNet adaptation (Δ = +0.0539, 11-contrast Holm p = 0.0165), while differences against visual-only (p = 0.2599), SigLIP 2 VTA probe (p = 0.3463), and extracted text (p = 1.0000) are not statistically significant after multiplicity adjustment.

## Repository structure

late-fusion-fragmented-multimodal-evidence/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
├── artifacts/
├── src/
│   ├── regenerate_tables.py
│   ├── regenerate_figures.py
│   └── vendor/
├── results/
│   ├── paired_statistics/
│   ├── text/
│   └── video/
├── figures/
├── notebooks/
└── docs/

## Reproduction

### Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce tables and statistical tests
```bash
# Default mode: instant evaluation on-the-fly with precomputed contrast statistics (~0.2s)
python src/regenerate_tables.py

# Live Monte Carlo mode: recomputes 2,000 cluster-bootstraps and 2,000 group-permutations live (~0.8s)
python src/regenerate_tables.py --recompute
```
This script computes Table 1, 2, and 3 evaluation metrics dynamically on-the-fly from stored out-of-fold and held-out prediction CSVs. Paired contrast statistics (2,000-sample cluster-bootstrap 95% CIs and 2,000 paired group permutations per contrast, matching `configs/experiment_config.json`) are loaded by default from precomputed statistical audit files for instant execution (~0.2s on CPU). Passing `--recompute` executes the live vectorized resampling directly from prediction columns in ~0.8s on CPU. Full end-to-end retraining routines are provided in `notebooks/run_experiments.ipynb`.

### Reproduce empirical figures
```bash
python src/regenerate_figures.py
```
This script regenerates the empirical plots:
- `fig_04_comment_comparison.png`: Held-out comment dot plot
- `fig_05_fusion_differences.png`: Source-paired differences forest plot
- `fig_06_seed_stability.png`: Multi-seed stability trajectory
- `fig_07_extraction_availability.png`: Performance across OCR/STT strata
- `fig_08_threshold_sensitivity.png`: Decision cutoff comparison (0.50 vs selected cutoff)

Figures 1, 2, and 3 are conceptual diagrams (corpus composition, fusion architecture, and nested CV protocol) extracted from the manuscript OpenXML media and provided as reference assets under `figures/`.

### End-to-end retraining
To re-extract features and retrain encoders from scratch on GPU hardware, execute:
`notebooks/run_experiments.ipynb`

## Checkpoints and data provenance

Out-of-fold sample predictions, decision cutoffs, and feature representations are preserved under `results/` and `artifacts/`. Intermediate per-sample predictions for inner roles are archived per seed and fold:
- `results/video/seed_{seed}/fold_{fold}/calibration_predictions.csv` (role: `threshold_selection`)
- `results/video/seed_{seed}/fold_{fold}/meta_fit_predictions.csv` (role: `meta_fit`)
- `results/video/seed_{seed}/fold_{fold}/test_predictions.csv` (role: `test`)

Their integrity was independently verified without refitting:
- `artifacts/threshold_selection_recovery_validation.csv`: Audits operating cutoffs on the validation split against historical `thresholds.json` (exact match, 0 discrepancies across 180 runs).
- `artifacts/meta_fit_recovery_validation.csv`: Audits input/output feature alignment and frozen-branch representations on the meta-fit split (180 runs, 0 refitting).

**Reproduction boundaries for raw media and extraction pipelines:**
Video speech-to-text transcripts (Whisper) and on-screen text extractions (OCR), along with 16-dimensional acoustic feature vectors and SigLIP 2 representations, are provided in pre-extracted, frozen, and aligned form in `artifacts/video_primary_manifest.csv` and `artifacts/siglip2_frozen_features.npz`. In accordance with child privacy protection regulations and repository hosting quotas, raw full-length video recordings (.mp4), raw audio waveforms (.wav), and large transformer model checkpoints (475 MB per model) are not distributed in this Git repository. Branch representations are frozen during late fusion evaluation as described in `docs/methods_and_claim_boundaries.md`.

## Data ethics and annotation note

- **Segment annotation**: Five independent native Indonesian annotators evaluated each segment across the complete 13,307-record master collection (Fleiss' multi-rater κ = 0.8853; mean pairwise Cohen's κ across all 10 bilateral combinations = 0.8855, with individual pair kappas ranging from 0.7951 to 0.9224; exact mean = 0.885504030446). All 10 bilateral pair scores are itemized in `artifacts/inter_annotator_agreement_audit.csv`. Final segment labels reflect majority voting (≥ 3/5). Annotation guidelines and benchmark exemplars are archived in `docs/` as task definitions; empirical annotator execution timesheets/session logs are not currently available.
- **Expert validation**: Two domain experts (Forensic Linguistics and Child Psychology) evaluated a stratified subsample of 1,331 records, achieving an inter-expert Cohen's kappa of 0.9075. This measures expert consistency on the audit sample and is not a post-adjudication relabeling of the full segment dataset.
- **Privacy protection**: Pre-computed features (`artifacts/siglip2_frozen_features.npz`) use consistent anonymous surrogate IDs (`CB_MASTER_XXXXX`). No raw usernames, profile URLs, or social media post identifiers are published.

## Citation

```bibtex
@article{muzakir2026latefusion,
  title={Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection},
  author={Muzakir, Ari and Ependi, Usman and Suyanto},
  year={2026}
}
```

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)). Adapted components from HEALNet retain their Apache 2.0 license under `src/vendor/HEALNET_LICENSE`.
