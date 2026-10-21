# V6 experiment package interpretation

This package is an internal evaluation on an already explored corpus. The corrected text holdout and canonical video folds are not external confirmation.

## Populations and labels
Text: high-agreement social comments, deduplicated after the documented normalization, grouped by informative content URL and platform-scoped author. Every split must contain every source platform. The subset scope must accompany reported scores.
Video: only source rows with matching segment/frame identity and normalized OCR/STT content are eligible. Exclusions are determined before predictions, never by performance. VIDEO_ALIGNMENT_AUDIT.csv retains every original row and eligibility reason. The source master labels are never changed.
LABEL_AUDIT.json records mathematical agreement metrics over the corpus.
Annotation protocol and inter-rater reliability:
- Fleiss' kappa (κ = 0.8853): Evaluated across the entire master collection of N = 13,307 records by five independent human annotators (anotator_1 through anotator_5).
- Annotator pairwise Cohen's kappa (mean κ = 0.8851): Evaluated across the same master collection of N = 13,307 records as the unweighted mean of all 10 bilateral annotator pairs (ranging between 0.871 and 0.899).
- Inter-expert agreement (Cohen's κ = 0.9075): Evaluated on a stratified gold-standard subsample of n = 1,331 records (10.0% of the corpus) between two independent domain experts (Expert 1: Indonesian Forensic Linguist; Expert 2: Child Developmental Psychologist). This reflects bilateral inter-expert agreement on the audit subsample, and must not be described as a post-adjudication score for the full segment corpus in the absence of itemized adjudication records.
- Documentary evidence: Independent human annotation guidelines and phase-specific gold standards are archived in docs/annotation_guidelines.docx, docs/gold_standard_phase_1.pdf, and docs/gold_standard_phase_2.pdf. The master segment labels reflect majority consensus (≥ 3/5) without performance-guided relabeling.

## Training and selection
Video uses the same canonical outer groups for all configurations. Each outer training population has four disjoint roles: encoder fit, encoder epoch selection, meta fitting, and decision-threshold calibration. Test groups and identical images are excluded from every training/selection role.
Text and image encoders start from generic pretrained checkpoints and are independently retrained per outer fold and seed. Video text preprocessing is identical during fit and inference. Absent OCR and STT gives a neutral text probability and zero text embedding. Audio standardization/classification fits only the encoder-fit role.
All fusion variants use the same branch probabilities. Meta LR uses C=0.2, group-balanced weights, and meta-fit observations only. Neural fusion uses fixed epochs on the meta-fit role. Main operating thresholds are selected only on calibration. Fixed-0.50 sensitivity is reported for every model.

## Baselines
HEALNet is the official NeurIPS 2024 fusion module pinned by commit and source hash, adapted to fold-specific frozen vector features and a binary head. It is not a reproduction of its biomedical study or original end-to-end setup. A gradient check verifies that each modality participates.
SigLIP2 is the official Google 2025 pretrained vision-language model with frozen features and supervised linear probes on the same outer partitions. This tests a stronger representation baseline and does not isolate the fusion mechanism. SigLIP2 text uses its pretrained 64-token limit.
Coverage-only and coverage-shuffled configurations are negative/shortcut controls. They must not be omitted merely because they perform well.

## Statistical scope
Primary comparisons use seed 42, specified before training. Seeds 2026 and 2027 estimate retraining variability and are all reported; the best test seed is never selected. VIDEO_SEED_SUMMARY.csv contains means and sample standard deviations.
Confidence intervals resample source groups; paired group permutations test predeclared contrasts with Holm correction. Intervals condition on fitted models and do not capture the full uncertainty of new data collection or every training choice. Segment McNemar p values are descriptive because windows within a source are dependent.
Text-extraction dropout is a synthetic fixed-threshold stress test, not a new real-world test set. LOPO outputs are explicitly frozen pretrained-feature probes and must not be presented as LOPO of the full fine-tuned coverage-aware system.

## Figure preparation
Use the stored sample predictions, thresholds, curve coordinates and paired comparisons. No final publication figures are generated here: their selection and design remain a separate discussion. ROC and PR use continuous probabilities; confusion/F1 use stored fold-specific decisions.
Do not label threshold selection as probability calibration. Do not mix this package with legacy 867-segment results or legacy text ensemble scores.
