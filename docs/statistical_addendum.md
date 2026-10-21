# Equal-VAT paired source-group statistical addendum

## Scope

This addendum closes the missing manuscript-focused comparison for `equal_vat` using the stored
`VIDEO_OOF_PREDICTIONS.csv`. No model was retrained and no threshold was reselected.

The implementation reproduces the original notebook (`V6_Reviewer_Experiments_L4.ipynb`) exactly:
primary seed = 42; n = 3,951 aligned video segments; 214 source groups; 2,000 multinomial
source-group bootstrap replicates; 2,000 paired source-group permutation replicates; two-sided
permutation p values; and Holm correction.

Important provenance note: these four `equal_vat` contrasts were added after the primary results were
available because `equal_vat` had the highest observed primary-seed Macro F1. They should therefore be
described as **post-hoc / exploratory manuscript-focused contrasts**, not as originally predeclared tests.

## Results

Seed-42 Macro F1:
- equal_vat: 0.561512
- text: 0.558906
- visual: 0.528120
- siglip2_vta_probe: 0.530354
- healnet_adapted_2024: 0.507626

| Contrast | Δ Macro F1 | 95% group-bootstrap CI | Raw permutation p | Holm p (4 new) | Holm p (all 11) | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| equal_vat vs text | +0.002606 | [-0.013822, 0.019038] | 0.760120 | 0.760120 | 1.000000 | No evidence that equal VAT outperforms text-only. |
| equal_vat vs visual | +0.033392 | [0.004167, 0.061703] | 0.025987 | 0.077961 | 0.259870 | Positive point estimate, but not significant after Holm correction. |
| equal_vat vs siglip2_vta_probe | +0.031158 | [0.001327, 0.059372] | 0.038481 | 0.077961 | 0.346327 | Positive point estimate, but not significant after Holm correction. |
| equal_vat vs healnet_adapted_2024 | +0.053886 | [0.025530, 0.084976] | 0.001499 | 0.005997 | 0.016492 | Significant advantage; survives both Holm families. |

The bootstrap confidence intervals above are ordinary per-comparison 95% intervals and are not
multiplicity-adjusted. Therefore, an interval can exclude zero while the Holm-adjusted permutation
test is not significant; this occurs for the visual and SigLIP2 VTA contrasts.

## Primary interpretation

1. **equal_vat vs text:** Δ Macro F1 = +0.002606,
   95% CI [-0.013822, 0.019038],
   Holm-adjusted p (4-test family) = 0.760120.
   There is no statistical evidence that equal tri-modal fusion outperforms the strong text branch.

2. **equal_vat vs visual:** Δ = +0.033392.
   The unadjusted CI is positive, but Holm-adjusted p = 0.077961;
   do not claim statistically significant superiority after multiplicity correction.

3. **equal_vat vs SigLIP2 VTA:** Δ = +0.031158.
   The unadjusted CI is positive, but Holm-adjusted p = 0.077961;
   again, do not claim significance after multiplicity correction.

4. **equal_vat vs HEALNet adapted 2024:** Δ = +0.053886,
   95% CI [0.025530, 0.084976],
   Holm-adjusted p (4 tests) = 0.005997.
   This contrast remains significant even under the conservative Holm correction across all 11 video
   contrasts (p = 0.016492).

## Multi-seed consistency

Text-only Macro F1 across seeds 42/2026/2027:
0.555968 ± 0.006651.

Equal-VAT Macro F1 across seeds 42/2026/2027:
0.555931 ± 0.008472.

This reinforces the main conclusion: equal-VAT is competitive, but its advantage over text-only is
not stable or statistically supported.

## Manuscript-ready Results paragraph

> At the primary seed, equal visual-audio-text fusion achieved a Macro F1 of
> 0.5615, compared with 0.5589 for the text-only branch.
> The paired source-group analysis showed only a small difference
> (Δ = +0.0026, 95% bootstrap CI
> [-0.0138, 0.0190],
> Holm-adjusted permutation p = 0.760), providing no evidence
> of superiority over text-only. Equal fusion produced higher point estimates than visual-only and the
> SigLIP2 VTA probe, but these contrasts did not remain significant after Holm correction across the
> four added comparisons (adjusted p = 0.078 for both).
> In contrast, equal fusion significantly exceeded the adapted HEALNet baseline
> (Δ = +0.0539, 95% CI
> [0.0255, 0.0850],
> adjusted p = 0.006). Across three training seeds,
> equal fusion (0.5559 ± 0.0085)
> and text-only (0.5560 ± 0.0067)
> were effectively equivalent.

## Manuscript-ready Methods addendum

> Additional manuscript-focused comparisons were performed using the stored primary-seed out-of-fold
> predictions without retraining. For each contrast, source groups were resampled using 2,000
> multinomial cluster-bootstrap replicates to estimate the 95% confidence interval of the Macro-F1
> difference. Statistical significance was assessed using 2,000 paired source-group permutations in
> which model assignments were swapped jointly for all segments belonging to a source group.
> Two-sided permutation p values were adjusted with the Holm procedure across the four added
> equal-fusion contrasts. Because these contrasts were selected after inspection of the primary
> point estimates, they are reported as exploratory post-hoc analyses.

## Recommended claim boundary

Safe:
- “Equal tri-modal fusion achieved the highest observed Macro F1 at the primary seed.”
- “Its improvement over text-only was small and not statistically significant.”
- “Its advantages over visual-only and SigLIP2 VTA did not survive Holm correction.”
- “It significantly outperformed the adapted HEALNet baseline in the added paired analysis.”
- “Across three seeds, text-only and equal tri-modal fusion showed essentially equivalent Macro F1.”

Avoid:
- “Multimodal fusion significantly outperformed text-only.”
- “Multimodal fusion consistently outperformed all baselines.”
- “All four added contrasts were predeclared.”

## Recommended file policy

Do not overwrite the original `VIDEO_PAIRED_GROUP_STATISTICS.csv`. Keep it as the immutable output
of the 20 October run. Use `VIDEO_EQUAL_VAT_PAIRED_GROUP_STATISTICS.csv` as the added analysis and
`VIDEO_PAIRED_GROUP_STATISTICS_EXTENDED.csv` only as a consolidated audit/supplementary table.
