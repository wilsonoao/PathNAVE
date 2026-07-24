# Post-eval Analysis

## Files generated
- `task_method_mrfh_precision_hitrate.png`: required comparison plot, one panel per task.
- `threshold_sensitivity_by_task.png`: hit-rate and MRfH curves over precision thresholds.
- `threshold_choice_tradeoff.png`: threshold trade-off summary.
- `summary_metrics.csv`: method/task metrics at precision threshold 0.05.
- `threshold_recommendation.csv`: threshold-level robustness table.

## Threshold recommendation
I would report a non-zero precision threshold of **0.05** as the reviewer-friendly default. Across all available method/task pairs, its mean hit rate is 66.82%, mean hit-rate drop from threshold 0.05 is 0.00 pp, worst drop is 0.00 pp, and mean MRfH drop is 0.0000.

Rationale: threshold 0.05 requires measurable overlap while preserving most of the ranking signal.

## Method/task summary at threshold 0.05
| method | task | case_micro_precision | per_roi_micro_precision | conditional_hit_cases | conditional_hit_rate | conditional_mrfh | processed_cases | per_roi_total | total_16x16_subpatches | coverage_16x16_multiscale | efficiency | efficiency_avg_per_case | efficiency_cases_with_total_size | has_explicit_hit_rate_file | hit_threshold | mrfh_at_hit_threshold | hit_rate_at_hit_threshold | hit_cases_at_hit_threshold | threshold_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPathAgent | CAMELYON16_detection | 0.0374 | 0.0370 | 40.0000 | 0.8333 | 0.6886 | 48.0000 | 760.0000 | 517868352.0000 | 0.1359 | 0.1495 | 0.1557 | 48.0000 | True | 0.0500 | 0.1777 | 41.7000 | 20 | precision |
| PathAgent | CAMELYON16_detection | 0.0204 | 0.0210 | 22.0000 | 0.3607 | 0.0641 | 61.0000 | 1633.0000 | 103445640.0000 | 0.0209 | 0.0242 | 0.0291 | 61.0000 | True | 0.0500 | 0.0604 | 37.7000 | 23 | precision |
| Pathology-CoT | CAMELYON16_detection | 0.1386 | 0.1436 | 20.0000 | 0.4167 | 0.2068 | 48.0000 | 330.0000 | 58905513.0000 | 0.0155 | 0.0165 | 0.0171 | 48.0000 | True | 0.0500 | 0.2294 | 39.6000 | 19 | precision |
| slide_seek | CAMELYON16_detection | 0.0896 | 0.0679 | 3.0000 | 0.0625 | 0.0243 | 48.0000 | 232.0000 | 13720000.0000 | 0.0036 | 0.0010 | 0.0026 | 48.0000 | True | 0.0500 | 0.0451 | 8.3000 | 4 | precision |
| CPathAgent | TCGA_BRCA_subtype | 0.3027 | 0.3124 | 51.0000 | 0.5484 | 0.2539 | 93.0000 | 918.0000 | 535513201.0000 | 0.1568 | 0.1743 | 0.2218 | 93.0000 | True | 0.0500 | 0.6916 | 84.9000 | 79 | precision |
| PathAgent | TCGA_BRCA_subtype | 0.2857 | 0.2857 | 3.0000 | 0.0349 | 0.0140 | 86.0000 | 1377.0000 | 90243072.0000 | 0.0282 | 0.0356 | 0.0357 | 86.0000 | True | 0.0500 | 0.4415 | 79.1000 | 68 | precision |
| Pathology-CoT | TCGA_BRCA_subtype | 0.5894 | 0.6060 | 9.0000 | 0.0918 | 0.0508 | 98.0000 | 434.0000 | 68056619.0000 | 0.0186 | 0.0202 | 0.0185 | 98.0000 | True | 0.0500 | 0.7435 | 85.7000 | 84 | precision |
| slide_seek | TCGA_BRCA_subtype | 0.3355 | 0.3465 | 8.0000 | 0.0860 | 0.0423 | 93.0000 | 1137.0000 | 48676992.0000 | 0.0142 | 0.0139 | 0.0206 | 93.0000 | True | 0.0500 | 0.7058 | 86.0000 | 80 | precision |
| CPathAgent | TCGA_LUNG_Classification | 0.3207 | 0.3316 | 329.0000 | 0.4754 | 0.2607 | 692.0000 | 5620.0000 | 3796297750.0000 | 0.1937 | 0.2164 | 0.2676 | 690.0000 | True | 0.0500 | 0.6701 | 81.6000 | 565 | precision |
| PathAgent | TCGA_LUNG_Classification | 0.3920 | 0.3530 | 60.0000 | 0.0747 | 0.0370 | 803.0000 | 18389.0000 | 1036362571.0000 | 0.0442 | 0.0476 | 0.0564 | 801.0000 | True | 0.0500 | 0.6040 | 87.0000 | 699 | precision |
| Pathology-CoT | TCGA_LUNG_Classification | 0.6456 | 0.6510 | 90.0000 | 0.1048 | 0.0850 | 859.0000 | 3370.0000 | 357013907.0000 | 0.0143 | 0.0169 | 0.0171 | 857.0000 | True | 0.0500 | 0.7749 | 84.2000 | 723 | precision |
| slide_seek | TCGA_LUNG_Classification | 0.3590 | 0.3778 | 58.0000 | 0.0676 | 0.0332 | 858.0000 | 20687.0000 | 680232896.0000 | 0.0272 | 0.0225 | 0.0396 | 856.0000 | True | 0.0500 | 0.6526 | 86.0000 | 738 | precision |

## Top threshold candidates
| precision_threshold | mean_hit_rate_pct | min_hit_rate_pct | mean_hit_rate_drop_pp | max_hit_rate_drop_pp | mean_mrfh | mean_mrfh_drop | max_mrfh_drop | reviewer_friendly_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 66.8167 | 8.3000 | 0.0000 | 0.0000 | 0.4830 | 0.0000 | 0.0000 | 0.5000 |
| 0.1000 | 64.4667 | 8.3000 | 2.3500 | 8.4000 | 0.4661 | 0.0170 | 0.0382 | -18.4897 |
| 0.1500 | 63.2417 | 8.3000 | 3.5750 | 12.5000 | 0.4544 | 0.0287 | 0.0618 | -27.6485 |
| 0.2000 | 62.0500 | 6.2000 | 4.7667 | 14.6000 | 0.4406 | 0.0425 | 0.0769 | -32.8162 |
| 0.2500 | 61.2083 | 6.2000 | 5.6083 | 14.6000 | 0.4330 | 0.0500 | 0.0881 | -33.3087 |
| 0.3000 | 60.4000 | 6.2000 | 6.4167 | 16.7000 | 0.4249 | 0.0581 | 0.1096 | -37.9787 |
| 0.3500 | 59.4500 | 6.2000 | 7.3667 | 18.8000 | 0.4160 | 0.0671 | 0.1311 | -42.8083 |
| 0.4000 | 58.8083 | 6.2000 | 8.0083 | 20.9000 | 0.4069 | 0.0761 | 0.1470 | -47.3305 |

## Missing-data note
CPathAgent/CAMELYON16_detection has `mrfh_by_threshold.csv` but no separate `hit_rate_by_threshold.csv`; the hit-rate values were read from `mrfh_by_threshold.csv`, which contains the same hit-rate columns.

The following curve files do not use `precision_threshold`, so they are included in threshold-0 comparison but excluded from the precision-threshold recommendation:
None.