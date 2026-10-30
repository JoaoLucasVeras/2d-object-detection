# Experiment plan

## Decisions pending

- Confirm GPU access, available disk space, and software environment.
- Select three model configurations across at least two families.
- Finalize datasets, labeled evaluation splits, and any fixed subsets.
- Define shared categories, taxonomy mappings, and ignore policies before evaluation.

## Intended evaluation matrix

| Configuration | Dataset/domain 1 | Dataset/domain 2 |
| --- | --- | --- |
| Model configuration 1 (pending) | Pending | Pending |
| Model configuration 2 (pending) | Pending | Pending |
| Model configuration 3 (pending) | Pending | Pending |

## Metrics

Prioritize mAP@[0.50:0.95], AP50, per-class AP, latency/FPS, and peak GPU memory. Include object-size AP where supported. State evaluator conventions; do not compare different category sets or annotation protocols as if they were identical.

For latency, specify batch size, precision, warmup, GPU synchronization, repetitions, and whether preprocessing, transfers, and post-processing are included.

## Required analyses

1. Cross-dataset generalization: inspect performance by shared class and object scale, acknowledging annotation and dataset differences.
2. Speed-accuracy tradeoffs: compare all configurations under documented timing conditions.

Include failure examples with ground truth and predictions and explain false positives, missed objects, localization errors, and occlusion where supported by evidence.

## Deliverables

- Reproducible scripts, configurations, and command logs
- Quantitative tables and selected failure figures
- PDF report (6-10 pages) or PDF presentation (15-25 slides)
- At least eight technical references: at least four original papers and two official repositories/framework documentation sources
