# 2D Object Detection Evaluation

Track A experimental project for CMPE 249.

## Research question

How do pretrained object detectors compare in accuracy and inference speed, and how well do they transfer from everyday images to driving scenes?

## Scope

- Evaluate at least three model configurations from at least two detector families.
- Evaluate at least two datasets or domains.
- Analyze cross-dataset generalization and speed-accuracy tradeoffs.
- Use pretrained checkpoints; fine-tuning is optional.
- Target hardware: NVIDIA GeForce RTX 3060 Ti. GPU execution is not yet verified.

Model configurations, checkpoints, datasets, and splits remain to be finalized after a GPU smoke test. COCO validation and KITTI are candidate datasets.

## Project layout

- `scripts/`: environment checks, inference, and evaluation tools
- `configs/`: exact experiment configurations
- `docs/`: experiment design, references, and investigation notes
- `results/`: small metrics tables and experiment logs
- `report/`: report source and final submission
- `data/` and `checkpoints/`: local only; excluded from Git

## Workflow

1. Verify CUDA/PyTorch access on the desktop and run a small inference test.
2. Select three configurations spanning at least two detector families.
3. Prepare two datasets and define shared class mappings and ignore rules.
4. Implement reproducible prediction and evaluation commands.
5. Validate the pipeline on a small sample from each dataset.
6. Evaluate all six model-dataset combinations.
7. Analyze accuracy, latency, memory, domain shift, and representative failures.
8. Produce a 6-10 page PDF report or 15-25 slide PDF presentation.

## Reproducibility

For each experiment record the model/checkpoint, dataset/split, input resolution, confidence/NMS settings, hardware, framework versions, exact command, evaluation protocol, and outputs. Preserve prediction artifacts locally so results can be recomputed. Document sampling, taxonomy mapping, ignored labels, and timing boundaries.

The submission also needs at least eight credible technical references, including four original research papers and two official repository/framework documentation sources. Include or link experiment code, configurations, or command logs.

## Status

Planning stage. No models have been evaluated and no experimental results are claimed.
