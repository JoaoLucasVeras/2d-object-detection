# 2D Object Detection

Accuracy, speed, and domain shift across pretrained detectors on COCO and KITTI. Track A experimental project for CMPE 249.

## Start on the GPU desktop

See [desktop setup](docs/DESKTOP_SETUP.md) for prerequisites and troubleshooting.

```powershell
git clone https://github.com/JoaoLucasVeras/2d-object-detection.git
cd 2d-object-detection
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

This installs the project environment, verifies CUDA and GPU NMS, and smoke-tests five pretrained detector configurations. Python 3.11 (64-bit), Git, and a working NVIDIA driver must already be installed.

## Research question

How do pretrained object detectors compare in accuracy and inference speed, and how well do they transfer from everyday images to driving scenes?

## Scope

- Evaluate at least three model configurations from at least two detector families.
- Evaluate at least two datasets or domains.
- Analyze cross-dataset generalization and speed-accuracy tradeoffs.
- Use pretrained checkpoints; fine-tuning is optional.
- Hardware: NVIDIA GeForce RTX 3060 Ti (verified).

Configurations: YOLO11n, YOLO11s, RT-DETR-L, Faster R-CNN MobileNetV3-320-FPN, and Faster R-CNN R50-FPN-v2 (three families); see `configs/models.json`. Datasets: COCO val2017 and KITTI val with the mapping in `configs/kitti_taxonomy.json`.

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
6. Evaluate all ten main model-dataset combinations, followed by the four ablations.
7. Analyze accuracy, latency, memory, domain shift, and representative failures.
8. Produce a 6-10 page PDF report or 15-25 slide PDF presentation.

## Reproducibility

For each experiment record the model/checkpoint, dataset/split, input resolution, confidence/NMS settings, hardware, framework versions, exact command, evaluation protocol, and outputs. Preserve prediction artifacts locally so results can be recomputed. Document sampling, taxonomy mapping, ignored labels, and timing boundaries.

The submission also needs at least eight credible technical references, including four original research papers and two official repository/framework documentation sources. Include or link experiment code, configurations, or command logs.

## Reproduce

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1        # downloads data, 5 models x 2 datasets
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ablations.ps1
.\.venv\Scripts\python.exe scripts\failure_analysis.py --dataset kitti --draw yolo11s
.\.venv\Scripts\python.exe scripts\failure_analysis.py --dataset coco
.\.venv\Scripts\python.exe scripts\summarize.py
.\.venv\Scripts\python.exe scripts\build_report.py
```

Data downloads need about 1.5 GB (COCO val2017, KITTI val images, KITTI labels). The full run took about 40 minutes on the RTX 3060 Ti.

## Status

All experiments are complete. Report: `report/report.pdf`. Tables: `results/summary.md`. Per-run records: `results/runs/`. Metrics: `results/metrics/`. Command log: `results/command_log.txt`. See `docs/experiment-plan.md` for decisions and results.
