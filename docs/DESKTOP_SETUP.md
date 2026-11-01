# Start on the RTX 3060 Ti desktop

## Prerequisites

- Windows 10/11, 64-bit.
- Git: https://git-scm.com/downloads/win
- Python 3.11, 64-bit, with the Python launcher (`py`): https://www.python.org/downloads/windows/
- A current NVIDIA driver for the RTX 3060 Ti: https://www.nvidia.com/Download/index.aspx
- Internet access and several GB free for the environment and checkpoints. Dataset storage is additional; plan capacity once datasets are finalized.
- The repository is public; cloning does not require GitHub authentication. Authentication is required to push changes.

The setup uses CUDA-enabled PyTorch wheels with their CUDA runtime. A separate CUDA Toolkit installation is not part of this setup. The installed NVIDIA driver must support the selected CUDA 12.6 runtime.

## Clone and start

Open PowerShell in the folder where you want to keep the project:

```powershell
git clone https://github.com/JoaoLucasVeras/2d-object-detection.git
cd 2d-object-detection
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The execution-policy option applies to that PowerShell process. The script creates a local `.venv`, installs pinned direct dependencies, checks package consistency, tests CUDA compute and torchvision GPU NMS, then downloads official model checkpoints and runs the three starter detectors on synthetic input. No datasets are downloaded.

The configurations are YOLO11 nano, YOLO11 small, and Faster R-CNN MobileNetV3 320 FPN. The two YOLO sizes belong to one family; Faster R-CNN supplies the second family. Input resolutions differ and must be disclosed in future comparisons.

Success means all three models print `PASS` and the script prints `Setup complete`. Synthetic images test execution; their detections and memory figures are not assignment evaluation results.

## Saved records

- `results/local/environment.json`: actual GPU, package versions, driver, CUDA check, and disk space.
- `results/local/environment-freeze.txt`: resolved package versions from the successful installation stage.
- `results/local/smoke-test.json`: individual model execution results.
- `checkpoints/`: downloaded pretrained weights.

These machine-specific artifacts stay local. Direct dependency versions are pinned; the freeze records transitive dependencies after installation, rather than claiming a complete cross-machine lock.

## Rerun checks

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

No virtual-environment activation is necessary. Setup can be rerun after an interrupted download. To install and verify CUDA without downloading model weights:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -SkipSmokeTest
```

If Python is missing, install Python 3.11 with the launcher and reopen PowerShell. If CUDA or GPU NMS fails, update the NVIDIA driver, restart, and rerun setup. Read the saved environment report for the exact failure. For download failures, check internet access and rerun. Do not continue to benchmark until the three smoke tests pass.

## Handoff to Codex

Open this cloned repository as the project on the desktop and ask:

> Continue the Track A object detection project. Read README.md, AGENTS.md, docs/experiment-plan.md, and results/local environment and smoke-test reports. Verify GPU readiness, then finalize COCO/driving dataset splits and class mappings and implement reproducible evaluation. No experiments have been completed yet.

## Next milestone

Dataset preparation, taxonomy mapping, full prediction/evaluation scripts, and timing protocols remain to be implemented. Passing setup makes the desktop ready for that milestone; it does not complete the six experimental runs.

## Installation sources

- Official PyTorch version pairs and CUDA wheels: https://pytorch.org/get-started/previous-versions/
- Ultralytics setup: https://docs.ultralytics.com/quickstart/
- YOLO11 usage: https://github.com/ultralytics/yolo11
