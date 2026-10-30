"""Record the actual host environment; optionally require a working CUDA device."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect():
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 1),
        "packages": {},
        "cuda_ready": False,
        "errors": [],
    }
    for name in ("torch", "torchvision", "ultralytics", "numpy", "pycocotools"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {name}")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        report["nvidia_smi"] = driver.stdout.strip() or driver.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    try:
        import torch
        import torchvision
        report["torch_cuda_runtime"] = torch.version.cuda
        report["torchvision_version"] = torchvision.__version__
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA. Check the NVIDIA driver and CUDA wheel installation.")
        device = torch.device("cuda:0")
        # Exercise both CUDA compute and the compiled torchvision detection operator.
        a = torch.ones((32, 32), device=device)
        assert (a @ a)[0, 0].item() == 32
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
        assert keep.tolist() == [0]
        torch.cuda.synchronize()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        report["cuda_ready"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/environment.json")
    args = parser.parse_args()
    report = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] or (args.require_cuda and not report["cuda_ready"]) else 0


if __name__ == "__main__":
    sys.exit(main())
