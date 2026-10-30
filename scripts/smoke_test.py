"""Download official pretrained weights and test each starter detector on CUDA.

Synthetic input verifies execution only; this is not an accuracy or latency benchmark.
"""
import gc
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints/torch"))


def main():
    import numpy as np
    import torch
    from torchvision.models import detection
    from ultralytics import RTDETR, YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run scripts/check_environment.py --require-cuda first.")
    configs = json.loads((ROOT / "configs/models.json").read_text())["models"]
    output = ROOT / "results/local/smoke-test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    # Ultralytics downloads missing weight filenames to the current working directory.
    os.chdir(checkpoints)
    records = []
    for config in configs:
        record = {"model": config, "passed": False}
        model = prediction = None
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                if config["backend"].startswith("ultralytics"):
                    model = (RTDETR if config["backend"] == "ultralytics-rtdetr" else YOLO)(config["checkpoint"])
                    prediction = model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                                               imgsz=config["image_size"], device=0,
                                               verbose=False, save=False)
                    assert prediction[0].boxes is not None
                    assert prediction[0].boxes.xyxy.shape[-1] == 4
                    assert next(model.model.parameters()).is_cuda
                else:
                    weights_enum, weights_name = config["checkpoint"].split(".")
                    weights = getattr(detection, weights_enum)[weights_name]
                    model = getattr(detection, config["builder"])(weights=weights).eval().cuda()
                    prediction = model([torch.zeros((3, 480, 640), device="cuda")])
                    assert prediction[0]["boxes"].shape[-1] == 4
                    assert prediction[0]["boxes"].is_cuda
                torch.cuda.synchronize()
                record["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                record["passed"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            del model, prediction
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        output.write_text(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Synthetic-input CUDA smoke test; not experimental metrics",
            "results": records,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"{config['id']}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
