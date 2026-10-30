"""Run one pretrained detector over one dataset and save COCO-format predictions.

Also records per-image GPU latency and peak memory. Timing starts from a decoded
uint8 image in host memory and ends with boxes on the CPU, so it includes
preprocessing, host-to-device transfer, the network, and post-processing (NMS
where the model uses it), but not disk reads or JPEG/PNG decoding.

Example:
  .venv/Scripts/python.exe scripts/run_inference.py --model yolo11n --dataset coco
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints/torch"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

DATASETS = {
    "coco": {"images": ROOT / "data/coco/val2017", "gt": ROOT / "data/coco/annotations/instances_val2017.json"},
    "kitti": {"images": ROOT / "data/kitti/val/images", "gt": ROOT / "data/kitti/annotations/kitti_val_coco.json"},
}
# Ultralytics class index (0-79) -> original COCO category id (1-90).
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                85, 86, 87, 88, 89, 90]


def load_config(model_id):
    cfg = json.loads((ROOT / "configs/models.json").read_text())
    model = next(m for m in cfg["models"] if m["id"] == model_id)
    return model, cfg["inference"]


class UltralyticsDetector:
    def __init__(self, model_cfg, settings):
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if model_cfg["backend"] == "ultralytics-rtdetr" else YOLO
        self.model = cls(model_cfg["checkpoint"])  # resolved/downloaded in checkpoints/ (the cwd)
        self.kwargs = dict(imgsz=model_cfg["image_size"], conf=settings["score_threshold"],
                           iou=settings["nms_iou"]["ultralytics"], max_det=300, device=0,
                           half=False, verbose=False, save=False)

    def __call__(self, image_bgr):
        boxes = self.model.predict(image_bgr, **self.kwargs)[0].boxes
        return (boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                np.asarray(COCO80_TO_91)[boxes.cls.cpu().numpy().astype(int)])


class TorchvisionDetector:
    def __init__(self, model_cfg, settings):
        from torchvision.models import detection
        builder = getattr(detection, model_cfg["builder"])
        weights_enum, weights_name = model_cfg["checkpoint"].split(".")
        weights = getattr(detection, weights_enum)[weights_name]
        self.model = builder(weights=weights, box_score_thresh=settings["score_threshold"],
                             box_nms_thresh=settings["nms_iou"]["torchvision"],
                             box_detections_per_img=settings["max_detections_per_image"]).eval().cuda()

    @torch.inference_mode()
    def __call__(self, image_bgr):
        rgb = torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1])).cuda()
        out = self.model([rgb.permute(2, 0, 1).float().div_(255)])[0]
        return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()


def build(model_cfg, settings):
    if model_cfg["backend"].startswith("ultralytics"):
        return UltralyticsDetector(model_cfg, settings)
    return TorchvisionDetector(model_cfg, settings)


def environment():
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False).stdout.strip()
    import ultralytics
    return {"gpu": torch.cuda.get_device_name(0), "driver": driver, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
            "python": platform.python_version(), "os": platform.platform()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, help="Only the first N images (pipeline checks only)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--imgsz", type=int, help="Ablation: override the configured input size")
    parser.add_argument("--max-det", type=int, help="Ablation: override detections kept per image")
    parser.add_argument("--pad-square", action="store_true",
                        help="Ablation: pad the bottom of each image to a square before inference, so a "
                             "stretching resize preserves aspect ratio (boxes stay in original coordinates)")
    args = parser.parse_args()

    model_cfg, settings = load_config(args.model)
    tag = ""
    if args.imgsz:
        model_cfg = {**model_cfg, "image_size": args.imgsz}
        tag += f"_imgsz{args.imgsz}"
    if args.pad_square:
        model_cfg = {**model_cfg, "resize": model_cfg["resize"] + "; input pre-padded to square (aspect preserved)"}
        tag += "_padsq"
    gt = json.loads(DATASETS[args.dataset]["gt"].read_text())
    images = sorted(gt["images"], key=lambda im: im["id"])[: args.limit]
    image_dir = DATASETS[args.dataset]["images"]
    os.chdir(ROOT / "checkpoints")  # Ultralytics downloads missing weights to the cwd.
    detector = build(model_cfg, settings)

    first = cv2.imread(str(image_dir / images[0]["file_name"]))
    for _ in range(args.warmup):
        detector(first)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    keep = args.max_det or settings["max_detections_per_image"]
    if args.max_det:
        tag += f"_maxdet{args.max_det}"
    predictions, latencies_ms = [], []
    for i, im in enumerate(images):
        image = cv2.imread(str(image_dir / im["file_name"]))
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.pad_square:
            h, w = image.shape[:2]
            image = cv2.copyMakeBorder(image, 0, max(w - h, 0), 0, max(h - w, 0),
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
        boxes, scores, labels = detector(image)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)
        order = np.argsort(-scores)[:keep]
        for (x1, y1, x2, y2), s, c in zip(boxes[order], scores[order], labels[order]):
            predictions.append({"image_id": im["id"], "category_id": int(c),
                                "bbox": [round(float(x1), 2), round(float(y1), 2),
                                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                "score": round(float(s), 5)})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(images)} images, median {np.median(latencies_ms):.1f} ms", flush=True)

    suffix = tag + (f"_limit{args.limit}" if args.limit else "")
    pred_path = ROOT / f"predictions/{args.dataset}/{args.model}{suffix}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(predictions))

    lat = np.asarray(latencies_ms)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.executable).name, "scripts/run_inference.py"] + sys.argv[1:]),
        "model": model_cfg, "dataset": args.dataset, "num_images": len(images),
        "settings": {**settings, "max_detections_per_image": keep, "warmup_iterations": args.warmup},
        "timing_boundary": "decoded uint8 image in host memory -> boxes on CPU (pre-process, H2D, network, post-process/NMS); excludes disk read and image decode",
        "latency_ms": {"mean": round(lat.mean(), 2), "median": round(float(np.median(lat)), 2),
                       "p90": round(float(np.percentile(lat, 90)), 2), "std": round(lat.std(), 2)},
        "fps_from_mean": round(1000 / lat.mean(), 1),
        "peak_gpu_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "num_predictions": len(predictions),
        "predictions_file": pred_path.relative_to(ROOT).as_posix(),
        "environment": environment(),
    }
    run_path = ROOT / f"results/runs/{args.dataset}_{args.model}{suffix}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.model} on {args.dataset}: {len(images)} images, mean {lat.mean():.1f} ms "
          f"({record['fps_from_mean']} FPS), peak {record['peak_gpu_memory_allocated_mib']} MiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
