# Experiment plan

## Status (2026-10-30)

All planned experiments are complete on the RTX 3060 Ti. The report is `report/report.pdf` (source `report/report.md`). Every number comes from runs recorded in `results/runs` and `results/metrics`. `results/summary.md` collects the tables, and `results/command_log.txt` holds every executed command.

## Decisions made

- **Models (5 configurations, 3 families):** YOLO11n, YOLO11s (YOLO); RT-DETR-L (DETR); Faster R-CNN MobileNetV3-Large-320-FPN and ResNet-50-FPN-v2 (R-CNN). See `configs/models.json`.
- **Datasets:** COCO val2017 (5,000 images, official annotations) and the KITTI 2D-object val split distributed by Ultralytics (1,496 images from the official training set), re-labeled with the official KITTI `label_2` files.
- **Taxonomy:** KITTI Car/Truck/Pedestrian are scored as COCO car/truck/person. Van, Person_sitting, Cyclist, Tram, Misc, and DontCare become class-specific ignore regions (`iscrowd=1`). See `configs/kitti_taxonomy.json`.
- **Protocol:** FP32, batch 1, score threshold 0.001, top-100 detections, pycocotools bbox evaluation. Latency runs from the decoded image in host memory to boxes on the CPU, with CUDA synchronization and 20 warm-up passes.

## Evaluation matrix (mAP@[.50:.95], %)

| Configuration | COCO val2017 (80 cls) | KITTI val (3 cls) |
| --- | --- | --- |
| YOLO11n, 640 | 38.5 | 26.8 |
| YOLO11s, 640 | 45.5 | 31.0 |
| RT-DETR-L, 640 | 52.2 | 38.3 |
| Faster R-CNN MobileNetV3 320 FPN | 23.1 | 16.1 |
| Faster R-CNN R50-FPN-v2, 800 | 47.0 | 36.5 |

Ablations (`scripts/run_ablations.ps1`): RT-DETR aspect-preserving input on KITTI, YOLO at 1280 px on KITTI, and YOLO11s top-300 detections on COCO.

## Analyses performed

1. Cross-dataset generalization by class, object size, occlusion, and box height.
2. Speed–accuracy trade-off (latency, FPS, peak GPU memory).
3. Architecture comparison: NMS vs NMS-free duplicates, score calibration, localization quality.
4. Failure analysis at each model's best-F1 threshold, with rendered examples in `report/figures`.

## Not done / possible extensions

- Fine-tuning on KITTI (optional for the assignment).
- FP16/TensorRT latency.
- Running Ultralytics' own validator to measure the multi-label NMS effect on the YOLO COCO gap.
