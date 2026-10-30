"""Score saved predictions with the official COCO evaluator (pycocotools).

For each model/dataset pair this writes results/metrics/<dataset>_<model>.json with
the 12 standard COCO numbers, per-class AP, and:
  * COCO: the same metrics restricted to the classes shared with KITTI (person, car, truck)
  * KITTI: AP for subsets by occlusion level and box height; objects outside the
    subset become ignore regions so detecting them is neither rewarded nor penalized.

Example:
  .venv/Scripts/python.exe scripts/evaluate.py --dataset kitti --model yolo11n
"""
import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
GT = {"coco": ROOT / "data/coco/annotations/instances_val2017.json",
      "kitti": ROOT / "data/kitti/annotations/kitti_val_coco.json"}
SHARED = [1, 3, 8]  # person, car, truck
STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
KITTI_SUBSETS = {
    "occlusion_0_fully_visible": lambda a: a["occluded"] == 0,
    "occlusion_1_partly": lambda a: a["occluded"] == 1,
    "occlusion_2_largely": lambda a: a["occluded"] == 2,
    "height_lt25px": lambda a: a["bbox"][3] < 25,
    "height_25_40px": lambda a: 25 <= a["bbox"][3] < 40,
    "height_ge40px": lambda a: a["bbox"][3] >= 40,
    "kitti_moderate": lambda a: a["bbox"][3] >= 25 and a["occluded"] <= 1 and a["truncated"] <= 0.3,
}


def make_coco(dataset_dict):
    coco = COCO()
    coco.dataset = dataset_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def run_eval(gt, dt, cat_ids, img_ids):
    ev = COCOeval(gt, dt, "bbox")
    ev.params.catIds = cat_ids
    ev.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    out = {name: round(float(v) * 100, 2) for name, v in zip(STAT_NAMES, ev.stats)}
    precision = ev.eval["precision"]  # [IoU thr, recall, class, area, maxDets]
    names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    per_class = {}
    for k, cat in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p50 = precision[0, :, k, 0, -1]
        n_gt = len(gt.getAnnIds(imgIds=img_ids, catIds=[cat], iscrowd=False))
        per_class[names[cat]] = {
            "AP": round(float(p[p > -1].mean()) * 100, 2) if (p > -1).any() else None,
            "AP50": round(float(p50[p50 > -1].mean()) * 100, 2) if (p50 > -1).any() else None,
            "num_gt": n_gt,
        }
    out["per_class"] = per_class
    return out


def subset_gt(gt_dict, keep):
    """Turn scored objects that fail `keep` into ignore regions."""
    d = copy.deepcopy(gt_dict)
    for a in d["annotations"]:
        if not a["iscrowd"] and not keep(a):
            a["iscrowd"] = 1
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=GT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", type=Path, help="Defaults to predictions/<dataset>/<model>.json")
    args = parser.parse_args()

    pred_path = (args.predictions or ROOT / f"predictions/{args.dataset}/{args.model}.json").resolve()
    gt_dict = json.loads(GT[args.dataset].read_text())
    predictions = json.loads(pred_path.read_text())
    gt = make_coco(gt_dict)
    image_ids = {im["id"] for im in gt_dict["images"]}
    if {p["image_id"] for p in predictions} - image_ids:
        raise ValueError("Predictions reference images outside the ground truth")
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
    # A --limit pipeline check scores only the first N images; full runs score every image.
    limit = int(pred_path.stem.split("_limit")[1]) if "_limit" in pred_path.stem else None
    img_ids = sorted(image_ids)[:limit]

    cat_ids = sorted(c["id"] for c in gt_dict["categories"])
    result = {"dataset": args.dataset, "model": args.model, "predictions_file": pred_path.relative_to(ROOT).as_posix(),
              "evaluator": "pycocotools COCOeval bbox, maxDets=100, IoU 0.50:0.05:0.95; iscrowd/ignore regions excluded",
              "all_classes": run_eval(gt, dt, cat_ids, img_ids)}
    if args.dataset == "coco":
        result["shared_classes"] = run_eval(gt, dt, SHARED, img_ids)
    else:
        result["subsets"] = {}
        for name, keep in KITTI_SUBSETS.items():
            sub = make_coco(subset_gt(gt_dict, keep))
            with contextlib.redirect_stdout(io.StringIO()):
                sub_dt = sub.loadRes(predictions)
            r = run_eval(sub, sub_dt, cat_ids, img_ids)
            result["subsets"][name] = {"AP": r["AP"], "AP50": r["AP50"], "AR100": r["AR100"],
                                       "per_class": r["per_class"]}

    out = ROOT / f"results/metrics/{args.dataset}_{pred_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    a = result["all_classes"]
    print(f"{pred_path.stem} on {args.dataset}: AP {a['AP']}  AP50 {a['AP50']}  AP75 {a['AP75']}  "
          f"APs {a['AP_small']}  APm {a['AP_medium']}  APl {a['AP_large']}  AR100 {a['AR100']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
