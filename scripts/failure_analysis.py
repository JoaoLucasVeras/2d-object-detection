"""Break detections into error types at an operating point and draw example failures.

Each model is evaluated at the score threshold that maximizes its F1 (IoU 0.5) on the
dataset, so models with differently calibrated scores are compared fairly. Matching is
greedy by score, per class, against scored ground truth. Unmatched detections are
classified, in order, as:
  ignored      - mostly inside an ignore region of that class (not an error)
  duplicate    - IoU >= 0.5 with a ground-truth object already matched
  class_conf   - IoU >= 0.5 with a ground-truth object of another evaluated class
  localization - 0.1 <= IoU < 0.5 with a ground-truth object of the same class
  background   - anything else
Missed objects are split by box height and KITTI occlusion level.

Example:
  .venv/Scripts/python.exe scripts/failure_analysis.py --dataset kitti --draw yolo11s
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT = {"coco": (ROOT / "data/coco/annotations/instances_val2017.json", ROOT / "data/coco/val2017"),
      "kitti": (ROOT / "data/kitti/annotations/kitti_val_coco.json", ROOT / "data/kitti/val/images")}
CLASSES = {1: "person", 3: "car", 8: "truck"}


def iou_matrix(a, b):
    """IoU and intersection-over-area-of-a between xywh boxes a [N,4] and b [M,4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    a, b = a.copy(), b.copy()
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9), inter / (area_a[:, None] + 1e-9)


def index(gt_dict, predictions):
    gts = defaultdict(list)
    for a in gt_dict["annotations"]:
        if a["category_id"] in CLASSES:
            gts[a["image_id"]].append(a)
    dets = defaultdict(list)
    for p in predictions:
        if p["category_id"] in CLASSES:
            dets[p["image_id"]].append(p)
    return gts, dets


def match_image(gts, dets, threshold):
    """Return per-detection and per-GT outcomes for one image."""
    dets = sorted((d for d in dets if d["score"] >= threshold), key=lambda d: -d["score"])
    scored = [g for g in gts if not g.get("iscrowd")]
    ignore = [g for g in gts if g.get("iscrowd")]
    g_box = np.array([g["bbox"] for g in scored]).reshape(-1, 4)
    d_box = np.array([d["bbox"] for d in dets]).reshape(-1, 4)
    i_box = np.array([g["bbox"] for g in ignore]).reshape(-1, 4)
    ious = iou_matrix(d_box, g_box)[0] if len(dets) and len(scored) else np.zeros((len(dets), len(scored)))
    ioa = iou_matrix(d_box, i_box)[1] if len(dets) and len(ignore) else np.zeros((len(dets), len(ignore)))
    matched = np.zeros(len(scored), bool)
    det_outcome = []
    for k, d in enumerate(dets):
        same = np.array([g["category_id"] == d["category_id"] for g in scored], bool)
        cand = np.where(same & ~matched & (ious[k] >= 0.5))[0]
        if len(cand):
            j = cand[np.argmax(ious[k, cand])]
            matched[j] = True
            det_outcome.append("tp")
        elif len(ignore) and any(ioa[k, m] >= 0.5 and ignore[m]["category_id"] == d["category_id"]
                                 for m in range(len(ignore))):
            det_outcome.append("ignored")
        elif (same & matched & (ious[k] >= 0.5)).any():
            det_outcome.append("duplicate")
        elif (~same & (ious[k] >= 0.5)).any():
            det_outcome.append("class_conf")
        elif (same & (ious[k] >= 0.1)).any():
            det_outcome.append("localization")
        else:
            det_outcome.append("background")
    return dets, det_outcome, scored, matched


def counts_at(gts, dets, image_ids, threshold):
    tp = fp = n_gt = 0
    for i in image_ids:
        _, outcome, scored, _ = match_image(gts[i], dets[i], threshold)
        tp += outcome.count("tp")
        fp += sum(o not in ("tp", "ignored") for o in outcome)
        n_gt += len(scored)
    return tp, fp, n_gt


def best_threshold(gts, dets, image_ids):
    best = (0, 0.5)
    for t in np.arange(0.05, 0.96, 0.05):
        tp, fp, n_gt = counts_at(gts, dets, image_ids, t)
        f1 = 2 * tp / max(2 * tp + fp + (n_gt - tp), 1)
        best = max(best, (f1, round(float(t), 2)))
    return best


def height_bin(h):
    return "<25px" if h < 25 else "25-40px" if h < 40 else "40-100px" if h < 100 else ">=100px"


def analyze(dataset, model, gt_dict):
    predictions = json.loads((ROOT / f"predictions/{dataset}/{model}.json").read_text())
    gts, dets = index(gt_dict, predictions)
    image_ids = [im["id"] for im in gt_dict["images"]]
    f1, threshold = best_threshold(gts, dets, image_ids)
    fp_types, missed_by_height, gt_by_height = Counter(), Counter(), Counter()
    missed_by_occ, gt_by_occ, per_class = Counter(), Counter(), defaultdict(Counter)
    for i in image_ids:
        d, outcome, scored, matched = match_image(gts[i], dets[i], threshold)
        for det, o in zip(d, outcome):
            per_class[CLASSES[det["category_id"]]][o] += 1
            if o not in ("tp", "ignored"):
                fp_types[o] += 1
        for g, m in zip(scored, matched):
            hb = height_bin(g["bbox"][3])
            gt_by_height[hb] += 1
            per_class[CLASSES[g["category_id"]]]["gt"] += 1
            if not m:
                missed_by_height[hb] += 1
                per_class[CLASSES[g["category_id"]]]["missed"] += 1
            if "occluded" in g:
                gt_by_occ[g["occluded"]] += 1
                missed_by_occ[g["occluded"]] += not m
    return {
        "model": model, "dataset": dataset, "threshold": threshold, "f1_iou50": round(f1, 4),
        "false_positive_types": dict(fp_types),
        "miss_rate_by_height": {k: round(missed_by_height[k] / gt_by_height[k], 3)
                                for k in ["<25px", "25-40px", "40-100px", ">=100px"] if gt_by_height[k]},
        "gt_count_by_height": dict(gt_by_height),
        "miss_rate_by_occlusion": {str(k): round(missed_by_occ[k] / gt_by_occ[k], 3) for k in sorted(gt_by_occ)},
        "per_class": {k: dict(v) for k, v in per_class.items()},
    }


def draw(dataset, model, gt_dict, threshold, count, out_dir):
    """Save the images with the most errors, showing GT and classified detections."""
    predictions = json.loads((ROOT / f"predictions/{dataset}/{model}.json").read_text())
    gts, dets = index(gt_dict, predictions)
    files = {im["id"]: im["file_name"] for im in gt_dict["images"]}
    scored_images = []
    for i in files:
        d, outcome, scored, matched = match_image(gts[i], dets[i], threshold)
        errors = int((~matched).sum()) + sum(o not in ("tp", "ignored") for o in outcome)
        scored_images.append((errors, i))
    colors = {"tp": (255, 160, 0), "ignored": (160, 160, 160), "duplicate": (0, 165, 255),
              "class_conf": (255, 0, 255), "localization": (0, 255, 255), "background": (0, 0, 255)}
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for _, i in sorted(scored_images, reverse=True)[:count]:
        image = cv2.imread(str(GT[dataset][1] / files[i]))
        d, outcome, scored, matched = match_image(gts[i], dets[i], threshold)
        for g in gts[i]:
            x, y, w, h = map(int, g["bbox"])
            if g.get("iscrowd"):
                cv2.rectangle(image, (x, y), (x + w, y + h), (128, 128, 128), 1)
        for g, m in zip(scored, matched):
            x, y, w, h = map(int, g["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0) if m else (0, 255, 0), 3 if not m else 1)
            if not m:
                cv2.putText(image, "MISS", (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for det, o in zip(d, outcome):
            x, y, w, h = map(int, det["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), colors[o], 2)
            if o != "tp":
                label = f"{CLASSES[det['category_id']]} {det['score']:.2f} {o}"
                cv2.putText(image, label, (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[o], 1)
        path = out_dir / f"failure_{dataset}_{model}_{Path(files[i]).stem}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(path.relative_to(ROOT).as_posix())
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=GT)
    parser.add_argument("--models", nargs="+", help="Defaults to every model in configs/models.json")
    parser.add_argument("--draw", help="Model whose worst images are rendered")
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    models = args.models or [m["id"] for m in json.loads((ROOT / "configs/models.json").read_text())["models"]]
    gt_dict = json.loads(GT[args.dataset][0].read_text())
    if args.dataset == "coco":  # shared classes only, so the two domains are comparable
        gt_dict["annotations"] = [a for a in gt_dict["annotations"] if a["category_id"] in CLASSES]
    results = [analyze(args.dataset, m, gt_dict) for m in models]
    for r in results:
        print(f"{r['model']}: thr {r['threshold']} F1 {r['f1_iou50']} FP {r['false_positive_types']} "
              f"miss-by-height {r['miss_rate_by_height']}")
    out = ROOT / f"results/failure_analysis_{args.dataset}.json"
    payload = {"description": __doc__.strip().splitlines()[0], "classes": CLASSES, "results": results}
    if args.draw:
        thr = next(r["threshold"] for r in results if r["model"] == args.draw)
        payload["figures"] = draw(args.dataset, args.draw, gt_dict, thr, args.count, ROOT / "report/figures")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
