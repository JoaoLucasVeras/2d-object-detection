"""Download and prepare the two evaluation datasets.

COCO val2017 (5,000 images) with the official instances annotations, and the
KITTI 2D object val split (1,496 images) converted to COCO-format ground truth
using configs/kitti_taxonomy.json.
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
URLS = {
    "coco_images": "http://images.cocodataset.org/zips/val2017.zip",
    "coco_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "kitti_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip",
    "kitti_labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
}


def download(url):
    target = RAW / url.rsplit("/", 1)[-1]
    if not target.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}", flush=True)
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        partial.rename(target)
    return target


def prepare_coco(keep_zips):
    out = DATA / "coco"
    ann_path = out / "annotations/instances_val2017.json"
    if not ann_path.exists():
        archive = download(URLS["coco_annotations"])
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            ann_path.write_bytes(z.read("annotations/instances_val2017.json"))
        if not keep_zips:
            archive.unlink()
    if not (out / "val2017").exists():
        archive = download(URLS["coco_images"])
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        if not keep_zips:
            archive.unlink()
    n = len(list((out / "val2017").glob("*.jpg")))
    print(f"COCO val2017: {n} images", flush=True)


def kitti_annotations(label_text, image_id, taxonomy, next_id):
    anns = []
    for line in label_text.strip().splitlines():
        f = line.split()
        cls = f[0]
        truncated, occluded = float(f[1]), int(f[2])
        x1, y1, x2, y2 = map(float, f[4:8])
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        rule = taxonomy[cls]
        base = {"image_id": image_id, "bbox": [x1, y1, w, h], "area": w * h,
                "kitti_class": cls, "truncated": truncated, "occluded": occluded}
        if "evaluate" in rule:
            anns.append({**base, "id": next_id, "category_id": rule["evaluate"], "iscrowd": 0})
            next_id += 1
        for cat in rule.get("ignore_for", []):
            anns.append({**base, "id": next_id, "category_id": cat, "iscrowd": 1})
            next_id += 1
    return anns, next_id


def prepare_kitti(keep_zips):
    out = DATA / "kitti"
    gt_path = out / "annotations/kitti_val_coco.json"
    if gt_path.exists():
        print(f"KITTI ground truth exists: {gt_path}")
        return
    taxonomy_cfg = json.loads((ROOT / "configs/kitti_taxonomy.json").read_text())
    taxonomy = taxonomy_cfg["kitti_to_coco"]
    image_dir = out / "val/images"
    if not image_dir.exists():
        archive = download(URLS["kitti_images"])
        image_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith("images/val/") and name.endswith(".png"):
                    (image_dir / Path(name).name).write_bytes(z.read(name))
        if not keep_zips:
            archive.unlink()
    label_archive = download(URLS["kitti_labels"])
    images, annotations, next_id = [], [], 1
    with zipfile.ZipFile(label_archive) as z:
        for image_id, path in enumerate(sorted(image_dir.glob("*.png")), start=1):
            with Image.open(path) as im:
                width, height = im.size
            images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
            text = z.read(f"training/label_2/{path.stem}.txt").decode()
            anns, next_id = kitti_annotations(text, image_id, taxonomy, next_id)
            annotations.extend(anns)
    categories = [{"id": int(k), "name": v} for k, v in taxonomy_cfg["evaluated_coco_categories"].items()]
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps({
        "info": {"description": "KITTI 2D object val split mapped to COCO categories",
                 "taxonomy": "configs/kitti_taxonomy.json"},
        "images": images, "annotations": annotations, "categories": categories,
    }))
    scored = sum(1 for a in annotations if not a["iscrowd"])
    print(f"KITTI val: {len(images)} images, {scored} scored boxes, "
          f"{len(annotations) - scored} ignore regions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coco", "kitti", "all"], default="all")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()
    if args.dataset in ("coco", "all"):
        prepare_coco(args.keep_zips)
    if args.dataset in ("kitti", "all"):
        prepare_kitti(args.keep_zips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
