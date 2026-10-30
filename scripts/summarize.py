"""Collect run records and metrics into results/summary.md and report figures."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
CFG = json.loads((ROOT / "configs/models.json").read_text())
MODELS = [m["id"] for m in CFG["models"]]
LABEL = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "rtdetr_l": "RT-DETR-L",
         "frcnn_mbv3_320": "FRCNN-MBv3-320", "frcnn_r50_v2": "FRCNN-R50-v2"}
FAMILY_COLOR = {"YOLO": "#2a78d6", "DETR (RT-DETR)": "#eb6834", "Faster R-CNN": "#1baf7a"}
MARKER = {"yolo11n": "o", "yolo11s": "s", "rtdetr_l": "D", "frcnn_mbv3_320": "o", "frcnn_r50_v2": "s"}
DATASET_COLOR = {"COCO": "#2a78d6", "KITTI": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(kind, dataset, name):
    path = ROOT / f"results/{kind}/{dataset}_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def family(model):
    return next(m["family"] for m in CFG["models"] if m["id"] == model)


def main_table(dataset):
    rows = ["| Model | Input | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 | Latency ms (mean / p90) | FPS | Peak mem MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        met, run = load("metrics", dataset, m), load("runs", dataset, m)
        if not met:
            continue
        a, lat = met["all_classes"], run["latency_ms"]
        rows.append(f"| {LABEL[m]} | {run['model']['image_size']} | **{a['AP']}** | {a['AP50']} | {a['AP75']} | "
                    f"{a['AP_small']} | {a['AP_medium']} | {a['AP_large']} | {a['AR100']} | "
                    f"{lat['mean']} / {lat['p90']} | {run['fps_from_mean']} | {run['peak_gpu_memory_allocated_mib']} |")
    return "\n".join(rows)


def cross_domain_table():
    rows = ["| Model | COCO AP (3 cls) | KITTI AP | Drop | COCO AP50 | KITTI AP50 | person C→K | car C→K | truck C→K |",
            "|---|---|---|---|---|---|---|---|---|"]
    data = {}
    for m in MODELS:
        c, k = load("metrics", "coco", m), load("metrics", "kitti", m)
        if not (c and k):
            continue
        cs, ka = c["shared_classes"], k["all_classes"]
        data[m] = (cs, ka)
        cls = " | ".join(f"{cs['per_class'][n]['AP']} → {ka['per_class'][n]['AP']}" for n in ("person", "car", "truck"))
        rows.append(f"| {LABEL[m]} | {cs['AP']} | {ka['AP']} | {round(ka['AP'] - cs['AP'], 1)} | "
                    f"{cs['AP50']} | {ka['AP50']} | {cls} |")
    size_rows = ["| Model | COCO AP_S / AP_M / AP_L (3 cls) | KITTI AP_S / AP_M / AP_L |", "|---|---|---|"]
    for m, (cs, ka) in data.items():
        size_rows.append(f"| {LABEL[m]} | {cs['AP_small']} / {cs['AP_medium']} / {cs['AP_large']} | "
                         f"{ka['AP_small']} / {ka['AP_medium']} / {ka['AP_large']} |")
    return "\n".join(rows), "\n".join(size_rows), data


def kitti_subset_table():
    names = ["occlusion_0_fully_visible", "occlusion_1_partly", "occlusion_2_largely",
             "height_ge40px", "height_25_40px", "height_lt25px", "kitti_moderate"]
    head = ["Fully visible", "Partly occl.", "Largely occl.", "h ≥ 40px", "25–40px", "h < 25px", "KITTI moderate"]
    rows = ["| Model | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for m in MODELS:
        k = load("metrics", "kitti", m)
        if k:
            rows.append(f"| {LABEL[m]} | " + " | ".join(str(k["subsets"][n]["AP"]) for n in names) + " |")
    return "\n".join(rows)


def ablation_table():
    rows = ["| Configuration | KITTI AP | AP50 | AP_S | AP_M | AP_L | Latency ms | Peak mem MiB |", "|---|---|---|---|---|---|---|---|"]
    found = False
    for base, tags in (("rtdetr_l", ["", "_padsq"]), ("yolo11s", ["", "_imgsz1280"]), ("yolo11n", ["", "_imgsz1280"])):
        for tag in tags:
            met, run = load("metrics", "kitti", base + tag), load("runs", "kitti", base + tag)
            if not met:
                continue
            found = found or bool(tag)
            a = met["all_classes"]
            desc = {"": "default", "_padsq": "aspect-preserving (pad to square)", "_imgsz1280": "input 1280"}[tag]
            rows.append(f"| {LABEL[base]} {desc} | **{a['AP']}** | {a['AP50']} | {a['AP_small']} | {a['AP_medium']} | "
                        f"{a['AP_large']} | {run['latency_ms']['mean']} | {run['peak_gpu_memory_allocated_mib']} |")
    return "\n".join(rows) if found else ""


def speed_accuracy_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, dataset in zip(axes, ("coco", "kitti")):
        for m in MODELS:
            met, run = load("metrics", dataset, m), load("runs", dataset, m)
            if not met:
                continue
            x, y = run["latency_ms"]["mean"], met["all_classes"]["AP"]
            ax.scatter(x, y, s=55, marker=MARKER[m], color=FAMILY_COLOR[family(m)],
                       edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(LABEL[m], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
        ax.set_xlabel("Mean latency per image (ms, batch 1, FP32)")
        ax.set_ylabel("mAP@[.50:.95] (%)")
        ax.set_title(f"{'COCO val2017 (80 classes)' if dataset == 'coco' else 'KITTI val (person/car/truck)'}",
                     fontsize=9, color=INK, loc="left")
        xmax = max(load("runs", dataset, m)["latency_ms"]["mean"] for m in MODELS if load("runs", dataset, m))
        ymax = max(load("metrics", dataset, m)["all_classes"]["AP"] for m in MODELS if load("metrics", dataset, m))
        ax.set_xlim(0, xmax * 1.35)
        ax.set_ylim(top=ymax + 3)
    fig.subplots_adjust(wspace=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(FIG / "speed_accuracy.png")
    plt.close(fig)


def cross_domain_figure(data):
    classes = ["person", "car", "truck"]
    fig, axes = plt.subplots(1, len(classes), figsize=(9, 3.0), sharey=True)
    width = 0.38
    for ax, cls in zip(axes, classes):
        xs = range(len(data))
        coco = [cs["per_class"][cls]["AP"] for cs, _ in data.values()]
        kitti = [ka["per_class"][cls]["AP"] for _, ka in data.values()]
        ax.bar([x - width / 2 - 0.01 for x in xs], coco, width, color=DATASET_COLOR["COCO"], label="COCO")
        ax.bar([x + width / 2 + 0.01 for x in xs], kitti, width, color=DATASET_COLOR["KITTI"], label="KITTI")
        ax.set_xticks(list(xs), [LABEL[m] for m in data], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(cls, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AP@[.50:.95] (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.savefig(FIG / "cross_domain_per_class.png")
    plt.close(fig)


def kitti_subset_figure():
    groups = [("occlusion_0_fully_visible", "visible"), ("occlusion_1_partly", "partly occl."),
              ("occlusion_2_largely", "largely occl."), ("height_ge40px", "h≥40px"),
              ("height_25_40px", "25–40px"), ("height_lt25px", "h<25px")]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    models = [m for m in MODELS if load("metrics", "kitti", m)]
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        k = load("metrics", "kitti", m)
        vals = [k["subsets"][g]["AP"] for g, _ in groups]
        xs = [g + (i - (len(models) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, vals, width * 0.92, color=FAMILY_COLOR[family(m)],
               hatch={"o": "", "s": "///", "D": ""}[MARKER[m]] if family(m) != "DETR (RT-DETR)" else "",
               edgecolor="white", linewidth=0.5, label=LABEL[m])
    ax.set_xticks(range(len(groups)), [n for _, n in groups])
    ax.set_ylabel("KITTI AP@[.50:.95] (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(FIG / "kitti_occlusion_height.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cd_table, size_table, data = cross_domain_table()
    sections = [
        "# Results summary\n\nGenerated by `scripts/summarize.py` from `results/runs` and `results/metrics`. "
        "AP values are percentages from pycocotools. Latency is mean per image, batch 1, FP32, on the GPU listed in each run record.",
        "## COCO val2017 (5,000 images, 80 classes)\n\n" + main_table("coco"),
        "Published COCO val mAP for reference: " + ", ".join(
            f"{LABEL[m]} {v}" for m, v in CFG["reference_coco_val_map"].items() if m in LABEL),
        "## KITTI val (1,496 images; person, car, truck)\n\n" + main_table("kitti"),
        "## Cross-domain: COCO restricted to person/car/truck vs KITTI\n\n" + cd_table,
        "### Object-size breakdown\n\n" + size_table,
        "## KITTI AP by occlusion level and box height\n\n" + kitti_subset_table(),
    ]
    abl = ablation_table()
    if abl:
        sections.append("## KITTI ablations: input geometry\n\n" + abl)
    (ROOT / "results/summary.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    speed_accuracy_figure()
    cross_domain_figure(data)
    kitti_subset_figure()
    print((ROOT / "results/summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
