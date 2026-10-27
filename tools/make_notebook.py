"""Builds notebooks/pipeline.ipynb (kept as a script so the notebook diff stays readable)."""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Solar Filament Segmentation Challenge 2026 - full pipeline

This notebook reproduces the submitted predictions (`submission.csv`) end to end:

1. grayscale H-alpha image (2048x2048) -> resize to 1024 -> **4 test-time views** (original, h-flip, v-flip, 180 deg rotation)
2. **Mask R-CNN** (ResNet-50-FPN v2, trained on all annotation categories) -> detections with score >= 0.7
3. each detection -> native-resolution square crop (1.8x context) -> 256x256 **U-Net refiner** -> mask at native resolution
4. **cross-view agreement**: cluster masks across views (IoU > 0.5), keep clusters seen in >= 3 of 4 views, pixel-wise majority vote
5. **panoptic painting** in descending score order (no overlaps), drop masks < 200 px, run-length encode

Only the competition's `test_images` and the two checkpoints stored in `weights/` are needed. Training code is in `training/`, the
experiments behind every table of the report are in `experiments/` (results in `results/`).
"""),
    code("""
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

import cv2
import numpy as np
import torch
import pycocotools.mask as mask_util

from filament import CFG, load_maskrcnn, load_refiner, predict_image, run_directory
from filament.weights import assemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__)
print("pipeline configuration:", CFG)
"""),
    md("""
## 1. Data and weights

`find_dir` looks for the competition's `test_images` folder under `/kaggle/input` or `./data`.
The two checkpoints are stored in this repository (`weights/`) as parts of at most 50 MB, because GitHub rejects larger
files. `assemble` concatenates the parts and verifies every part and the final file by SHA-256 (`weights/MANIFEST.json`);
the assembled files are written to `weights/assembled/`. No download and no re-training are needed.
"""),
    code("""
def find_dir(name, roots=("/kaggle/input", "data", "../data")):
    for root in map(Path, roots):
        if root.exists():
            for p in root.rglob(name):
                if p.is_dir() and any(p.iterdir()):
                    return p
    raise FileNotFoundError(f"{name} not found; put the competition data under ./data")


TEST_DIR = find_dir("test_images")
detector_path = assemble("maskrcnn_allcat_fold0.pth")
refiner_path = assemble("refiner.pth")
print("test images:", TEST_DIR, "->", len(list(TEST_DIR.iterdir())), "files")
"""),
    code("""
detector = load_maskrcnn(detector_path, device, CFG["image_size"])
refiner = load_refiner(refiner_path, device)
"""),
    md("""
## 2. Predict the test set

`run_directory` runs steps 1-5 for every image and writes one RLE-counts row per filament
(`filament_id = <image id>_<n>`). About 10 minutes on one T4 GPU for the 180 test images.
"""),
    code("""
rows = run_directory(TEST_DIR, detector, refiner, device, out_csv="submission.csv")
"""),
    md("## 3. Check the file and look at a few predictions"),
    code("""
import pandas as pd
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["filament_id", "segmentation_rle"] and sub.filament_id.is_unique
sub["image"] = sub.filament_id.str.rsplit("_", n=1).str[0]
print(len(sub), "filaments in", sub.image.nunique(), "images; mean", round(sub.groupby("image").size().mean(), 2), "per image")
# every row must decode to a 2048x2048 mask
m = mask_util.decode({"size": [2048, 2048], "counts": sub.segmentation_rle.iloc[0].encode()})
print("first mask:", m.shape, int(m.sum()), "px")
"""),
    code("""
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, image_id in zip(axs, sub.image.drop_duplicates().sample(3, random_state=0)):
    path = next(TEST_DIR.glob(image_id + ".*"))
    img = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply(cv2.imread(str(path), 0)), cv2.COLOR_GRAY2RGB)
    for r in sub[sub.image == image_id].segmentation_rle:
        mask = mask_util.decode({"size": [2048, 2048], "counts": r.encode()})
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 60, 60), 3)
    ax.imshow(img); ax.set_title(image_id); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
## 4. Re-training (optional)

* `training/train_detector.py` - Mask R-CNN, all four annotation categories, fold 0 (`CFG['folds_to_train']`)
* `training/train_refiner.py` - crop U-Net on native-resolution crops around annotated filaments

Both scripts expect the competition data under `/kaggle/input` or `./data`. Held-out evaluation with the official
per-annotator Panoptic Quality (`src/filament/metrics.py`) is in `experiments/`.
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
out = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
