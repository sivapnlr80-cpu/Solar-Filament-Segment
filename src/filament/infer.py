"""Inference pipeline.

grayscale image -> 4 TTA views -> Mask R-CNN -> score filter -> native-resolution crop U-Net refinement
-> cross-view clustering (>= 3 of 4 views, majority vote) -> score-ordered panoptic painting -> RLE.
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from tqdm import tqdm

CFG = {
    "image_size": 1024,         # detector input (the 2048 px image is resized)
    "score_threshold": 0.70,    # detections below this score are discarded
    "coarse_mask_thr": 0.30,    # detector soft mask -> coarse mask used to place the refiner crop
    "refiner_threshold": 0.50,
    "min_area": 200,            # px at native resolution
    "iou_match_threshold": 0.5,  # cross-view clustering
    "min_agreement": 3,         # a filament must appear in >= 3 of the 4 views
    "crop_size": 256,
    "crop_context": 1.8,
    "crop_minimum": 96,
}

# (name, forward transform on a CHW tensor, inverse transform on an HxW mask)
TTA_VIEWS = [
    ("orig", lambda t: t, lambda m: m),
    ("hflip", lambda t: torch.flip(t, [2]), lambda m: np.ascontiguousarray(m[:, ::-1])),
    ("vflip", lambda t: torch.flip(t, [1]), lambda m: np.ascontiguousarray(m[::-1, :])),
    ("rot180", lambda t: torch.rot90(t, 2, [1, 2]), lambda m: np.ascontiguousarray(np.rot90(m, 2))),
]


def square_bounds(mask, context, minimum):
    """Square window around a binary mask: side = context x extent (>= minimum), clipped to the image."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, min(minimum, w), min(minimum, h)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(h, w))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return int(x0), int(y0), int(x1), int(y1)


@torch.no_grad()
def detect_and_refine(image, detector, refiner, device, cfg=CFG):
    """Return [(score, rle, view_index)] of refined filament masks at native resolution."""
    oh, ow = image.shape
    s = cfg["image_size"]
    t = torch.from_numpy(cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)).float() \
        .unsqueeze(0).repeat(3, 1, 1) / 255.0
    views = [fwd(t).contiguous().to(device) for _, fwd, _ in TTA_VIEWS]
    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            outs = detector(views)
    else:
        outs = detector(views)

    cs = cfg["crop_size"]
    dets = []
    for view_idx, out in enumerate(outs):
        inv = TTA_VIEWS[view_idx][2]
        scores = out["scores"].cpu().numpy()
        soft_all = out["masks"].cpu().float().numpy()
        crops, meta = [], []
        for j in range(len(scores)):
            if scores[j] < cfg["score_threshold"]:
                continue
            soft = inv(cv2.resize(soft_all[j, 0], (ow, oh), interpolation=cv2.INTER_LINEAR))
            coarse = (soft > cfg["coarse_mask_thr"]).astype(np.uint8)
            if int(coarse.sum()) < cfg["min_area"]:
                continue
            x0, y0, x1, y1 = square_bounds(coarse, cfg["crop_context"], cfg["crop_minimum"])
            crops.append(cv2.resize(image[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR))
            meta.append((float(scores[j]), x0, y0, x1, y1))
        if not crops:
            continue
        x = torch.from_numpy(np.stack(crops)).float().unsqueeze(1).repeat(1, 3, 1, 1).to(device) / 255.0
        refined = refiner(x).sigmoid().cpu().numpy()[:, 0]
        for k, (score, x0, y0, x1, y1) in enumerate(meta):
            rb = (cv2.resize(refined[k], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                  >= cfg["refiner_threshold"]).astype(np.uint8)
            if int(rb.sum()) < cfg["min_area"]:
                continue
            full = np.zeros((oh, ow), np.uint8)
            full[y0:y1, x0:x1] = rb
            dets.append((score, mask_util.encode(np.asfortranarray(full)), view_idx))
    return dets


def cluster_by_iou(dets, iou_thresh):
    """Union-find over detections from different views whose masks overlap with IoU > iou_thresh."""
    n = len(dets)
    if n == 0:
        return []
    rles = [d[1] for d in dets]
    iou = mask_util.iou(rles, rles, [0] * n)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dets[i][2] != dets[j][2] and iou[i, j] > iou_thresh:
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def paint_panoptic(cands, min_area, shape):
    """Paint candidates in descending score order so that final masks never overlap."""
    claimed = np.zeros(shape, np.uint8)
    kept = []
    for score, mask in sorted(cands, key=lambda c: -c[0]):
        remaining = mask * (1 - claimed)
        if int(remaining.sum()) < min_area:
            continue
        claimed |= remaining
        kept.append((score, remaining))
    return kept


def consolidate(dets, shape, cfg=CFG):
    """Cross-view agreement + majority vote + panoptic painting -> [(score, binary mask)]."""
    survivors = []
    for cluster in cluster_by_iou(dets, cfg["iou_match_threshold"]):
        if len({dets[i][2] for i in cluster}) < cfg["min_agreement"]:
            continue
        votes = np.zeros(shape, np.int16)
        for i in cluster:
            votes += mask_util.decode(dets[i][1]).astype(np.int16)
        consensus = (votes >= len(cluster) // 2 + 1).astype(np.uint8)
        if int(consensus.sum()) < cfg["min_area"]:
            continue
        survivors.append((float(np.mean([dets[i][0] for i in cluster])), consensus))
    return paint_panoptic(survivors, cfg["min_area"], shape)


def predict_image(image, detector, refiner, device, cfg=CFG):
    """image: HxW uint8 grayscale -> [(score, binary mask)] of non-overlapping filaments."""
    dets = detect_and_refine(image, detector, refiner, device, cfg)
    return consolidate(dets, image.shape, cfg) if dets else []


def rle_counts(mask):
    """Competition format: RLE counts string only (image size is fixed at 2048 x 2048)."""
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("utf-8")


def run_directory(image_dir, detector, refiner, device, out_csv, cfg=CFG):
    """Predict every image in ``image_dir`` and write the submission CSV (filament_id, segmentation_rle)."""
    image_dir = Path(image_dir)
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    rows, n_empty, t0 = [], 0, time.time()
    for path in tqdm(files, desc="predict"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        kept = predict_image(image, detector, refiner, device, cfg)
        if not kept:
            n_empty += 1
        for n, (_, mask) in enumerate(kept, start=1):
            rows.append((f"{path.stem}_{n}", rle_counts(mask)))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    print(f"{len(rows)} filaments in {len(files) - n_empty}/{len(files)} images ({n_empty} empty), "
          f"{time.time() - t0:.0f}s -> {out_csv}")
    return rows
