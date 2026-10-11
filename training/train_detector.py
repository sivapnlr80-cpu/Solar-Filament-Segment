#!/usr/bin/env python3
"""
Filament Mask R-CNN training with ALL annotation categories (v49)
==================================================================
v36 dropped category 3 ("Unidentifiable" chirality, 37% of all annotations) and category 4.
The official ground truth scores every annotated filament, so those were never learned.
This kernel collapses all categories into one filament class.

Folds are defined EXACTLY as in v36 (KFold over files that have >=1 category-1/2 annotation,
seed 42) so held-out images are identical; the training set additionally contains
category-3/4 annotations and any file that only has such annotations.

Output: maskrcnn_allcat_fold{k}.pth for the folds listed in CFG['folds_to_train'].

Runtime estimate on T4: ~6 epochs per fold * ~9 min/epoch * 5 folds =~ 4.5h.
Safe within Kaggle's 9h kernel timeout.

Inference-time ensembling happens elsewhere (inference kernel averages the
5 models' predicted-mask probabilities, then applies score-ordered
panoptic painting).
"""

import os
import sys
import gc
import json
import time
import subprocess
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_HOME"] = "/tmp/torch_cache"

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

try:
    import pycocotools.mask as mask_utils
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pycocotools"])
    import pycocotools.mask as mask_utils

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CFG = {
    'image_size':        1024,
    'batch_size':        2,
    'grad_accum':        4,
    'epochs_per_fold':  10,       # reduced from 8 (v28 gains plateaued by ep 5)
    'lr':                1e-4,
    'weight_decay':      1e-4,
    'num_workers':       2,
    'num_classes':       2,
    'keep_categories':   {1, 2},        # ONLY used to define the fold split (identical to v36)
    'train_categories':  {1, 2, 3, 4},  # all annotated filaments are trained on
    'folds_to_train':    [0],
    'min_ann_area':      50,
    'n_folds':           5,
    'seed':              42,
}


def seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | torch {torch.__version__}")


def find_paths():
    roots = [Path("/kaggle/input"), Path("data")]
    def first_dir(name):
        for sr in roots:
            if sr.exists():
                for m in sr.rglob(name):
                    if m.is_dir() and any(m.iterdir()):
                        return m
        return None
    def first_json():
        for sr in roots:
            if sr.exists():
                for m in list(sr.rglob("*Annotations*.json")) + list(sr.rglob("*annotations*.json")):
                    if m.is_file(): return m
        return None
    return first_dir("train_images"), first_json()


TRAIN_IMG_DIR, ANNOTATIONS = find_paths()
print(f"TRAIN_IMG_DIR: {TRAIN_IMG_DIR}")
print(f"ANNOTATIONS  : {ANNOTATIONS}")
assert TRAIN_IMG_DIR and ANNOTATIONS


def load_metadata():
    with open(ANNOTATIONS, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    keep = CFG['train_categories']
    id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    img_anns = {}
    total, kept = 0, 0
    for ann in coco.get('annotations', []):
        total += 1
        if ann.get('category_id') not in keep:
            continue
        if ann.get('area', 0) < CFG['min_ann_area']:
            continue
        kept += 1
        img_anns.setdefault(ann['image_id'], []).append(ann)
    records = [{'image_id': iid, 'file_name': fn, 'anns': img_anns.get(iid, [])}
               for iid, fn in id_to_file.items()]
    df = pd.DataFrame(records)
    df = df[df['anns'].map(len) > 0].reset_index(drop=True)
    print(f"Metadata: {len(df)} images, {kept}/{total} annotations kept "
          f"(cats {sorted(keep)}, area>={CFG['min_ann_area']})")
    return df


df_meta = load_metadata()


def v36_split_files():
    with open(ANNOTATIONS, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    files = set()
    for ann in coco.get('annotations', []):
        if ann.get('category_id') in CFG['keep_categories'] and ann.get('area', 0) >= CFG['min_ann_area']:
            files.add(id_to_file[ann['image_id']])
    return np.array(sorted(files))


unique_files = v36_split_files()
print(f"Files used to define folds (v36-identical): {len(unique_files)} | "
      f"files with any training annotation: {df_meta['file_name'].nunique()}")


class FilamentInstanceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, size: int, train_mode: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.size = size
        self.train_mode = train_mode

    def __len__(self):
        return len(self.df)

    def _polygons_to_mask(self, polys, h, w):
        m = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(m, [pts], 1)
        return m

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self.img_dir / row['file_name']
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((2048, 2048), dtype=np.uint8)
        H, W = img.shape[:2]

        inst_masks = []
        for ann in row['anns']:
            seg = ann.get('segmentation')
            if not seg: continue
            if isinstance(seg, list):
                m = self._polygons_to_mask(seg, H, W)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg).astype(np.uint8)
            else:
                continue
            if m.sum() > 0: inst_masks.append(m)

        s = self.size
        img_r = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        masks_r = []
        for m in inst_masks:
            mr = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
            if mr.sum() > 0: masks_r.append(mr)

        if self.train_mode and len(masks_r) > 0:
            if np.random.rand() < 0.5:
                img_r = np.ascontiguousarray(img_r[:, ::-1])
                masks_r = [np.ascontiguousarray(m[:, ::-1]) for m in masks_r]
            if np.random.rand() < 0.5:
                img_r = np.ascontiguousarray(img_r[::-1, :])
                masks_r = [np.ascontiguousarray(m[::-1, :]) for m in masks_r]

        img_t = torch.from_numpy(img_r).float().unsqueeze(0).repeat(3, 1, 1) / 255.0

        if len(masks_r) == 0:
            return img_t, {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, s, s), dtype=torch.uint8),
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.uint8),
            }

        boxes, keep_idx = [], []
        for i, m in enumerate(masks_r):
            ys, xs = np.where(m > 0)
            if len(xs) == 0: continue
            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max()) + 1.0, float(ys.max()) + 1.0
            if x1 <= x0 + 1 or y1 <= y0 + 1: continue
            boxes.append([x0, y0, x1, y1]); keep_idx.append(i)

        if len(boxes) == 0:
            return img_t, {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, s, s), dtype=torch.uint8),
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.uint8),
            }

        masks_t = torch.from_numpy(np.stack([masks_r[i] for i in keep_idx], axis=0)).to(torch.uint8)
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        return img_t, {
            'boxes': boxes_t,
            'labels': torch.ones((len(boxes),), dtype=torch.int64),
            'masks': masks_t,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': area_t,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.uint8),
        }


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def build_model():
    m = maskrcnn_resnet50_fpn_v2(weights='DEFAULT',
                                 min_size=CFG['image_size'],
                                 max_size=CFG['image_size'])
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, CFG['num_classes'])
    in_feats_m = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_m, 256, CFG['num_classes'])
    return m


def move_targets(targets, device):
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]


def train_one_epoch(model, loader, opt, device, epoch, fold, grad_accum):
    model.train()
    total, n = 0.0, 0
    opt.zero_grad()
    for i, (imgs, targets) in enumerate(loader):
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        targets = move_targets(targets, device)
        if all(len(t['boxes']) == 0 for t in targets):
            continue
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values()) / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step(); opt.zero_grad()
        total += float(sum(loss_dict.values()).detach())
        n += 1
    opt.step(); opt.zero_grad()
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices = []
    for imgs, targets in loader:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        outs = model(imgs)
        for out, tgt in zip(outs, targets):
            H, W = imgs[0].shape[-2:]
            gt = torch.zeros((H, W), dtype=torch.float32)
            for m in tgt['masks']:
                gt = torch.maximum(gt, m.float().cpu())
            if len(out['masks']) == 0:
                pr = torch.zeros((H, W), dtype=torch.float32)
            else:
                pr = (out['masks'].squeeze(1) > 0.5).float().cpu().max(0).values
            inter = float((pr * gt).sum())
            total = float(pr.sum() + gt.sum())
            dices.append((2 * inter + 1e-6) / (total + 1e-6))
    return float(np.mean(dices)) if dices else 0.0


def train_fold(fold_id: int, train_files: np.ndarray, val_files: np.ndarray):
    val_set = set(val_files.tolist())
    tr_df = df_meta[~df_meta['file_name'].isin(val_set)].reset_index(drop=True)
    va_df = df_meta[ df_meta['file_name'].isin(val_set)].reset_index(drop=True)
    print(f"\n===== Fold {fold_id} / {CFG['n_folds']} =====")
    print(f"  {len(tr_df)} train / {len(va_df)} val rows | {len(train_files)}/{len(val_files)} files")

    tr_ds = FilamentInstanceDataset(tr_df, TRAIN_IMG_DIR, CFG['image_size'], train_mode=True)
    va_ds = FilamentInstanceDataset(va_df, TRAIN_IMG_DIR, CFG['image_size'], train_mode=False)
    tr_ld = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=True,
                       num_workers=CFG['num_workers'], collate_fn=collate_fn,
                       pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=CFG['batch_size'], shuffle=False,
                       num_workers=CFG['num_workers'], collate_fn=collate_fn,
                       pin_memory=True)

    model = build_model().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs_per_fold'], eta_min=1e-6)

    ckpt_path = f"/kaggle/working/maskrcnn_allcat_fold{fold_id}.pth"
    best = 0.0
    for ep in range(1, CFG['epochs_per_fold'] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, tr_ld, opt, device, ep, fold_id, CFG['grad_accum'])
        va_dice = evaluate(model, va_ld, device)
        sch.step()
        elapsed = (time.time() - t0) / 60
        print(f"  [fold {fold_id}] ep {ep}/{CFG['epochs_per_fold']} | "
              f"tr_loss={tr_loss:.4f} va_dice={va_dice:.4f} ({elapsed:.1f}m)")
        if va_dice > best:
            best = va_dice
            torch.save({
                'state_dict': model.state_dict(),
                'cfg': CFG,
                'fold': fold_id,
                'best_val_dice': best,
                'epoch': ep,
            }, ckpt_path)

    print(f"  [fold {fold_id}] best va_dice = {best:.4f} -> {ckpt_path}")
    del model, opt, sch, tr_ld, va_ld, tr_ds, va_ds
    torch.cuda.empty_cache(); gc.collect()
    return best


def main():
    kf = KFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    fold_scores = []
    total_start = time.time()
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(unique_files)):
        if fold_id not in CFG['folds_to_train']:
            continue
        best = train_fold(fold_id, unique_files[tr_idx], unique_files[va_idx])
        fold_scores.append(best)
        elapsed_total = (time.time() - total_start) / 60
        print(f"  [progress] fold {fold_id} done | total elapsed {elapsed_total:.1f}m")

    print("\n===== all-category training complete =====")
    for i, s in enumerate(fold_scores):
        print(f"  fold {i}: val_dice={s:.4f}")
    print(f"  mean val_dice: {np.mean(fold_scores):.4f}  std: {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()
