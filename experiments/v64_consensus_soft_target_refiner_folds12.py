#!/usr/bin/env python3
"""
A/B on fold-0 held-out images: v36 detector (categories 1,2 only) vs all-category detector.
Official per-annotator PQ + recall per GT category (1 Left, 2 Right, 3 Unidentifiable).

- GT = every annotator-image entry (id "<annotator>-<obs>") scored separately against ALL
  predictions of that observation (TP: IoU>0.5, FP/FN per entry, PQ = sumIoU/(TP+.5FP+.5FN)).
- Fold k detector (maskrcnn_v36_fold{k}) is evaluated ONLY on its held-out fold
  (KFold(5, shuffle, seed=42) over sorted unique files, same as training).
- A CropUNet refiner is retrained per fold on the other folds' crops, so the refiner never sees
  validation images.
- Sweeps: mode (detector-only / refiner thr), score thr, single view vs 4-view TTA agreement.
- Also prints the inter-annotator ceiling under the official metric.
Limitation: the 5-model ensemble agreement cannot be validated OOF (each image is in-train for
4 of 5 folds); within-model 4-view TTA agreement is the closest proxy.
"""
import os, sys, gc, time, json, subprocess, warnings, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore")
try:
    from pycocotools.coco import COCO
    import pycocotools.mask as mask_util
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pycocotools"])
    from pycocotools.coco import COCO
    import pycocotools.mask as mask_util

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

SEED = 42
N_FOLDS = 5
IMG = 1024
MIN_AREA = 200
SCORE_FLOOR = 0.30
COARSE_THR = 0.30
MASK_THR = 0.50
REF_THRS = [0.3, 0.4, 0.5, 0.6]
CROP, CONTEXT, MIN_CROP = 256, 1.8, 96
REF_EPOCHS = 10
SCORE_THRS = [0.4, 0.5, 0.6, 0.7]
SANITY_FILES = 20

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def find_dir(name):
    for sr in [Path("/kaggle/input"), Path("data")]:
        if sr.exists():
            for m in sr.rglob(name):
                if m.is_dir() and any(m.iterdir()): return m
    return None


def find_files(pat):
    out = []
    for sr in [Path("/kaggle/input"), Path(".")]:
        if sr.exists():
            out += [m for m in sr.rglob(pat) if m.is_file()]
    return sorted(out)


TRAIN_DIR = find_dir("train_images")
ANN_PATH = find_files("MAGFiLO_1.0_Annotations_kaggle2026_train.json")[0]
NEW_CKPTS = {int(p.stem[-1]): p for p in find_files("maskrcnn_allcat_fold*.pth")}
assert TRAIN_DIR and {0, 1, 2} <= set(NEW_CKPTS), NEW_CKPTS
print("allcat ckpts:", NEW_CKPTS)

# ---------------------------------------------------------------- data / GT
coco = COCO(str(ANN_PATH))
print("Categories:", {c['id']: c['name'] for c in coco.dataset['categories']})
cat_counts = defaultdict(int)
for a in coco.dataset['annotations']:
    cat_counts[a['category_id']] += 1
print("Annotations per category:", dict(cat_counts))

file_of = {i: info['file_name'] for i, info in coco.imgs.items()}
gt_rles, gt_cats = {}, {}
for iid in coco.imgs:
    rl, ct = [], []
    for a in coco.imgToAnns.get(iid, []):
        r = coco.annToRLE(a)
        if float(mask_util.area(r)) > 0:
            rl.append(r); ct.append(a['category_id'])
    if rl:
        gt_rles[iid] = rl; gt_cats[iid] = ct
entries_by_file = defaultdict(list)
entries_cats_by_file = defaultdict(list)
for iid, rl in gt_rles.items():
    entries_by_file[file_of[iid]].append(rl)
    entries_cats_by_file[file_of[iid]].append(gt_cats[iid])
print(f"GT entries (annotator-images): {len(gt_rles)} over {len(entries_by_file)} observations; "
      f"entries/obs: {pd.Series([len(v) for v in entries_by_file.values()]).value_counts().sort_index().to_dict()}")

# training-fold definition (identical to training script: cats {1,2}, area>=50, >=1 ann)
kept = defaultdict(int)
for a in coco.dataset['annotations']:
    if a['category_id'] in (1, 2) and a.get('area', 0) >= 50:
        kept[a['image_id']] += 1
unique_files = np.array(sorted({file_of[i] for i in kept}))
print(f"unique_files: {len(unique_files)}")
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(unique_files))


# ---------------------------------------------------------------- official PQ
def pq_stats(pred_rles, entries):
    S = TP = FP = FN = 0.0
    n_pred = len(pred_rles)
    for gt in entries:
        n_gt = len(gt)
        if n_pred == 0:
            FN += n_gt; continue
        iou = np.asarray(mask_util.iou(gt, pred_rles, [0] * n_pred))
        hit = iou > 0.5
        S += float(iou[hit].sum()); TP += int(hit.sum())
        FP += int((hit.sum(0) == 0).sum()); FN += int((hit.sum(1) == 0).sum())
    return np.array([S, TP, FP, FN], dtype=np.float64)


def cat_hits(pred_rles, entries, cats, store):
    n_pred = len(pred_rles)
    for gt, ct in zip(entries, cats):
        if n_pred == 0:
            hit = np.zeros(len(gt), bool)
        else:
            hit = (np.asarray(mask_util.iou(gt, pred_rles, [0] * n_pred)) > 0.5).any(1)
        for h, c in zip(hit, ct):
            store[c][0] += int(h); store[c][1] += 1


def pq_of(st):
    d = st[1] + 0.5 * st[2] + 0.5 * st[3]
    return float(st[0] / d) if d > 0 else 0.0


# inter-annotator ceiling under the official metric (leave-one-annotator-out)
st = np.zeros(4)
for fn, ents in entries_by_file.items():
    if len(ents) < 2: continue
    for a in range(len(ents)):
        st += pq_stats(ents[a], [e for b, e in enumerate(ents) if b != a])
CEILING = pq_of(st)
print(f"\n[CEILING] inter-annotator PQ under official metric: {CEILING:.4f} "
      f"(TP={int(st[1])} FP={int(st[2])} FN={int(st[3])})\n", flush=True)


# ---------------------------------------------------------------- refiner
def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
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


class ConvBlock(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        e = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(e.conv1, e.bn1, e.relu); self.pool = e.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = e.layer1, e.layer2, e.layer3, e.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        s0 = self.stem(x); s1 = self.layer1(self.pool(s0)); s2 = self.layer2(s1)
        s3 = self.layer3(s2); s4 = self.layer4(s3)
        d = self.dec4(torch.cat([self.up4(s4), s3], 1)); d = self.dec3(torch.cat([self.up3(d), s2], 1))
        d = self.dec2(torch.cat([self.up2(d), s1], 1)); d = self.dec1(torch.cat([self.up1(d), s0], 1))
        return self.head(self.up0(d))


class CropDS(Dataset):
    def __init__(self, imgs, msks):
        self.imgs, self.msks = imgs, msks
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        im = torch.from_numpy(self.imgs[i].copy()).float() / 255.0
        mk = torch.from_numpy(self.msks[i].copy()).float().unsqueeze(0)
        k = random.randrange(4)
        im, mk = torch.rot90(im, k, (0, 1)), torch.rot90(mk, k, (1, 2))
        if random.random() < 0.5: im, mk = torch.flip(im, (1,)), torch.flip(mk, (2,))
        if random.random() < 0.5: im = im ** random.uniform(0.85, 1.20)
        if random.random() < 0.5: im = torch.clamp(im * random.uniform(0.90, 1.10), 0, 1)
        return im.unsqueeze(0).repeat(3, 1, 1), mk


def train_refiner(imgs, msks, epochs):
    model = CropUNet(True).to(device)
    ld = DataLoader(CropDS(imgs, msks), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(epochs):
        model.train()
        for x, y in ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                lg = model(x)
                p = lg.sigmoid()
                inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(lg, y) + \
                    0.5 * (1 - (2 * inter + 1) / (den + 1)).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
    model.eval()
    return model


print("Building crop cache (most complete annotation set per file)...", flush=True)
best_entry = {}
for iid, rl in gt_rles.items():
    fn = file_of[iid]
    n = len(coco.imgToAnns.get(iid, []))
    if fn not in best_entry or n >= best_entry[fn][1]:
        best_entry[fn] = (iid, n)
crop_imgs, crop_msks, crop_files = [], [], []
for fn in tqdm(unique_files):
    iid = best_entry[fn][0]
    image = cv2.imread(str(TRAIN_DIR / fn), cv2.IMREAD_GRAYSCALE)
    for a in coco.imgToAnns.get(iid, []):
        m = coco.annToMask(a).astype(np.uint8)
        if m.sum() < 10: continue
        x0, y0, x1, y1 = square_bounds(m)
        crop_imgs.append(cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR))
        crop_msks.append(cv2.resize(m[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_NEAREST))
        crop_files.append(fn)
crop_imgs, crop_msks, crop_files = np.stack(crop_imgs), np.stack(crop_msks), np.array(crop_files)
print(f"crops: {len(crop_imgs)}", flush=True)


# ---------------------------------------------------------------- detector + TTA
def build_mrcnn():
    m = maskrcnn_resnet50_fpn_v2(weights=None, min_size=IMG, max_size=IMG)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return m


VIEWS = [
    ('orig',   lambda t: t,                         lambda m: m),
    ('hflip',  lambda t: torch.flip(t, [2]),        lambda m: np.ascontiguousarray(m[:, ::-1])),
    ('vflip',  lambda t: torch.flip(t, [1]),        lambda m: np.ascontiguousarray(m[::-1, :])),
    ('rot180', lambda t: torch.rot90(t, 2, [1, 2]), lambda m: np.ascontiguousarray(np.rot90(m, 2))),
]


def enc(b):
    return mask_util.encode(np.asfortranarray(b.astype(np.uint8)))


@torch.no_grad()
def detect(model, refiner, fn, views):
    image = cv2.imread(str(TRAIN_DIR / fn), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    t = torch.from_numpy(cv2.resize(image, (IMG, IMG), interpolation=cv2.INTER_LINEAR)).float() \
        .unsqueeze(0).repeat(3, 1, 1) / 255.0
    batch = [views[v][1](t).contiguous().to(device) for v in range(len(views))]
    with torch.cuda.amp.autocast():
        outs = model(batch)
    dets = []
    for vi, out in enumerate(outs):
        inv = views[vi][2]
        scores = out['scores'].cpu().numpy(); soft_all = out['masks'].cpu().float().numpy()
        crops, meta = [], []
        for j in range(len(scores)):
            if scores[j] < SCORE_FLOOR: continue
            soft = inv(cv2.resize(soft_all[j, 0], (w, h), interpolation=cv2.INTER_LINEAR))
            d = {'score': float(scores[j]), 'view': vi, 'det': None, 'ref': {}}
            b5 = soft > MASK_THR
            if int(b5.sum()) >= MIN_AREA: d['det'] = enc(b5)
            coarse = soft > COARSE_THR
            if refiner is not None and int(coarse.sum()) >= MIN_AREA:
                x0, y0, x1, y1 = square_bounds(coarse)
                crops.append(cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR))
                meta.append((d, x0, y0, x1, y1))
            dets.append(d)
        if crops:
            x = torch.from_numpy(np.stack(crops)).float().unsqueeze(1).repeat(1, 3, 1, 1).to(device) / 255.0
            pr = refiner(x).sigmoid().cpu().numpy()[:, 0]
            for (d, x0, y0, x1, y1), p in zip(meta, pr):
                pm = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                for rt in REF_THRS:
                    rb = pm >= rt
                    if int(rb.sum()) >= MIN_AREA:
                        full = np.zeros((h, w), np.uint8); full[y0:y1, x0:x1] = rb
                        d['ref'][rt] = enc(full)
    return dets


# ---------------------------------------------------------------- post-processing sweep
class Obs:
    def __init__(self, dets):
        self.dets, self.cache = dets, {}
    def rle(self, i, mode):
        d = self.dets[i]
        return d['det'] if mode == 'det' else d['ref'].get(mode)
    def mask(self, i, mode):
        k = (i, mode)
        if k not in self.cache:
            self.cache[k] = mask_util.decode(self.rle(i, mode)).astype(np.uint8)
        return self.cache[k]


def paint(cands):
    """cands: list of (score, rle, mask_or_None, decode_fn) -> list of rle"""
    cands = sorted(cands, key=lambda c: -c[0])
    n = len(cands)
    if n == 0: return []
    rles = [c[1] for c in cands]
    if n > 1:
        iou = np.asarray(mask_util.iou(rles, rles, [0] * n)); np.fill_diagonal(iou, 0)
        no_overlap = not (iou > 0).any()
    else:
        no_overlap = True
    if no_overlap:
        return [r for r in rles if float(mask_util.area(r)) >= MIN_AREA]
    claimed = None; out = []
    for sc, r, m, dec in cands:
        b = m if m is not None else dec()
        if claimed is None: claimed = np.zeros_like(b)
        rem = b * (1 - claimed)
        if int(rem.sum()) < MIN_AREA: continue
        claimed |= rem; out.append(enc(rem))
    return out


def predict(obs, mode, thr, views_cfg):
    idx = [i for i, d in enumerate(obs.dets)
           if d['score'] >= thr and obs.rle(i, mode) is not None]
    if views_cfg[0] == 'orig':
        idx = [i for i in idx if obs.dets[i]['view'] == 0]
        return paint([(obs.dets[i]['score'], obs.rle(i, mode), None, (lambda i=i: obs.mask(i, mode)))
                      for i in idx])
    agree = views_cfg[1]
    n = len(idx)
    if n == 0: return []
    rles = [obs.rle(i, mode) for i in idx]
    iou = np.asarray(mask_util.iou(rles, rles, [0] * n))
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(n):
        for b in range(a + 1, n):
            if obs.dets[idx[a]]['view'] == obs.dets[idx[b]]['view']: continue
            if iou[a, b] > 0.5: parent[find(a)] = find(b)
    groups = defaultdict(list)
    for a in range(n): groups[find(a)].append(a)
    cands = []
    for g in groups.values():
        if len({obs.dets[idx[a]]['view'] for a in g}) < agree: continue
        votes = sum(obs.mask(idx[a], mode).astype(np.int16) for a in g)
        cons = (votes >= (len(g) // 2 + 1)).astype(np.uint8)
        if int(cons.sum()) < MIN_AREA: continue
        cands.append((float(np.mean([obs.dets[idx[a]]['score'] for a in g])), enc(cons), cons, None))
    return paint(cands)




# ================================================================ v64: consensus soft-target refiner
# Annotators disagree on a filament's extent (localisation misses are ~half of all misses and their predicted
# area is smaller / larger than the GT equally often). Under per-annotator PQ the best mask is the one that
# maximises expected IoU against ANY annotator, i.e. roughly the majority-vote region. We therefore train the
# refiner on the mean of the masks that the annotators who drew the filament produced (soft target).
#
# Variants per fold (same detector, same crops/windows, refiners trained ONLY on the other folds):
#   base_a, base_b : current recipe (most complete annotator set per image, hard masks); two random restarts = noise level
#   allhard        : crops around EVERY annotator's filament, each with its own hard mask  (isolates the data-size effect)
#   soft           : same crops as allhard, target = mean mask of the annotators that drew this filament
OUT = Path('/kaggle/working'); OUT.mkdir(parents=True, exist_ok=True)
FOLDS_TO_RUN = [1, 2]
CFGS = [(0.7, 3), (0.8, 3)]
GROUP_IOU = 0.3


def load_det(path):
    m = build_mrcnn().to(device)
    st_ = torch.load(path, map_location=device)
    m.load_state_dict(st_['state_dict'] if isinstance(st_, dict) and 'state_dict' in st_ else st_)
    return m.eval()


def build_consensus_crops():
    imgs, hard, soft, files = [], [], [], []
    n_groups, n_multi = 0, 0
    for fn in tqdm(unique_files, desc="consensus crops"):
        ents = entries_by_file[fn]
        items = [(a, g) for a, gt in enumerate(ents) for g in range(len(gt))]
        if not items:
            continue
        image = cv2.imread(str(TRAIN_DIR / fn), cv2.IMREAD_GRAYSCALE)
        idx = {it: k for k, it in enumerate(items)}
        parent = list(range(len(items)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(ents)):
            for b in range(a + 1, len(ents)):
                if not ents[a] or not ents[b]:
                    continue
                iou = np.asarray(mask_util.iou(ents[a], ents[b], [0] * len(ents[b])))
                for g, h in zip(*np.where(iou > GROUP_IOU)):
                    parent[find(idx[(a, int(g))])] = find(idx[(b, int(h))])
        groups = defaultdict(list)
        for it in items:
            groups[find(idx[it])].append(it)
        for members in groups.values():
            masks = [mask_util.decode(ents[a][g]) for a, g in members]
            soft_full = np.mean(np.stack(masks), axis=0).astype(np.float32)
            n_groups += 1; n_multi += int(len(members) > 1)
            for m in masks:
                if int(m.sum()) < 10:
                    continue
                x0, y0, x1, y1 = square_bounds(m)
                imgs.append(cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR))
                hard.append(cv2.resize(m[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_NEAREST))
                soft.append(cv2.resize(soft_full[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR).astype(np.float16))
                files.append(fn)
    print(f"consensus crops: {len(imgs)} from {n_groups} filament groups ({n_multi} drawn by >=2 annotators)", flush=True)
    return np.stack(imgs), np.stack(hard), np.stack(soft), np.array(files)


all_imgs, all_hard, all_soft, all_files = build_consensus_crops()
print("mean soft target value inside hard mask:", float(all_soft[all_hard > 0].astype(np.float32).mean()), flush=True)

rows = []
acc, acc_n = {}, {}
for K in FOLDS_TO_RUN:
    tr_files, va_files = unique_files[folds[K][0]], unique_files[folds[K][1]]
    sel_old = np.isin(crop_files, tr_files)
    sel_new = np.isin(all_files, tr_files)
    refiners = {}
    for seed, (name, (x, y)) in enumerate([
            ('base_a', (crop_imgs[sel_old], crop_msks[sel_old])),
            ('base_b', (crop_imgs[sel_old], crop_msks[sel_old])),
            ('allhard', (all_imgs[sel_new], all_hard[sel_new])),
            ('soft', (all_imgs[sel_new], all_soft[sel_new]))]):
        torch.manual_seed(1000 * K + seed); np.random.seed(1000 * K + seed); random.seed(1000 * K + seed)
        t0 = time.time()
        refiners[name] = train_refiner(x, y, REF_EPOCHS)
        print(f"fold {K}: refiner {name} trained on {len(x)} crops in {time.time()-t0:.0f}s", flush=True)
    det = load_det(NEW_CKPTS[K])
    for v in refiners:
        for rt in REF_THRS:
            for cfg in CFGS:
                acc[(K, v, rt, cfg)] = np.zeros(4)
    acc_n[K] = len(va_files)
    for fn in tqdm(va_files, desc=f"fold {K}"):
        ents = entries_by_file[fn]
        for v, ref in refiners.items():
            obs = Obs(detect(det, ref, fn, VIEWS))
            for rt in REF_THRS:
                for cfg in CFGS:
                    acc[(K, v, rt, cfg)] += pq_stats(predict(obs, rt, cfg[0], ('tta4', cfg[1])), ents)
            obs.cache.clear()
    if 'soft' in refiners:
        torch.save(refiners['soft'].state_dict(), OUT / f'refiner_soft_fold{K}.pth')
    del det, refiners; gc.collect(); torch.cuda.empty_cache()

for (K, v, rt, cfg), st_ in acc.items():
    rows.append(dict(fold=K, variant=v, ref_thr=rt, score_thr=cfg[0], agree=cfg[1], PQ=pq_of(st_),
                     mean_iou_tp=float(st_[0] / max(st_[1], 1)), TP=int(st_[1]), FP=int(st_[2]), FN=int(st_[3])))
df = pd.DataFrame(rows)
tot = {}
for (K, v, rt, cfg), st_ in acc.items():
    tot[(v, rt, cfg)] = tot.get((v, rt, cfg), 0) + st_
pooled = pd.DataFrame([dict(fold='pooled', variant=v, ref_thr=rt, score_thr=cfg[0], agree=cfg[1], PQ=pq_of(s),
                            mean_iou_tp=float(s[0] / max(s[1], 1)), TP=int(s[1]), FP=int(s[2]), FN=int(s[3]))
                       for (v, rt, cfg), s in tot.items()])
df = pd.concat([df, pooled], ignore_index=True)
df.to_csv(OUT / 'consensus_refiner.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 500)
pv = df[df.fold == 'pooled'].pivot_table(index=['score_thr', 'ref_thr'], columns='variant', values='PQ').round(4)
pv_iou = df[df.fold == 'pooled'].pivot_table(index=['score_thr', 'ref_thr'], columns='variant', values='mean_iou_tp').round(4)
per_fold = df[(df.fold != 'pooled') & (df.ref_thr == 0.5) & (df.score_thr == 0.7)].pivot_table(index='fold', columns='variant', values='PQ').round(4)
summary = (f"folds {FOLDS_TO_RUN}\nPOOLED PQ (rows: score_thr, ref_thr)\n{pv.to_string()}\n\n"
           f"POOLED mean IoU of true positives\n{pv_iou.to_string()}\n\nper fold at score 0.7, ref_thr 0.5\n{per_fold.to_string()}")
(OUT / 'summary.txt').write_text(summary, encoding='utf-8')
print(summary)
