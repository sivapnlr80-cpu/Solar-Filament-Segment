#!/usr/bin/env python3
"""
Train U-Net mask refiner for solar filament segmentation.
For each GT filament annotation: crop native image around it → 256×256 → train
CropUNet (ResNet18 encoder-decoder) with Dice+BCE loss.
Output: /kaggle/working/refiner.pth
"""

import os, sys, random, time, warnings
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pycocotools"])
    from pycocotools.coco import COCO

from torchvision.models import resnet18, ResNet18_Weights

warnings.filterwarnings("ignore")

SEED = 2026
CROP_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
CONTEXT = 1.8
MIN_CROP = 96

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def find_dir(name):
    for sr in [Path("/kaggle/input"), Path("data")]:
        if sr.exists():
            for m in sr.rglob(name):
                if m.is_dir() and any(m.iterdir()): return m
    return None


def find_file(name):
    for sr in [Path("/kaggle/input"), Path("data")]:
        if sr.exists():
            for m in sr.rglob(name):
                if m.is_file(): return m
    return None


TRAIN_DIR = find_dir("train_images")
ANN_PATH = find_file("MAGFiLO_1.0_Annotations_kaggle2026_train.json")
print(f"Train images: {TRAIN_DIR}")
print(f"Annotations:  {ANN_PATH}")
assert TRAIN_DIR and ANN_PATH

coco = COCO(str(ANN_PATH))

records = []
for img_id in coco.imgs:
    info = coco.imgs[img_id]
    anns = coco.imgToAnns.get(img_id, [])
    records.append({
        'id': img_id, 'file_name': info['file_name'],
        'n_filaments': len(anns),
        'total_area': sum(float(a.get('area', 0)) for a in anns)
    })
df = pd.DataFrame(records)
manifest = (df.sort_values(['file_name', 'n_filaments', 'total_area', 'id'])
              .drop_duplicates('file_name', keep='last')
              .reset_index(drop=True))
print(f"Images: {len(manifest)}, Total annotations: {manifest['n_filaments'].sum()}")


def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, min(minimum, w), min(minimum, h)
    cx = (xs.min() + xs.max() + 1) / 2
    cy = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(h, w))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return int(x0), int(y0), int(x1), int(y1)


print("Building crop cache...")
crop_images_list, crop_masks_list = [], []
for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
    img_path = TRAIN_DIR / row['file_name']
    if not img_path.exists():
        continue
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    for ann in coco.imgToAnns.get(row['id'], []):
        mask = coco.annToMask(ann).astype(np.uint8)
        if mask.sum() < 10:
            continue
        x0, y0, x1, y1 = square_bounds(mask)
        crop_images_list.append(
            cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR))
        crop_masks_list.append(
            cv2.resize(mask[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST))

crop_images = np.stack(crop_images_list)
crop_masks = np.stack(crop_masks_list)
del crop_images_list, crop_masks_list
print(f"Crops: {len(crop_images)}")


class CropDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images, self.masks, self.augment = images, masks, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].copy()).float() / 255.0
        mask = torch.from_numpy(self.masks[idx].copy()).float().unsqueeze(0)
        if self.augment:
            k = random.randrange(4)
            image = torch.rot90(image, k, (0, 1))
            mask = torch.rot90(mask, k, (1, 2))
            if random.random() < 0.5:
                image = torch.flip(image, (1,))
                mask = torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.20)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.90, 1.10), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2
        self.layer3, self.layer4 = enc.layer3, enc.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d = self.dec4(torch.cat([self.up4(s4), s3], 1))
        d = self.dec3(torch.cat([self.up3(d), s2], 1))
        d = self.dec2(torch.cat([self.up2(d), s1], 1))
        d = self.dec1(torch.cat([self.up1(d), s0], 1))
        return self.head(self.up0(d))


def soft_dice_loss(logits, targets):
    p = logits.sigmoid()
    inter = (p * targets).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


dataset = CropDataset(crop_images, crop_masks, augment=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True)

model = CropUNet(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda')

print(f"\nTraining CropUNet: {EPOCHS} epochs, {len(dataset)} crops, bs={BATCH_SIZE}")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = n = 0
    for images, masks in tqdm(loader, desc=f"Epoch {epoch:02d}/{EPOCHS}", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
            dice = soft_dice_loss(logits, masks)
            loss = 0.5 * bce + 0.5 * dice
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(images)
        n += len(images)
    scheduler.step()
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss {loss_sum/n:.4f} | "
          f"LR {optimizer.param_groups[0]['lr']:.6f}")

elapsed = time.time() - t0
print(f"\nTraining done: {elapsed:.1f}s ({elapsed/60:.1f} min)")

out_path = "/kaggle/working/refiner.pth"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(model.state_dict(), out_path)
sz = Path(out_path).stat().st_size / 1e6
print(f"Saved: {out_path} ({sz:.1f} MB)")
