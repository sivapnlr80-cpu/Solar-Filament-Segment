"""Network definitions: Mask R-CNN detector and the crop-refinement U-Net."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(size: int = 1024, num_classes: int = 2, pretrained: bool = False):
    """torchvision Mask R-CNN (ResNet-50-FPN v2) with a 2-class (background / filament) head.

    ``pretrained=True`` starts from COCO weights (training); inference loads our own checkpoint.
    """
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, min_size=size, max_size=size)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, 256, num_classes)
    return model


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    """U-Net with a ResNet-18 encoder; input: 3-channel 256x256 crop in [0, 1], output: mask logits."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2
        self.layer3, self.layer4 = enc.layer3, enc.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

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


def load_maskrcnn(path, device, size: int = 1024):
    model = build_maskrcnn(size).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    return model.eval()


def load_refiner(path, device):
    model = CropUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()
