"""Solar Filament Segmentation Challenge 2026 - solution package."""
from .infer import CFG, predict_image, run_directory  # noqa: F401
from .models import build_maskrcnn, CropUNet, load_maskrcnn, load_refiner  # noqa: F401
