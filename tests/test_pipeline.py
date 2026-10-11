"""Unit tests on synthetic data (no weights or competition data needed): python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filament import infer, metrics, models  # noqa: E402


def blob(shape, y0, y1, x0, x1):
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def enc(m):
    return mask_util.encode(np.asfortranarray(m))


class MetricTests(unittest.TestCase):
    shape = (300, 300)

    def test_perfect_prediction(self):
        gt = [enc(blob(self.shape, 10, 40, 10, 100))]
        stats = metrics.pq_stats(gt, [gt])
        self.assertAlmostEqual(metrics.pq_of(stats), 1.0)

    def test_no_prediction_is_all_false_negatives(self):
        gt = [enc(blob(self.shape, 10, 40, 10, 100))]
        stats = metrics.pq_stats([], [gt, gt])
        self.assertEqual(stats[3], 2)
        self.assertEqual(metrics.pq_of(stats), 0.0)

    def test_unmatched_prediction_counts_as_false_positive_per_entry(self):
        gt = [enc(blob(self.shape, 10, 40, 10, 100))]
        pred = gt + [enc(blob(self.shape, 200, 230, 200, 260))]
        stats = metrics.pq_stats(pred, [gt, gt])       # two annotator entries, one spurious prediction
        self.assertEqual((stats[1], stats[2], stats[3]), (2, 2, 0))
        self.assertAlmostEqual(metrics.pq_of(stats), 2 / (2 + 0.5 * 2))

    def test_iou_threshold_is_strict(self):
        gt = [enc(blob(self.shape, 0, 10, 0, 100))]
        half = [enc(blob(self.shape, 0, 10, 0, 50))]    # IoU exactly 0.5 -> not a match
        stats = metrics.pq_stats(half, [gt])
        self.assertEqual(stats[1], 0)


class PostProcessingTests(unittest.TestCase):
    shape = (256, 256)

    def dets(self, views, mask=None):
        mask = blob(self.shape, 60, 100, 40, 200) if mask is None else mask
        return [(0.9, enc(mask), v) for v in views]

    def test_square_bounds_inside_image_and_min_size(self):
        m = blob(self.shape, 100, 110, 100, 120)
        x0, y0, x1, y1 = infer.square_bounds(m, 1.8, 96)
        self.assertEqual((x1 - x0), (y1 - y0))
        self.assertGreaterEqual(x1 - x0, 96)
        self.assertTrue(0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256)

    def test_three_of_four_views_survive_two_do_not(self):
        cfg = dict(infer.CFG, min_area=50)
        self.assertEqual(len(infer.consolidate(self.dets([0, 1, 2]), self.shape, cfg)), 1)
        self.assertEqual(len(infer.consolidate(self.dets([0, 3]), self.shape, cfg)), 0)

    def test_same_view_duplicates_do_not_count_as_agreement(self):
        cfg = dict(infer.CFG, min_area=50)
        self.assertEqual(len(infer.consolidate(self.dets([0, 0, 0]), self.shape, cfg)), 0)

    def test_painting_removes_overlap_and_prefers_higher_score(self):
        a = blob(self.shape, 50, 100, 50, 150)
        b = blob(self.shape, 70, 120, 50, 150)
        kept = infer.paint_panoptic([(0.6, b), (0.9, a)], 50, self.shape)
        self.assertEqual(len(kept), 2)
        self.assertEqual(int((kept[0][1] & kept[1][1]).sum()), 0)
        self.assertEqual(kept[0][0], 0.9)
        self.assertEqual(int(kept[0][1].sum()), int(a.sum()))

    def test_small_masks_are_dropped(self):
        tiny = blob(self.shape, 10, 12, 10, 12)
        self.assertEqual(infer.paint_panoptic([(0.9, tiny)], 200, self.shape), [])

    def test_rle_counts_roundtrip(self):
        m = blob((2048, 2048), 100, 140, 300, 900)
        counts = infer.rle_counts(m)
        back = mask_util.decode({"size": [2048, 2048], "counts": counts.encode()})
        self.assertTrue(np.array_equal(back, m))


class FakeDetector:
    """Returns the same filament in every TTA view (the soft mask is transformed like the input)."""

    def __init__(self, size):
        base = torch.zeros(size, size)
        base[300:360, 200:700] = 1.0
        self.base = base

    def __call__(self, views):
        outs = []
        for (_, fwd, _) in infer.TTA_VIEWS:
            soft = fwd(self.base.unsqueeze(0)).contiguous()              # 1 x H x W
            outs.append({"scores": torch.tensor([0.95, 0.2]),
                         "masks": torch.stack([soft, torch.zeros_like(soft)])})   # N x 1 x H x W
        return outs


class FakeRefiner:
    def __call__(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 8.0)   # sigmoid ~ 1 everywhere


class EndToEndTests(unittest.TestCase):
    def test_predict_image_with_stub_networks(self):
        image = (np.random.rand(2048, 2048) * 255).astype(np.uint8)
        kept = infer.predict_image(image, FakeDetector(infer.CFG["image_size"]), FakeRefiner(), torch.device("cpu"))
        self.assertEqual(len(kept), 1)                    # the low-score detection is filtered, 4 views agree
        score, mask = kept[0]
        self.assertAlmostEqual(score, 0.95, places=5)
        self.assertEqual(mask.shape, (2048, 2048))
        self.assertGreater(int(mask.sum()), 200)


class ModelTests(unittest.TestCase):
    def test_refiner_output_shape(self):
        net = models.CropUNet().eval()
        with torch.no_grad():
            out = net(torch.rand(2, 3, 256, 256))
        self.assertEqual(tuple(out.shape), (2, 1, 256, 256))

    def test_detector_head_has_two_classes(self):
        det = models.build_maskrcnn(size=512)
        self.assertEqual(det.roi_heads.box_predictor.cls_score.out_features, 2)


if __name__ == "__main__":
    unittest.main()
