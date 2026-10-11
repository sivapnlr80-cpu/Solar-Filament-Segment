"""Panoptic Quality as scored by the organizers: per annotator-image entry, against ALL predictions of the image.

PQ = sum(IoU of TPs) / (|TP| + 0.5 |FP| + 0.5 |FN|); a prediction/ground-truth pair is a TP if IoU > 0.5.
Statistics are accumulated over entries (ground-truth sets) before dividing.
"""
import numpy as np
import pycocotools.mask as mask_util


def pq_stats(pred_rles, entries):
    """pred_rles: list of RLEs of one image; entries: list of ground-truth RLE lists (one per annotator).

    Returns np.array([sum_iou_tp, TP, FP, FN]) accumulated over the entries.
    """
    s = tp = fp = fn = 0.0
    n_pred = len(pred_rles)
    for gt in entries:
        if n_pred == 0:
            fn += len(gt)
            continue
        iou = np.asarray(mask_util.iou(gt, pred_rles, [0] * n_pred))
        hit = iou > 0.5
        s += float(iou[hit].sum())
        tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return np.array([s, tp, fp, fn], dtype=np.float64)


def pq_of(stats):
    denom = stats[1] + 0.5 * stats[2] + 0.5 * stats[3]
    return float(stats[0] / denom) if denom > 0 else 0.0
