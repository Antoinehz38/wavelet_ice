from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.cwt_scheduler import TimeWindow


def compute_iou(box_a, box_b):
    """Compute Intersection over Union (IoU) between two rectangles.

    Expected format: (x, y, w, h) or (x, y, w, h, label).
    """
    xa1, ya1 = box_a[0], box_a[1]
    xa2, ya2 = box_a[0] + box_a[2], box_a[1] + box_a[3]

    xb1, yb1 = box_b[0], box_b[1]
    xb2, yb2 = box_b[0] + box_b[2], box_b[1] + box_b[3]

    x_inter1 = max(xa1, xb1)
    y_inter1 = max(ya1, yb1)
    x_inter2 = min(xa2, xb2)
    y_inter2 = min(ya2, yb2)

    inter_w = max(0, x_inter2 - x_inter1)
    inter_h = max(0, y_inter2 - y_inter1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]

    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """Match predictions to ground-truth boxes at the given IoU threshold.

    Returns (TP, FP, FN).
    """
    preds = list(pred_boxes)
    gts = list(gt_boxes)

    tp = 0
    fp = 0

    for p_box in preds:
        best_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(gts):
            iou = compute_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            tp += 1
            gts.pop(best_gt_idx)
        else:
            fp += 1

    fn = len(gts)

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    """Compute Precision, Recall and F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def evaluate_coco_style(pred_boxes, gt_boxes):
    """Full evaluation over IoU range 0.50 -> 0.95 (10 steps)."""
    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    results = {}
    f1_scores = []

    print(f"\n DETAILED EVALUATION ({len(pred_boxes)} Preds vs {len(gt_boxes)} GT)")
    print(f"{'IoU Thresh':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)

    for thresh in iou_thresholds:
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, thresh)
        prec, rec, f1 = calculate_metrics(tp, fp, fn)

        results[thresh] = {"p": prec, "r": rec, "f1": f1}
        f1_scores.append(f1)

        print(f"{thresh:.2f}{' ':<8} | {prec:.2f}{' ':<6} | {rec:.2f}{' ':<6} | {f1:.2f}")

    avg_f1 = np.mean(f1_scores)
    print("-" * 50)
    print(f"FINAL SCORE (mF1 @ .50:.95) : {avg_f1:.4f}\n")

    return avg_f1, results


# ---------------------------------------------------------------------------
# Merging boxes from multiple analysis windows
# ---------------------------------------------------------------------------


def _pixel_to_global(
    boxes: list,
    win_start: int,
    win_length: int,
    img_w: int,
) -> List[Tuple[float, float, float, float]]:
    """Convert pixel-space boxes to global sample coordinates (x-axis only).

    The y-axis (frequency) is kept in pixel space since it is consistent
    across all windows.

    Parameters
    ----------
    boxes : list of (x, y, w, h) or (x, y, w, h, label)
        Detected boxes in pixel coordinates of the compressed image.
    win_start : int
        First sample index of the time window.
    win_length : int
        Length of the time window in samples.
    img_w : int
        Width (in pixels) of the compressed image.

    Returns
    -------
    List of (x_global, y, w_global, h [, label])
        Boxes expressed in global sample coordinates for x, pixel coords for y.
    """
    scale_x = win_length / img_w  # samples per pixel
    global_boxes = []
    for box in boxes:
        x, y, w, h = box[:4]
        x_global = win_start + x * scale_x
        w_global = w * scale_x
        entry = (x_global, y, w_global, h)
        if len(box) > 4:
            entry = (*entry, box[4])
        global_boxes.append(entry)
    return global_boxes


def _boxes_overlap(a, b, iou_threshold: float = 0.0) -> bool:
    """Check if two boxes overlap enough to be merged.

    When *iou_threshold* is 0 any intersection is enough.
    Boxes format: (x, y, w, h [, …]).
    """
    if iou_threshold <= 0.0:
        # Simple overlap check
        xa1, ya1, xa2, ya2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        xb1, yb1, xb2, yb2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        return xa1 < xb2 and xa2 > xb1 and ya1 < yb2 and ya2 > yb1
    else:
        return compute_iou(a, b) >= iou_threshold


def _union_box(a, b) -> Tuple[float, float, float, float]:
    """Return the bounding box that encloses both *a* and *b*."""
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def merge_boxes(
    window_boxes: List[Tuple[list, "TimeWindow", int]],
    iou_threshold: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """Merge detection boxes coming from multiple analysis windows.

    Each element of *window_boxes* is a tuple::

        (boxes, time_window, img_w)

    where
    * **boxes** – list of ``(x, y, w, h)`` in compressed-image pixel coords,
    * **time_window** – a :class:`TimeWindow` (needs ``.start`` and ``.length``),
    * **img_w** – width in pixels of the compressed spectrogram for that window.

    The function:

    1. Converts every box to **global sample coordinates** (x-axis) so that
       boxes from different windows share the same reference frame.
    2. Iteratively merges boxes that overlap (controlled by *iou_threshold*).

    Parameters
    ----------
    window_boxes : list of (boxes, TimeWindow, img_w)
        Detection results per window.
    iou_threshold : float, optional
        Minimum IoU to consider two boxes as overlapping.
        ``0.0`` (default) merges any pair that has **any** intersection.

    Returns
    -------
    List of (x, y, w, h) in global sample coordinates (x) and pixel coords (y).
    """
    # 1. Collect all boxes in a single global coordinate system
    all_global: List[Tuple[float, float, float, float]] = []
    for boxes, tw, img_w in window_boxes:
        converted = _pixel_to_global(boxes, tw.start, tw.length, img_w)
        all_global.extend([(b[0], b[1], b[2], b[3]) for b in converted])

    if not all_global:
        return []

    # 2. Greedy iterative merge until stable
    merged = list(all_global)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _boxes_overlap(current, merged[j], iou_threshold):
                    current = _union_box(current, merged[j])
                    used[j] = True
                    changed = True
            new_merged.append(current)
            used[i] = True

        merged = new_merged

    print(f"[merge_boxes] {len(all_global)} boxes → {len(merged)} after merge")
    return merged