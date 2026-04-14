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
    img_h: int,
) -> List[Tuple[float, float, float, float]]:
    """Convert pixel-space boxes to global coordinates.

    * **x-axis** (time) → global sample index.
    * **y-axis** (frequency) → normalised to [0, 1] so that boxes from
      different windows (potentially with different image sizes) are
      comparable.

    Parameters
    ----------
    boxes : list of (x, y, w, h [, label])
        Detected boxes in compressed-image pixel coordinates.
    win_start : int
        First sample index of the time window.
    win_length : int
        Length of the time window in samples.
    img_w : int
        Width (pixels) of the compressed spectrogram.
    img_h : int
        Height (pixels) of the compressed spectrogram.

    Returns
    -------
    List of (x_global, y_norm, w_global, h_norm)
    """
    scale_x = win_length / img_w   # samples per pixel (time)
    scale_y = 1.0 / img_h          # normalised per pixel (freq)
    global_boxes: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        x, y, w, h = box[:4]
        x_global = win_start + x * scale_x
        w_global = w * scale_x
        y_norm = y * scale_y
        h_norm = h * scale_y
        global_boxes.append((x_global, y_norm, w_global, h_norm))
    return global_boxes


def _freq_overlap(box_a, box_b, tolerance: float) -> bool:
    """Check whether two boxes cover a similar frequency band.

    Both boxes are expressed in normalised frequency coordinates
    ``(x, y, w, h)`` where y and h are in [0, 1].

    Two bands match when both edges are within *tolerance* of each other.
    """
    ya_min, ya_max = box_a[1], box_a[1] + box_a[3]
    yb_min, yb_max = box_b[1], box_b[1] + box_b[3]
    return abs(ya_min - yb_min) <= tolerance and abs(ya_max - yb_max) <= tolerance


def merge_boxes(
    window_boxes: List[Tuple[list, "TimeWindow", int, int]],
    freq_tolerance: float = 0.05,
) -> List[Tuple[float, float, float, float]]:
    """Merge detection boxes coming from multiple analysis windows.

    ``build_transition_windows`` produces windows centred on transition
    points (moments where a signal starts or ends).  Each window yields
    detected boxes in compressed pixel space.  The same physical signal
    will therefore appear as a **start box** in one window and an **end
    box** in a later window.

    This function:

    1. Converts every box from **compressed pixel coordinates** to
       **global sample coordinates** (time) and **normalised frequency**
       so that boxes from different windows share a common reference.
    2. **Matches** boxes from different windows that share the same
       frequency band (within *freq_tolerance*), treating them as the
       same physical signal observed at its beginning and its end.
    3. For each matched group the final box spans from the **earliest
       detection** to the **latest detection** (i.e. the signal is
       assumed present over the whole interval even though intermediate
       samples were not computed).

    Parameters
    ----------
    window_boxes : list of (boxes, TimeWindow, img_w, img_h)
        Detection results per window.  *boxes* is a list of
        ``(x, y, w, h)`` in compressed-image pixel coords.
        *img_w* and *img_h* are the dimensions of the compressed
        spectrogram used for detection.
    freq_tolerance : float, optional
        Maximum allowed difference (in normalised frequency, 0-1) between
        the upper / lower edges of two boxes for them to be considered
        the same signal.  Default ``0.05`` (5 % of the frequency axis).

    Returns
    -------
    List of (x, y, w, h) in **global sample coordinates** (time) and
    **normalised frequency** [0, 1] (freq).
    """
    if not window_boxes:
        return []

    # ------------------------------------------------------------------
    # 1.  Convert every box to global (sample, norm-freq) coordinates
    #     and tag it with its source window index.
    # ------------------------------------------------------------------
    tagged_boxes: List[Tuple[Tuple[float, float, float, float], int]] = []
    for win_idx, (boxes, tw, img_w, img_h) in enumerate(window_boxes):
        if not boxes or img_w == 0:
            continue
        converted = _pixel_to_global(boxes, tw.start, tw.length, img_w, img_h)
        for gb in converted:
            tagged_boxes.append((gb, win_idx))

    if not tagged_boxes:
        return []

    # ------------------------------------------------------------------
    # 2.  Match boxes across windows by frequency band similarity.
    #     We build groups: each group = one physical signal seen in ≥1
    #     windows.  Within a group we keep the union frequency band and
    #     extend the time span from earliest to latest.
    # ------------------------------------------------------------------
    groups: List[List[Tuple[float, float, float, float]]] = []

    used = [False] * len(tagged_boxes)
    for i in range(len(tagged_boxes)):
        if used[i]:
            continue
        box_i, win_i = tagged_boxes[i]
        group = [box_i]
        used[i] = True

        for j in range(i + 1, len(tagged_boxes)):
            if used[j]:
                continue
            box_j, win_j = tagged_boxes[j]
            # Only match boxes from *different* windows
            if win_j == win_i:
                continue
            if _freq_overlap(box_i, box_j, freq_tolerance):
                group.append(box_j)
                used[j] = True

        groups.append(group)

    # ------------------------------------------------------------------
    # 3.  Build the final merged boxes.
    #     For each group the time span covers from the earliest x to the
    #     latest x+w.  The frequency band is the union of all members.
    # ------------------------------------------------------------------
    merged: List[Tuple[float, float, float, float]] = []
    for group in groups:
        x_min = min(b[0] for b in group)
        x_max = max(b[0] + b[2] for b in group)
        y_min = min(b[1] for b in group)
        y_max = max(b[1] + b[3] for b in group)
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    print(
        f"[merge_boxes] {len(tagged_boxes)} boxes from "
        f"{len(window_boxes)} windows → {len(merged)} merged signals"
    )
    return merged