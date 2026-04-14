import numpy as np


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