from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Dict, Any


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

def is_begin(box: tuple, w: TimeWindow, img_w: int, img_h: int, margin: int = 0) -> bool:
    x, y, box_w, box_h = box
    if x > margin:
        return True
    if w.start == 0:
        return True
        
    return False

def is_end(box: tuple, w: TimeWindow, img_w: int, total_samples: int = 100_000_000, margin: int = 0) -> bool:
    x, y, box_w, box_h = box
    if (x + box_w) < (img_w - margin):
        return True
    if w.end >= total_samples:
        return True
    return False

def pixel_to_sample(x: int, w_start: int, w_length: int, img_w: int) -> int:
    """Convertit une coordonnée X en numéro de sample."""
    samples_per_pixel = w_length / img_w
    return w_start + int(x * samples_per_pixel)

def pixel_to_freq(y: int, h: int, img_h: int) -> Tuple[float, float]:
    """
    Convertit Y et la hauteur en fréquences normalisées [-0.5, 0.5].
    Hypothèse : y=0 est le haut de l'image (+0.5), y=img_h est le bas (-0.5).
    """
    # y est le bord haut de la box (fréquence haute)
    # y + h est le bord bas de la box (fréquence basse)
    freq_upper = 0.5 - (y / img_h)
    freq_lower = 0.5 - ((y + h) / img_h)
    return freq_lower, freq_upper

def generate_sigmf_annotations(
    unitary_boxes: List[Tuple], 
    matched_boxes: List[Tuple], 
    default_scale: float = 1.0
) -> Dict[str, Any]:
    
    annotations = []

    for u_data in unitary_boxes:
        box, w, img_w, img_h = u_data
        desc = "prediction_unitary"
        x, y, box_w, box_h = box
        
        start_sample = pixel_to_sample(x, w.start, w.length, img_w)
        end_sample = pixel_to_sample(x + box_w, w.start, w.length, img_w)
        
        f_lower, f_upper = pixel_to_freq(y, box_h, img_h)
        
        annotations.append({
            "core:sample_start": start_sample,
            "core:sample_count": end_sample - start_sample,
            "core:description": desc,
            "core:freq_lower_edge": f_lower,
            "core:freq_upper_edge": f_upper,
            "west:scale": default_scale
        })

    for match in matched_boxes:
        b_data, e_data = match
        
        box_b, w_b, img_w_b, img_h_b, = b_data
        x_b, y_b, w_b_box, h_b = box_b
        
        box_e, w_e, img_w_e, img_h_e,= e_data
        x_e, y_e, w_e_box, h_e = box_e
        
        start_sample = pixel_to_sample(x_b, w_b.start, w_b.length, img_w_b)
        end_sample = pixel_to_sample(x_e + w_e_box, w_e.start, w_e.length, img_w_e)
        
        f_lower, f_upper = pixel_to_freq(y_b, h_b, img_h_b)
        
        annotations.append({
            "core:sample_start": start_sample,
            "core:sample_count": end_sample - start_sample,
            "core:description": "prediction_merged",
            "core:freq_lower_edge": f_lower,
            "core:freq_upper_edge": f_upper,
            "west:scale": default_scale
        })

    annotations = sorted(annotations, key=lambda a: a["core:sample_start"])

    return {
        "global": {
            "core:sample_rate": 1.0,
            "core:datatype": "cf32_le"
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:sample_count": 100000000,
                "core:frequency": 0.0
            }
        ],
        "annotations": annotations
    }


def merge_boxes(total_boxes: List[Tuple[List[tuple], TimeWindow, int, int]], margin: int = 5, freq_margin: int = 10) -> dict:
    boxes_begin: List[Tuple[tuple, TimeWindow, int, int]] = []
    boxes_end: List[Tuple[tuple, TimeWindow, int, int]] = []
    boxes_unitary: List[Tuple[tuple, TimeWindow, int, int]] = []
    boxes_middle: List[Tuple[tuple, TimeWindow, int, int]] = []
    
    # 1. Tri des boîtes
    for boxes, w, img_w, img_h in total_boxes:
        for box in boxes:
            is_b = is_begin(box, w, img_w, img_h, margin)
            # Attention: dans ton brouillon tu passais img_h au lieu de total_samples à is_end
            is_e = is_end(box, w, img_w, total_samples=100_000_000, margin=margin)
            
            if is_b and is_e:
                boxes_unitary.append((box, w, img_w, img_h))
            elif is_b and not is_e:
                boxes_begin.append((box, w, img_w, img_h))
            elif not is_b and is_e:
                boxes_end.append((box, w, img_w, img_h))
            else:
                # Signal qui traverse toute l'image de part en part
                boxes_middle.append((box, w, img_w, img_h)) 

    matched_boxes: List[Tuple[Tuple, Tuple]] = []
    
    available_ends = boxes_end.copy()

    # 2. Matching des boîtes
    for b_data in boxes_begin:
        box_b, w_b, img_w_b, img_h_b = b_data
        x_b, y_b, w_b_box, h_b = box_b
        
        best_match_idx = -1
        
        for i, e_data in enumerate(available_ends):
            box_e, w_e, img_w_e, img_h_e = e_data
            x_e, y_e, w_e_box, h_e = box_e
            
            # Condition 1 : La fenêtre de fin doit être après la fenêtre de début
            if w_e.start >= w_b.end:
                
                # Condition 2 : Les fréquences doivent correspondre (Y et Hauteur)
                y_match = abs(y_b - y_e) <= freq_margin
                h_match = abs(h_b - h_e) <= freq_margin
                
                if y_match and h_match:
                    # Match trouvé !
                    matched_boxes.append((b_data, e_data))
                    best_match_idx = i
                    break # On a trouvé la suite, on arrête de chercher pour ce début
        
        # Si on a trouvé un match, on l'enlève de la liste des fins disponibles 
        # pour éviter qu'une fin soit rattachée à plusieurs débuts
        if best_match_idx != -1:
            available_ends.pop(best_match_idx)

    # Je te retourne tout dans un dictionnaire pour que ce soit facile à explorer
    dict_final={
        "unitary": boxes_unitary,
        "matched": matched_boxes,
        "unmatched_begins": len(boxes_begin) - len(matched_boxes),
        "unmatched_ends": len(available_ends),
        "middles": boxes_middle
    }
    return generate_sigmf_annotations(unitary_boxes=boxes_unitary, matched_boxes=matched_boxes, default_scale=1.0)



if __name__ == '__main__':
    total_boxes = [([(np.int64(0), np.int64(776), np.int64(1900), np.int64(94)), 
                     (np.int64(1500), np.int64(1099), np.int64(1400), np.int64(108)), 
                     (np.int64(0), np.int64(0), np.int64(4000), np.int64(98)), 
                     (np.int64(0), np.int64(1326), np.int64(4000), np.int64(117)), 
                     (np.int64(1800), 41, np.int64(2200), 118)], 
                     TimeWindow(start=0, end=2000000, active_annotations=[0, 1, 2, 3, 4], descriptions=['PSK2', 'AM_SSB', 'OOK', 'FM', 'PSK2']), 4000, 1500), 
                     ([(np.int64(0), np.int64(0), np.int64(4000), np.int64(157)), (np.int64(0), np.int64(775), np.int64(4000), np.int64(96)), (np.int64(0), np.int64(1326), np.int64(4000), np.int64(117))], TimeWindow(start=47827672, end=49827672, active_annotations=[0, 1, 2, 3, 4], descriptions=['PSK2', 'AM_SSB', 'OOK', 'FM', 'PSK2']), 4000, 1500), ([(np.int64(0), 45, np.int64(2000), np.int64(112)), (np.int64(0), np.int64(0), np.int64(4000), np.int64(98)), (np.int64(0), np.int64(776), np.int64(4000), np.int64(94)), (np.int64(0), np.int64(1326), np.int64(4000), np.int64(117))], TimeWindow(start=54164375, end=56164375, active_annotations=[0, 2, 3, 4], descriptions=['PSK2', 'OOK', 'FM', 'PSK2']), 4000, 1500), ([(np.int64(0), np.int64(0), np.int64(2000), np.int64(98)), (np.int64(0), np.int64(776), np.int64(4000), np.int64(95)), (np.int64(0), np.int64(1326), np.int64(4000), np.int64(117))], TimeWindow(start=62643878, end=64643878, active_annotations=[0, 2, 3], descriptions=['PSK2', 'OOK', 'FM']), 4000, 1500), ([(np.int64(0), np.int64(1327), np.int64(1999), np.int64(115)), (np.int64(0), np.int64(776), np.int64(4000), np.int64(95))], TimeWindow(start=78808910, end=80808910, active_annotations=[0, 2], descriptions=['PSK2', 'OOK']), 4000, 1500), ([(np.int64(0), np.int64(776), np.int64(1998), np.int64(94))], TimeWindow(start=82473102, end=84473102, active_annotations=[0], descriptions=['PSK2']), 4000, 1500)]

    print(merge_boxes(total_boxes))

