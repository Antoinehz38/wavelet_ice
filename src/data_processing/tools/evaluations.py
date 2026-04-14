import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(boxA, boxB):
    """
    Calcule l'Intersection over Union (IoU) entre deux rectangles.
    Format attendu : (x, y, w, h)
    """
    # Conversion (x, y, w, h) -> (x1, y1, x2, y2)
    # x1,y1 = Haut-Gauche / x2,y2 = Bas-Droite
    xA_1, yA_1 = boxA[0], boxA[1]
    xA_2, yA_2 = boxA[0] + boxA[2], boxA[1] + boxA[3]

    xB_1, yB_1 = boxB[0], boxB[1]
    xB_2, yB_2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    # Calcul des coordonnees de l'intersection
    x_inter_1 = max(xA_1, xB_1)
    y_inter_1 = max(yA_1, yB_1)
    x_inter_2 = min(xA_2, xB_2)
    y_inter_2 = min(yA_2, yB_2)

    # Aire de l'intersection (on clamp a 0 si pas de superposition)
    inter_w = max(0, x_inter_2 - x_inter_1)
    inter_h = max(0, y_inter_2 - y_inter_1)
    inter_area = inter_w * inter_h

    # Aires des boites individuelles
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    # Aire de l'Union = Aire A + Aire B - Aire Intersection
    union_area = boxA_area + boxB_area - inter_area

    # Protection division par zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def _match_boxes_optimal(pred_boxes, gt_boxes, iou_threshold):
    """
    Calcule un matching biparti global optimal en maximisant la somme des IoU.
    Les paires sous le seuil sont ignorees et les boites peuvent rester non appariees.
    Retourne une liste de tuples (pred_idx, gt_idx, iou).
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0 or num_gts == 0:
        return []

    iou_matrix = np.zeros((num_preds, num_gts), dtype=float)
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                iou_matrix[pred_idx, gt_idx] = iou

    # Padding carre avec score nul pour permettre les non-assignations.
    size = max(num_preds, num_gts)
    score_matrix = np.zeros((size, size), dtype=float)
    score_matrix[:num_preds, :num_gts] = iou_matrix

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    matches = []
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        if pred_idx < num_preds and gt_idx < num_gts:
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > 0:
                matches.append((pred_idx, gt_idx, float(iou)))

    return matches


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """
    Associe les predictions aux verites terrain pour un seuil donne.
    Retourne (TP, FP, FN).
    """
    matches = _match_boxes_optimal(pred_boxes, gt_boxes, iou_threshold)
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    """Calcule Precision, Rappel et F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def pixel_to_freq_linear(y_pixel, total_height, f_max=0.5):
    """Convertit Pixel Y -> frequence normalisee sur l'axe lineaire."""
    y_pixel = min(max(float(y_pixel), 0.0), float(total_height))
    target_freq = f_max - (2 * f_max * y_pixel / total_height)
    return float(max(-f_max, min(f_max, target_freq)))


def clip_gt_box(annotation, params):
    """
    Construit la bbox GT visible dans la fenetre courante et ses parametres physiques.
    Retourne None si l'annotation n'est pas visible.
    """
    offset = params['offset']
    duration = params['duration']
    img_height = params['img_height']
    f_max = params['f_max']

    t0 = annotation['core:sample_start'] - offset
    t1 = t0 + annotation['core:sample_count']

    if t1 <= 0 or t0 >= duration:
        return None

    t0_clipped = max(0, t0)
    t1_clipped = min(duration, t1)

    f0 = max(-f_max, float(annotation['core:freq_lower_edge']))
    f1 = min(f_max, float(annotation['core:freq_upper_edge']))

    y_top = int(max(0, min(img_height, img_height * (f_max - f1) / (2 * f_max))))
    y_bottom = int(max(0, min(img_height, img_height * (f_max - f0) / (2 * f_max))))

    return {
        'bbox': (int(t0_clipped), int(y_top), int(t1_clipped - t0_clipped), int(y_bottom - y_top)),
        't0': float(t0_clipped),
        't1': float(t1_clipped),
        'f0': float(f0),
        'f1': float(f1),
        'label': annotation.get('core:description', ''),
    }


def box_to_physical_params(box, params):
    """Convertit une bbox pixel (x, y, w, h) en parametres physiques t0/t1/f0/f1."""
    x, y, w, h = box
    img_height = params['img_height']
    f_max = params['f_max']

    t0 = float(x)
    t1 = float(x + w)
    f1 = pixel_to_freq_linear(y, img_height, f_max)
    f0 = pixel_to_freq_linear(y + h, img_height, f_max)

    return {
        't0': t0,
        't1': t1,
        'f0': f0,
        'f1': f1,
    }

def enrich_physical_params(params_dict):
    """Ajoute tc, fc, B et D à partir de t0/t1/f0/f1."""
    enriched = dict(params_dict)
    enriched['tc'] = float((enriched['t0'] + enriched['t1']) / 2.0)
    enriched['fc'] = float((enriched['f0'] + enriched['f1']) / 2.0)
    enriched['B'] = float(enriched['f1'] - enriched['f0'])
    enriched['D'] = float(enriched['t1'] - enriched['t0'])
    return enriched


def match_boxes_detailed(pred_boxes, gt_boxes, iou_threshold):
    """
    Associe les predictions aux verites terrain pour un seuil donne.
    Retourne les metriques globales et le detail des matches.
    """
    gt_bboxes = [gt_item['bbox'] for gt_item in gt_boxes]
    optimal_matches = _match_boxes_optimal(pred_boxes, gt_bboxes, iou_threshold)
    matches = []

    for pred_idx, gt_idx, iou in sorted(optimal_matches, key=lambda item: item[0]):
        gt_item = gt_boxes[gt_idx]
        matches.append({
            'pred_index': pred_idx,
            'pred_box': tuple(int(v) for v in pred_boxes[pred_idx]),
            'gt_box': tuple(int(v) for v in gt_item['bbox']),
            'gt_label': gt_item.get('label', ''),
            'iou': float(iou),
            'gt_params': {
                't0': float(gt_item['t0']),
                't1': float(gt_item['t1']),
                'f0': float(gt_item['f0']),
                'f1': float(gt_item['f1']),
            },
        })

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def save_evaluation_json(report, output_json_path):
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapport d'evaluation sauvegarde : {output_path}")


def evaluate_coco_style(pred_boxes, gt_boxes, params=None, output_json_path=None):
    """
    Evaluation complete sur la plage IoU 0.5 -> 0.95 (10 steps).
    """
    # On definit les seuils comme dans le papier (de 0.5 a 0.95 par pas de 0.05)
    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    results = {}
    f1_scores = []

    print(f"\n EVALUATION DETAILLEE ({len(pred_boxes)} Preds vs {len(gt_boxes)} GT)")
    print(f"{'IoU Thresh':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)

    for thresh in iou_thresholds:
        simple_gt_boxes = [gt['bbox'] if isinstance(gt, dict) else gt for gt in gt_boxes]
        tp, fp, fn = match_boxes(pred_boxes, simple_gt_boxes, thresh)
        prec, rec, f1 = calculate_metrics(tp, fp, fn)

        results[thresh] = {'p': prec, 'r': rec, 'f1': f1}
        f1_scores.append(f1)

        print(f"{thresh:.2f}{' ':<8} | {prec:.2f}{' ':<6} | {rec:.2f}{' ':<6} | {f1:.2f}")

    avg_f1 = np.mean(f1_scores)
    print("-" * 50)
    print(f"SCORE FINAL (mF1 @ .50:.95) : {avg_f1:.4f}\n")

    report = {
        'summary': {
            'num_predictions': int(len(pred_boxes)),
            'num_metadata_boxes': int(len(gt_boxes)),
            'prediction_minus_metadata': int(len(pred_boxes) - len(gt_boxes)),
            'mf1_50_95': float(avg_f1),
        },
        'iou_sweep': {
            f"{thresh:.2f}": {
                'precision': float(values['p']),
                'recall': float(values['r']),
                'f1': float(values['f1']),
            }
            for thresh, values in results.items()
        },
    }

    if params is not None and gt_boxes and isinstance(gt_boxes[0], dict):
        tp, fp, fn, matches = match_boxes_detailed(pred_boxes, gt_boxes, 0.5)
        print("-----------------------")
        print("Rapport métriques :")
        print("-----------------------")
        print()
        print(f"Nb pred - Nb metadata = {len(pred_boxes)} - {len(gt_boxes)} = {len(pred_boxes) - len(gt_boxes)}")
        print(f"BBox valides (IoU >= 0.50) : {tp} | FP : {fp} | FN : {fn}")

        detailed_matches = []
        for match in matches:
            pred_params = enrich_physical_params(box_to_physical_params(match['pred_box'], params))
            gt_params = enrich_physical_params(match['gt_params'])
            deltas = {
                't0': float(gt_params['t0'] - pred_params['t0']),
                't1': float(gt_params['t1'] - pred_params['t1']),
                'f0': float(gt_params['f0'] - pred_params['f0']),
                'f1': float(gt_params['f1'] - pred_params['f1']),
                'tc': float(gt_params['tc'] - pred_params['tc']),
                'fc': float(gt_params['fc'] - pred_params['fc']),
                'B': float(gt_params['B'] - pred_params['B']),
                'D': float(gt_params['D'] - pred_params['D']),
            }

            detailed_match = {
                'pred_index': int(match['pred_index']),
                'gt_label': match['gt_label'],
                'iou': float(match['iou']),
                'prediction': pred_params,
                'metadata': gt_params,
                'delta_meta_minus_prediction': deltas,
            }
            detailed_matches.append(detailed_match)

            print(f"\nBBox valide #{match['pred_index']} | IoU = {match['iou']:.3f}")
            print(
                "Prediction : "
                f"t0={pred_params['t0']:.2f}, t1={pred_params['t1']:.2f}, "
                f"f0={pred_params['f0']:.6f}, f1={pred_params['f1']:.6f}, "
                f"tc={pred_params['tc']:.2f}, fc={pred_params['fc']:.6f}, "
                f"B={pred_params['B']:.6f}, D={pred_params['D']:.2f}"
            )
            print(
                "Metadata   : "
                f"t0={gt_params['t0']:.2f}, t1={gt_params['t1']:.2f}, "
                f"f0={gt_params['f0']:.6f}, f1={gt_params['f1']:.6f}, "
                f"tc={gt_params['tc']:.2f}, fc={gt_params['fc']:.6f}, "
                f"B={gt_params['B']:.6f}, D={gt_params['D']:.2f}"
            )
            print(
                "Delta meta - pred : "
                f"t0={deltas['t0']:.2f}, t1={deltas['t1']:.2f}, "
                f"f0={deltas['f0']:.6f}, f1={deltas['f1']:.6f}, "
                f"tc={deltas['tc']:.2f}, fc={deltas['fc']:.6f}, "
                f"B={deltas['B']:.6f}, D={deltas['D']:.2f}"
            )

        report['valid_matches_iou_0_50'] = detailed_matches
    if output_json_path is not None:
        report.setdefault('valid_matches_iou_0_50', [])
        save_evaluation_json(report, output_json_path)

    return avg_f1, results, report
