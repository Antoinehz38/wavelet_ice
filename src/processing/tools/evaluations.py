from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Dict, Any
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


from src.cwt_scheduler import TimeWindow

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
    img_width = params['img_width']
    duration = params['duration']

    time_scale = duration / img_width if img_width > 0 else 0.0

    t0 = float(x * time_scale)
    t1 = float((x + w) * time_scale)
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


def compute_relative_percent_deltas(reference_params, deltas):
    """Calcule les deltas relatifs en pourcentage par rapport aux metadata."""
    percent_deltas = {}
    for key, delta_value in deltas.items():
        reference_value = float(reference_params[key])
        if np.isclose(reference_value, 0.0):
            percent_deltas[key] = None
        else:
            percent_deltas[key] = float((delta_value / reference_value) * 100.0)
    return percent_deltas


def format_metric_values(values, formats, suffix_map=None):
    """Formate une ligne de metriques avec suffixes eventuels."""
    rendered = []
    for key, fmt in formats.items():
        value = values.get(key)
        suffix = "" if suffix_map is None else suffix_map.get(key, "")
        if value is None:
            rendered.append(f"{key}=n/a")
        else:
            rendered.append(f"{key}={value:{fmt}}{suffix}")
    return ", ".join(rendered)


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
        print("-----------------------")
        print("Rapport métriques :")
        print("-----------------------")
        print()
        print(f"Nb pred - Nb metadata = {len(pred_boxes)} - {len(gt_boxes)} = {len(pred_boxes) - len(gt_boxes)}")
        metric_formats = {
            't0': '.2f',
            't1': '.2f',
            'f0': '.4f',
            'f1': '.4f',
            'tc': '.2f',
            'fc': '.4f',
            'B': '.4f',
            'D': '.2f',
        }
        percent_suffixes = {key: '%' for key in metric_formats}
        valid_matches_by_iou = {}

        for thresh in iou_thresholds:
            tp, fp, fn, matches = match_boxes_detailed(pred_boxes, gt_boxes, thresh)
            print()
            print("" + "-"*30)
            print(f"IoU >= {thresh:.2f} : {tp} valides")

            detailed_matches = []
            avg_L1_score, avg_L2_score, avg_accuracy_percent = 0.0, 0.0, 0.0    
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
                deltas_percent = compute_relative_percent_deltas(gt_params, deltas)

                detailed_match = {
                    'pred_index': int(match['pred_index']),
                    'gt_label': match['gt_label'],
                    'iou': float(match['iou']),
                    'score_L1': score_L1(pred_params, gt_params),
                    'score_L2': score_L2(pred_params, gt_params),
                    'accuracy_percent': score_accuracy(pred_params, gt_params),
                    'prediction': pred_params,
                    'metadata': gt_params,
                    'delta_meta_minus_prediction': deltas,
                    'delta_meta_minus_prediction_percent': deltas_percent,
                }
                detailed_matches.append(detailed_match)
                avg_L1_score = avg_L1_score + detailed_match['score_L1']
                avg_L2_score = avg_L2_score + detailed_match['score_L2']
                avg_accuracy_percent = avg_accuracy_percent + detailed_match['accuracy_percent']

                print(f"\nBBox valide #{match['pred_index']} | IoU = {match['iou']:.3f}")
                print(f"Score L1 : {detailed_match['score_L1']:.3f}")
                print(f"Score L2 : {detailed_match['score_L2']:.3f}")
                print(f"Accuracy moyenne en % : {detailed_match['accuracy_percent']:.2f}%")
                print(f"Prediction : {format_metric_values(pred_params, metric_formats)}")
                print(f"Metadata   : {format_metric_values(gt_params, metric_formats)}")
                print(f"Delta meta - pred : {format_metric_values(deltas, metric_formats)}")
                print(f"Delta meta - pred (%) : {format_metric_values(deltas_percent, metric_formats, percent_suffixes)}")

            avg_L1_score = avg_L1_score / tp if tp > 0 else None
            avg_L2_score = avg_L2_score / tp if tp > 0 else None # on pourrait diviser par len(matches), mais je pars du principe que tp = len(matches) et que c'est plus clair de faire le lien direct
            avg_accuracy_percent = avg_accuracy_percent / tp if tp > 0 else None

            valid_matches_by_iou[f"{thresh:.2f}"] = {

                "avg_L1_score": avg_L1_score,
                "avg_L2_score": avg_L2_score,
                "avg_accuracy_percent": avg_accuracy_percent,
                "matches": detailed_matches,
            }

        report['valid_matches_by_iou'] = valid_matches_by_iou
        report['valid_matches_iou_0_50'] = valid_matches_by_iou.get('0.50', [])
    if output_json_path is not None:
        report.setdefault('valid_matches_by_iou', {})
        report.setdefault('valid_matches_iou_0_50', [])
        save_evaluation_json(report, output_json_path)

    return avg_f1, results, report



def score_accuracy(pred_params, gt_params):
    delta_t0_percent = abs(gt_params['t0'] - pred_params['t0']) / abs(gt_params['t0']) * 100 if not np.isclose(gt_params['t0'], 0.0) else 0
    delta_t1_percent = abs(gt_params['t1'] - pred_params['t1']) / abs(gt_params['t1']) * 100 if not np.isclose(gt_params['t1'], 0.0) else 0
    delta_f0_percent = abs(gt_params['f0'] - pred_params['f0']) / abs(gt_params['f0']) * 100 if not np.isclose(gt_params['f0'], 0.0) else 0
    delta_f1_percent = abs(gt_params['f1'] - pred_params['f1']) / abs(gt_params['f1']) * 100 if not np.isclose(gt_params['f1'], 0.0) else 0

    average_accuracy_percent = 100 - np.mean([delta_t0_percent, delta_t1_percent, delta_f0_percent, delta_f1_percent])
    return average_accuracy_percent

def score_L1(pred_params, gt_params):
    delta_t0 = abs(gt_params['t0'] - pred_params['t0'])
    delta_t1 = abs(gt_params['t1'] - pred_params['t1'])
    delta_f0 = abs(gt_params['f0'] - pred_params['f0'])
    delta_f1 = abs(gt_params['f1'] - pred_params['f1'])
    return delta_t0 + delta_t1 + delta_f0 + delta_f1

def score_L2(pred_params, gt_params):
    delta_t0 = gt_params['t0'] - pred_params['t0']
    delta_t1 = gt_params['t1'] - pred_params['t1']
    delta_f0 = gt_params['f0'] - pred_params['f0']
    delta_f1 = gt_params['f1'] - pred_params['f1']
    return np.sqrt(delta_t0**2 + delta_t1**2 + delta_f0**2 + delta_f1**2)   

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

