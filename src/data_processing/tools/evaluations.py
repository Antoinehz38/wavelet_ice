import json
from pathlib import Path

import numpy as np

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

    # Calcul des coordonnées de l'intersection
    x_inter_1 = max(xA_1, xB_1)
    y_inter_1 = max(yA_1, yB_1)
    x_inter_2 = min(xA_2, xB_2)
    y_inter_2 = min(yA_2, yB_2)

    # Aire de l'intersection (on clamp à 0 si pas de superposition)
    inter_w = max(0, x_inter_2 - x_inter_1)
    inter_h = max(0, y_inter_2 - y_inter_1)
    inter_area = inter_w * inter_h

    # Aires des boîtes individuelles
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    # Aire de l'Union = Aire A + Aire B - Aire Intersection
    union_area = boxA_area + boxB_area - inter_area

    # Protection division par zéro
    if union_area == 0: return 0.0

    return inter_area / union_area

def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """
    Associe les prédictions aux vérités terrain pour un seuil donné.
    Retourne (TP, FP, FN).
    """
    # Copies pour ne pas modifier les listes originales
    preds = list(pred_boxes)
    gts = list(gt_boxes)
    
    tp = 0
    fp = 0
    
    # Pour chaque prédiction, on cherche le meilleur match GT
    # Note: Dans un vrai coco-eval, on trie d'abord par score de confiance.
    # Ici on prend l'ordre de la liste (souvent géométrique).
    
    for p_box in preds:
        best_iou = 0
        best_gt_idx = -1
        
        # On cherche le GT qui chevauche le plus cette prédiction
        for i, gt_box in enumerate(gts):
            iou = compute_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # Verdict pour cette prédiction
        if best_iou >= iou_threshold:
            # C'est un MATCH !
            tp += 1
            # On retire ce GT de la liste pour ne pas le matcher deux fois
            # (Un GT ne peut être trouvé qu'une seule fois)
            gts.pop(best_gt_idx)
        else:
            # C'est une fausse alarme (False Positive)
            fp += 1
            
    # Les GT restants n'ont pas été trouvés (False Negatives)
    fn = len(gts)
    
    return tp, fp, fn

def calculate_metrics(tp, fp, fn):
    """Calcule Précision, Rappel et F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def pixel_to_freq_linear(y_pixel, total_height, f_max=0.5):
    """Convertit Pixel Y -> fréquence normalisée sur l'axe linéaire."""
    y_pixel = min(max(float(y_pixel), 0.0), float(total_height))
    target_freq = f_max - (2 * f_max * y_pixel / total_height)
    return float(max(-f_max, min(f_max, target_freq)))

def clip_gt_box(annotation, params):
    """
    Construit la bbox GT visible dans la fenêtre courante et ses paramètres physiques.
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
    """Convertit une bbox pixel (x, y, w, h) en paramètres physiques t0/t1/f0/f1."""
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

def match_boxes_detailed(pred_boxes, gt_boxes, iou_threshold):
    """
    Associe les prédictions aux vérités terrain pour un seuil donné.
    Retourne les métriques globales et le détail des matches.
    """
    remaining_gts = list(gt_boxes)
    matches = []
    fp = 0

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_item in enumerate(remaining_gts):
            iou = compute_iou(pred_box, gt_item['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_item = remaining_gts.pop(best_gt_idx)
            matches.append({
                'pred_index': pred_idx,
                'pred_box': tuple(int(v) for v in pred_box),
                'gt_box': tuple(int(v) for v in gt_item['bbox']),
                'gt_label': gt_item.get('label', ''),
                'iou': float(best_iou),
                'gt_params': {
                    't0': float(gt_item['t0']),
                    't1': float(gt_item['t1']),
                    'f0': float(gt_item['f0']),
                    'f1': float(gt_item['f1']),
                },
            })
        else:
            fp += 1

    tp = len(matches)
    fn = len(remaining_gts)
    return tp, fp, fn, matches

def save_evaluation_json(report, output_json_path):
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapport d'évaluation sauvegardé : {output_path}")

def evaluate_coco_style(pred_boxes, gt_boxes, params=None, output_json_path=None):
    """
    Évaluation complète sur la plage IoU 0.5 -> 0.95 (10 steps).
    """
    # On définit les seuils comme dans le papier (de 0.5 à 0.95 par pas de 0.05)
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    
    results = {}
    f1_scores = []
    
    print(f"\n ÉVALUATION DÉTAILLÉE ({len(pred_boxes)} Preds vs {len(gt_boxes)} GT)")
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
        print(f"Nb pred - Nb metadata = {len(pred_boxes)} - {len(gt_boxes)} = {len(pred_boxes) - len(gt_boxes)}")
        print(f"BBox valides (IoU >= 0.50) : {tp} | FP : {fp} | FN : {fn}")

        detailed_matches = []
        for match in matches:
            pred_params = box_to_physical_params(match['pred_box'], params)
            gt_params = match['gt_params']
            deltas = {
                't0': float(gt_params['t0'] - pred_params['t0']),
                't1': float(gt_params['t1'] - pred_params['t1']),
                'f0': float(gt_params['f0'] - pred_params['f0']),
                'f1': float(gt_params['f1'] - pred_params['f1']),
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
                f"f0={pred_params['f0']:.6f}, f1={pred_params['f1']:.6f}"
            )
            print(
                "Metadata   : "
                f"t0={gt_params['t0']:.2f}, t1={gt_params['t1']:.2f}, "
                f"f0={gt_params['f0']:.6f}, f1={gt_params['f1']:.6f}"
            )
            print(
                "Delta meta - pred : "
                f"t0={deltas['t0']:.2f}, t1={deltas['t1']:.2f}, "
                f"f0={deltas['f0']:.6f}, f1={deltas['f1']:.6f}"
            )

        report['valid_matches_iou_0_50'] = detailed_matches
    if output_json_path is not None:
        report.setdefault('valid_matches_iou_0_50', [])
        save_evaluation_json(report, output_json_path)

    return avg_f1, results, report
