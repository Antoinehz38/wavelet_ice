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

def evaluate_coco_style(pred_boxes, gt_boxes):
    """
    Évaluation complète sur la plage IoU 0.5 -> 0.95 (10 steps).
    """
    # On définit les seuils comme dans le papier (de 0.5 à 0.95 par pas de 0.05)
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    
    results = {}
    f1_scores = []
    
    print(f"\n📊 ÉVALUATION DÉTAILLÉE ({len(pred_boxes)} Preds vs {len(gt_boxes)} GT)")
    print(f"{'IoU Thresh':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)
    
    for thresh in iou_thresholds:
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, thresh)
        prec, rec, f1 = calculate_metrics(tp, fp, fn)
        
        results[thresh] = {'p': prec, 'r': rec, 'f1': f1}
        f1_scores.append(f1)
        
        print(f"{thresh:.2f}{' ':<8} | {prec:.2f}{' ':<6} | {rec:.2f}{' ':<6} | {f1:.2f}")
        
    avg_f1 = np.mean(f1_scores)
    print("-" * 50)
    print(f"🏆 SCORE FINAL (mF1 @ .50:.95) : {avg_f1:.4f}\n")
    
    return avg_f1, results