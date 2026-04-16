import cv2
import os
import re
import numpy as np
from collections import defaultdict

from src.processing.tools.loaders import load_metadata
from src.processing.tools.dsp import freq_to_pixel_linear


PARAMS = {
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 1_000_000,
    'f_min': 0.005,
    'f_max': 0.5,
    
    'rc_fc': 1.0,            
    'rc_B': 0.06,            
    'rc_beta': 0.12,         
    
    'detect_db_range': 28,
    'detect_kernel': (200, 2),
    'downsample_factor': 500,
    'saveRaw': False,
    'addPrediction': False,

    'stft_nperseg': 256, #taille fenetre, si grand : meilleure resolution temporelle, moins frequentielle et inversement
    'stft_noverlap': 192, #recouvrement, souvent 75% de nperseg
    'stft_nfft': 512,          # mettre None pour prendre nperseg, si nfft > nperseg → zero-padding
    'stft_window': 'hann',     # fenetre de hann ou de hamming ou autre, il faudrait en tester plusieurs
  
    'dwt_wavelet': 'db4',
    'dwt_mode': 'periodization',
    'dwt_level': 256,         # None => niveau max automatique
    'dwt_use_abs': True,       # |coeff|^2
    'dwt_include_approx': False,

    'bump_fc': 0.30,        # w=2*pi*fc
    'bump_B': 0.005,         # demi-largeur du support : sigma= 2*pi*B

    'morse_beta': 1150.0,
    'morse_gamma': 3.0, #produit temps frequence P=beta/gamma
}

def bbox_from_meta(meta_path, window_start, window_end, ds, scale_y) -> list:
    meta = load_metadata(meta_path)
    gt_boxes_pixels = []

    if meta:
        for ann in meta.get("annotations", []):
            signal_start = ann['core:sample_start']
            signal_count = ann['core:sample_count']
            signal_end = signal_start + signal_count
            overlap_start = max(signal_start, window_start)
            overlap_end = min(signal_end, window_end)
            if overlap_start < overlap_end:
                local_x = overlap_start - window_start
                local_w = overlap_end - overlap_start

                y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], PARAMS['img_height'], PARAMS['f_max'])
                y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], PARAMS['img_height'], PARAMS['f_max'])

                if y_start < 0:
                    y_start = 0
                if y_end > PARAMS['img_height']:
                    y_end = PARAMS['img_height']
                
                h = y_end - y_start

                cx = int(local_x / ds)
                cw = int(local_w / ds)
                cy = int(y_start * scale_y)
                ch = int(h * scale_y)

                gt_boxes_pixels.append(((cx, cy, cw, ch),ann.get('core:description', 'UNKNOWN'))) 
    return gt_boxes_pixels


def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for box, name in boxes:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def analyze_energy_on_one_image(file_path, meta_path, debug=False):
    filename = os.path.basename(file_path)
    match = re.search(r'start_(\d+)_length_(\d+)', filename)
    if not match:
        print(f"Erreur : Impossible de lire la fenêtre dans le nom {filename}")
        return None

    window_start = int(match.group(1))
    window_length = int(match.group(2))
    window_end = window_start + window_length

    compressed_spec = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if compressed_spec is None:
        print(f"Erreur : Impossible de charger l'image {file_path}")
        return None

    img_h = compressed_spec.shape[0]
    ds = 250
    scale_y = img_h / PARAMS['img_height']

    gt_boxes_pixels = bbox_from_meta(meta_path, window_start, window_end, ds, scale_y)


    metrics = make_metrics(compressed_spec, gt_boxes_pixels, debug=debug)

    if debug == True:
        out = draw_boxes(compressed_spec, gt_boxes_pixels)
        cv2.imwrite('energy_plus.png', out)
    

    return metrics


import numpy as np

import numpy as np
import cv2

def calculate_bleed_metrics_multi(image, target_box, all_boxes, time_margin=50, 
                                  freq_bleed_ratio=1.5, margin_ratio_underfill=0.05, 
                                  roll_off_tolerance=0.05, debug=False):
    x, y, w, h = [int(v) for v in target_box]
    img_h, img_w = image.shape

    empty_metrics = {
        'bleed_x': 0.0, 'bleed_y': 0.0, 'bleed_overall': 0.0,
        'underfill_x': 1.0, 'underfill_y': 1.0, 'underfill_overall': 1.0
    }
    
    if w <= 0 or h <= 0 or x >= img_w or y >= img_h:
        return empty_metrics

    # 1. DÉCOUPAGE DE LA ROI (Spatio-Temporelle)
    freq_margin = int(h * freq_bleed_ratio)
    
    x1_roi = max(0, x - time_margin)
    x2_roi = min(img_w, x + w + time_margin)
    y1_roi = max(0, y - freq_margin)
    y2_roi = min(img_h, y + h + freq_margin)

    # Extraction de la sous-image locale
    roi_image = image[y1_roi:y2_roi, x1_roi:x2_roi]
    roi_h, roi_w = roi_image.shape

    # Recalibrage des coordonnées de la cible dans le nouveau repère de la ROI
    cx = x - x1_roi
    cy = y - y1_roi

    # 2. MASQUAGE DES VOISINS DANS LA ROI
    bg_mask = np.ones((roi_h, roi_w), dtype=bool)
    bleed_mask = np.ones((roi_h, roi_w), dtype=bool)

    for box in all_boxes:
        if box == target_box: continue
        bx, by, bw, bh = [int(v) for v in box]
        
        # Vérifier si la boîte voisine intersecte notre ROI
        if bx < x2_roi and (bx + bw) > x1_roi and by < y2_roi and (by + bh) > y1_roi:
            # Conversion des coordonnées du voisin dans le repère de la ROI
            bx1_roi = max(0, bx - x1_roi)
            by1_roi = max(0, by - y1_roi)
            bx2_roi = min(roi_w, bx + bw - x1_roi)
            by2_roi = min(roi_h, by + bh - y1_roi)
            
            # Masquer le voisin
            bg_mask[by1_roi:by2_roi, bx1_roi:bx2_roi] = False
            bleed_mask[by1_roi:by2_roi, bx1_roi:bx2_roi] = False

    # Retirer également la boîte cible du bg_mask pour avoir le vrai bruit de fond
    bg_mask[max(0, cy):min(roi_h, cy+h), max(0, cx):min(roi_w, cx+w)] = False

    # --- BLOC DE DEBUG (Sauvegarde de l'image masquée) ---
    if debug:
        # On copie l'image ROI pour ne pas altérer les vraies données
        image_masqued = roi_image.copy()
        
        # On met en noir tout ce qui a été exclu par le bleed_mask (les signaux voisins)
        image_masqued[~bleed_mask] = 0
        
        # Optionnel mais pratique : dessiner un rectangle clair (valeur 255) autour de notre cible
        cv2.rectangle(image_masqued, (cx, cy), (cx + w, cy + h), (255), 1)

        filename = f"mask_{np.random.uniform(0, 1):.4f}.png"
        cv2.imwrite(filename, image_masqued)
    # -----------------------------------------------------

    # 3. CALCUL DU BRUIT DE FOND LOCAL
    bg_pixels = roi_image[bg_mask]
    bg_level = np.median(bg_pixels) if len(bg_pixels) > 0 else np.median(roi_image)

    # 4. CALCUL DE L'ÉNERGIE INTERNE (Cible)
    inside_pixels = roi_image[max(0, cy):min(roi_h, cy+h), max(0, cx):min(roi_w, cx+w)]
    mean_in = np.mean(inside_pixels) - bg_level
    if mean_in <= 0: return empty_metrics

    # --- 5. CALCUL UNDERFILL (Vide interne) ---
    margin_h_in = max(1, int(h * margin_ratio_underfill))
    margin_w_in = max(1, int(w * margin_ratio_underfill))
    
    core_y1, core_y2 = min(cy + margin_h_in, cy + h), max(cy + h - margin_h_in, cy)
    core_x1, core_x2 = min(cx + margin_w_in, cx + w), max(cx + w - margin_w_in, cx)
    
    core_roi = roi_image[core_y1:core_y2, core_x1:core_x2]
    mean_core = np.mean(core_roi) - bg_level if core_roi.size > 0 else mean_in
    threshold = max(1e-5, mean_core) * roll_off_tolerance

    # Bords verticaux (Haut/Bas)
    in_y_pixels = np.concatenate([
        roi_image[cy:core_y1, cx:cx+w].flatten(), 
        roi_image[core_y2:cy+h, cx:cx+w].flatten()
    ])
    mean_in_y = np.mean(in_y_pixels) - bg_level if in_y_pixels.size > 0 else mean_in
    
    # Bords horizontaux (Gauche/Droite)
    in_x_pixels = np.concatenate([
        roi_image[core_y1:core_y2, cx:core_x1].flatten(), 
        roi_image[core_y1:core_y2, core_x2:cx+w].flatten()
    ])
    mean_in_x = np.mean(in_x_pixels) - bg_level if in_x_pixels.size > 0 else mean_in

    underfill_y = 1.0 - min(1.0, max(0, mean_in_y) / threshold)
    underfill_x = 1.0 - min(1.0, max(0, mean_in_x) / threshold)
    
    # --- 6. CALCUL BLEED (Bavement externe) ---
    y_pixels_out, x_pixels_out = [], []
    
    # Extraction du bleed Y en utilisant le masque pour ignorer les éventuels voisins
    if cy > 0:
        top_roi = roi_image[0:cy, cx:cx+w]
        y_pixels_out.extend(top_roi[bleed_mask[0:cy, cx:cx+w]])
    if cy + h < roi_h:
        bot_roi = roi_image[cy+h:roi_h, cx:cx+w]
        y_pixels_out.extend(bot_roi[bleed_mask[cy+h:roi_h, cx:cx+w]])

    # Extraction du bleed X
    if cx > 0:
        left_roi = roi_image[cy:cy+h, 0:cx]
        x_pixels_out.extend(left_roi[bleed_mask[cy:cy+h, 0:cx]])
    if cx + w < roi_w:
        right_roi = roi_image[cy:cy+h, cx+w:roi_w]
        x_pixels_out.extend(right_roi[bleed_mask[cy:cy+h, cx+w:roi_w]])

    mean_out_y = np.mean(y_pixels_out) - bg_level if len(y_pixels_out) > 0 else 0
    mean_out_x = np.mean(x_pixels_out) - bg_level if len(x_pixels_out) > 0 else 0
    all_out = y_pixels_out + x_pixels_out
    mean_out_all = np.mean(all_out) - bg_level if len(all_out) > 0 else 0

    return {
        'bleed_x': max(0, mean_out_x) / mean_in,
        'bleed_y': max(0, mean_out_y) / mean_in,
        'bleed_overall': max(0, mean_out_all) / mean_in,
        'underfill_x': underfill_x,
        'underfill_y': underfill_y,
        'underfill_overall': (underfill_x + underfill_y) / 2
    }

def make_metrics(compressed_spec, gt_boxes_pixels, margin_ratio=0.2, debug=False):
    all_boxes = [item[0] for item in gt_boxes_pixels]
    
    stats = defaultdict(lambda: {
        'b_x': [], 'b_y': [], 'b_all': [],
        'u_x': [], 'u_y': [], 'u_all': []
    })
    
    for box_coords, label in gt_boxes_pixels:
        metrics = calculate_bleed_metrics_multi(
            image=compressed_spec, target_box=box_coords, 
            all_boxes=all_boxes, debug=debug
        )
        
        stats[label]['b_x'].append(metrics['bleed_x'])
        stats[label]['b_y'].append(metrics['bleed_y'])
        stats[label]['b_all'].append(metrics['bleed_overall'])
        
        stats[label]['u_x'].append(metrics['underfill_x'])
        stats[label]['u_y'].append(metrics['underfill_y'])
        stats[label]['u_all'].append(metrics['underfill_overall'])
        
    final_report = {}
    for label, s in stats.items():
        if s['b_all']: 
            final_report[label] = {
                'count': len(s['b_all']),
                'bleed': {
                    'x': round(np.mean(s['b_x']), 4),
                    'y': round(np.mean(s['b_y']), 4),
                    'overall': round(np.mean(s['b_all']), 4)
                },
                'underfill': {
                    'x': round(np.mean(s['u_x']), 4),
                    'y': round(np.mean(s['u_y']), 4),
                    'overall': round(np.mean(s['u_all']), 4)
                }
            }
            
    return final_report


def analyze_energy_dataset(image_path_list, meta_path):
    global_accumulator = {}

    for file_path in image_path_list:
        frame_metrics = analyze_energy_on_one_image(file_path, meta_path)

        if not frame_metrics:
            continue

        for modulation_name, data in frame_metrics.items():
            
            if modulation_name not in global_accumulator:
                global_accumulator[modulation_name] = {
                    'count': 0,
                    'b_x_sum': 0.0, 'b_y_sum': 0.0, 'b_all_sum': 0.0,
                    'u_x_sum': 0.0, 'u_y_sum': 0.0, 'u_all_sum': 0.0
                }

            count = data['count']
            global_accumulator[modulation_name]['count'] += count

            global_accumulator[modulation_name]['b_x_sum'] += data['bleed']['x'] * count
            global_accumulator[modulation_name]['b_y_sum'] += data['bleed']['y'] * count
            global_accumulator[modulation_name]['b_all_sum'] += data['bleed']['overall'] * count

            global_accumulator[modulation_name]['u_x_sum'] += data['underfill']['x'] * count
            global_accumulator[modulation_name]['u_y_sum'] += data['underfill']['y'] * count
            global_accumulator[modulation_name]['u_all_sum'] += data['underfill']['overall'] * count

    final_global_report = {}
    
    for modulation_name, stats in global_accumulator.items():
        total_count = stats['count']
        if total_count > 0:
            final_global_report[modulation_name] = {
                'total_count': total_count,
                'bleed': {
                    'x': round(stats['b_x_sum'] / total_count, 4),
                    'y': round(stats['b_y_sum'] / total_count, 4),
                    'overall': round(stats['b_all_sum'] / total_count, 4)
                },
                'underfill': {
                    'x': round(stats['u_x_sum'] / total_count, 4),
                    'y': round(stats['u_y_sum'] / total_count, 4),
                    'overall': round(stats['u_all_sum'] / total_count, 4)
                }
            }

    return final_global_report



if __name__ == "__main__":
    # Exemple d'utilisation
    file_path = "/home/antoine/Downloads/raw_STFT_start_0_length_1000000.png"
    meta_path = "/home/antoine/Downloads/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta"
    print('=== Analyse du fichier 1 ===')
    print(analyze_energy_on_one_image(file_path, meta_path, debug=True))

    file_path = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/raw_Raised_Cosine_start_106028_length_1000000.png"
    meta_path = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
    print('\n=== Analyse du fichier 2 ===')
    print(analyze_energy_on_one_image(file_path, meta_path, debug=False))