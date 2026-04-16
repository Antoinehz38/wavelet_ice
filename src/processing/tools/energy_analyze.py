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


    metrics = make_metrics(compressed_spec, gt_boxes_pixels)

    if debug == True:
        out = draw_boxes(compressed_spec, gt_boxes_pixels)
        cv2.imwrite('energy.png', out)
    

    return metrics


def calculate_bleed_metrics_multi(image, target_box, all_boxes, margin_ratio_bleed=0.2, margin_ratio_underfill=0.05, roll_off_tolerance=0.05):
    x, y, w, h = [int(v) for v in target_box]
    img_h, img_w = image.shape

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    empty_metrics = {
        'bleed_x': 0.0, 'bleed_y': 0.0, 'bleed_overall': 0.0,
        'underfill_x': 1.0, 'underfill_y': 1.0, 'underfill_overall': 1.0
    }

    if x2 <= x1 or y2 <= y1:
        return empty_metrics

    # Mask out other signal boxes
    valid_mask = np.ones((img_h, img_w), dtype=bool)
    for box in all_boxes:
        if box == target_box: continue
        bx, by, bw, bh = [int(v) for v in box]
        bx1, by1 = max(0, bx), max(0, by)
        bx2, by2 = min(img_w, bx + bw), min(img_h, by + bh)
        valid_mask[by1:by2, bx1:bx2] = False

    # Estimate background noise level
    bg_pixels = image[valid_mask]
    bg_level = np.median(bg_pixels) if len(bg_pixels) > 0 else np.median(image)

    inside_roi = image[y1:y2, x1:x2]
    mean_in = np.mean(inside_roi) - bg_level
    if mean_in <= 0:
        return empty_metrics

    # --- Underfill Calculation (with roll-off tolerance) ---
    margin_h_in = max(1, int((y2 - y1) * margin_ratio_underfill))
    margin_w_in = max(1, int((x2 - x1) * margin_ratio_underfill))
    
    core_y1, core_y2 = min(y1 + margin_h_in, y2), max(y2 - margin_h_in, y1)
    core_x1, core_x2 = min(x1 + margin_w_in, x2), max(x2 - margin_w_in, x1)
    
    core_roi = image[core_y1:core_y2, core_x1:core_x2]
    mean_core = np.mean(core_roi) - bg_level if core_roi.size > 0 else mean_in
    mean_core = max(1e-5, mean_core)
    
    in_top = image[y1:core_y1, x1:x2].flatten()
    in_bottom = image[core_y2:y2, x1:x2].flatten()
    in_y_pixels = np.concatenate([in_top, in_bottom])
    mean_in_y = np.mean(in_y_pixels) - bg_level if in_y_pixels.size > 0 else mean_in
    
    in_left = image[core_y1:core_y2, x1:core_x1].flatten()
    in_right = image[core_y1:core_y2, core_x2:x2].flatten()
    in_x_pixels = np.concatenate([in_left, in_right])
    mean_in_x = np.mean(in_x_pixels) - bg_level if in_x_pixels.size > 0 else mean_in
    
    in_all_pixels = np.concatenate([in_y_pixels, in_x_pixels])
    mean_in_overall = np.mean(in_all_pixels) - bg_level if in_all_pixels.size > 0 else mean_in

    threshold = mean_core * roll_off_tolerance
    
    fill_ratio_y = min(1.0, max(0, mean_in_y) / threshold)
    fill_ratio_x = min(1.0, max(0, mean_in_x) / threshold)
    fill_ratio_overall = min(1.0, max(0, mean_in_overall) / threshold)
    
    underfill_y = 1.0 - fill_ratio_y
    underfill_x = 1.0 - fill_ratio_x
    underfill_overall = 1.0 - fill_ratio_overall

    # --- Bleed Calculation ---
    margin_h = max(1, int(h * margin_ratio_bleed))
    margin_w = max(1, int(w * margin_ratio_bleed))
    
    y_pixels, x_pixels = [], []
    
    if y1 > 0:
        top_roi, top_mask = image[max(0, y1 - margin_h):y1, x1:x2], valid_mask[max(0, y1 - margin_h):y1, x1:x2]
        y_pixels.extend(top_roi[top_mask]) 
    if y2 < img_h:
        bot_roi, bot_mask = image[y2:min(img_h, y2 + margin_h), x1:x2], valid_mask[y2:min(img_h, y2 + margin_h), x1:x2]
        y_pixels.extend(bot_roi[bot_mask])

    if x1 > 0:
        left_roi, left_mask = image[y1:y2, max(0, x1 - margin_w):x1], valid_mask[y1:y2, max(0, x1 - margin_w):x1]
        x_pixels.extend(left_roi[left_mask])
    if x2 < img_w:
        right_roi, right_mask = image[y1:y2, x2:min(img_w, x2 + margin_w)], valid_mask[y1:y2, x2:min(img_w, x2 + margin_w)]
        x_pixels.extend(right_roi[right_mask])

    mean_out_y = np.mean(y_pixels) - bg_level if len(y_pixels) > 0 else 0
    mean_out_x = np.mean(x_pixels) - bg_level if len(x_pixels) > 0 else 0
    
    all_pixels_out = y_pixels + x_pixels
    mean_out_overall = np.mean(all_pixels_out) - bg_level if len(all_pixels_out) > 0 else 0

    return {
        'bleed_x': max(0, mean_out_x) / mean_in,
        'bleed_y': max(0, mean_out_y) / mean_in,
        'bleed_overall': max(0, mean_out_overall) / mean_in,
        'underfill_x': underfill_x,
        'underfill_y': underfill_y,
        'underfill_overall': underfill_overall
    }

def make_metrics(compressed_spec, gt_boxes_pixels, margin_ratio=0.2):
    all_boxes = [item[0] for item in gt_boxes_pixels]
    
    stats = defaultdict(lambda: {
        'b_x': [], 'b_y': [], 'b_all': [],
        'u_x': [], 'u_y': [], 'u_all': []
    })
    
    for box_coords, label in gt_boxes_pixels:
        metrics = calculate_bleed_metrics_multi(
            image=compressed_spec, target_box=box_coords, 
            all_boxes=all_boxes
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
    file_path = "/home/antoine/Downloads/raw_Bump_start_1628831_length_1000000.png"
    meta_path = "/home/antoine/Downloads/west-wideband-modrec-ex100-tmpl15-20.04.sigmf-meta"
    print('=== Analyse du fichier 1 ===')
    print(analyze_energy_on_one_image(file_path, meta_path, debug=True))

    file_path = "/home/antoine/Downloads/raw_Bump_start_0_length_1000000.png"
    meta_path = "/home/antoine/Downloads/west-wideband-modrec-ex29-tmpl8-20.04.sigmf-meta"
    print('\n=== Analyse du fichier 2 ===')
    print(analyze_energy_on_one_image(file_path, meta_path, debug=True))