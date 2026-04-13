import os, cv2, datetime
import numpy as np
from dataclasses import dataclass
from src.data_processing.tools.viz import save_viz_comparison, compress_spectrogram
from src.data_processing.tools import evaluations, dsp, vision, dsp_rc
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet

from src.data_processing.tools.loaders import load_iq_data
from src.cwt_scheduler import TimeWindow


def run_signal_processing_pipeline(input_file: str, meta: dict, output_dir: str, time_window: TimeWindow, params: dict) -> None:
    # Bornes absolues de la fenêtre temporelle courante
    win_start = time_window.start
    win_end   = time_window.start + time_window.length
    sig = load_iq_data(input_file, num_samples=time_window.length, offset=time_window.start)

    if sig is None: return None

    if params.get('transform', 'cwt') == 'cwt':
        spec = dsp.compute_dual_linear_cwt(
            sig, params['wavelet'], params['img_height'],
            params['f_min'], params['f_max'], params['fs']
        )

    elif params['transform'] == 'cwt_rc':
        rc = RaisedCosineWavelet(
            fc=params['rc_fc'],
            B=params['rc_B'],
            beta=params['rc_beta']
        )
        spec = dsp_rc.compute_dual_linear_cwt_rc(
            sig, rc, params['img_height'],
            params['f_min'], params['f_max'], params['fs']
        )
    else:
        raise ValueError("PARAMS['transform'] doit être 'cwt' ou 'cwt_rc'")


    
    compressed_spec = compress_spectrogram(spec, params['downsample_factor'])

    if params.get('saveRaw', False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spectrogram_{timestamp}.png"
        cv2.imwrite(os.path.join(output_dir, filename), compressed_spec)
        return None

    boxes = []
    gt_boxes_pixels = []
    if params.get('addPrediction', False):
        print("adding prediction ... ")
       
        boxes = vision.detect_box(compressed_spec, delta_t=100, intensite_lissage=0.015, roll_off_threshold=0.30)
        print(f"-> {len(boxes)} objets détectés.")

        # Facteurs de conversion spectrogramme original -> image compressée
        img_h, img_w = compressed_spec.shape[:2]
        win_len = time_window.length  # nombre d'échantillons dans la fenêtre

        # Ratios : échantillons/pixels → pixels compressés
        scale_x = img_w / win_len
        scale_y = img_h / params['img_height']

        gt_boxes_pixels = []
        if meta:
            for ann in meta.get("annotations", []):
                ann_start = ann['core:sample_start']
                ann_end   = ann_start + ann['core:sample_count']

                # Ne garder que les annotations qui chevauchent la fenêtre courante
                if ann_end <= win_start or ann_start >= win_end:
                    continue

                y_start = dsp.freq_to_pixel_linear(ann['core:freq_upper_edge'], params['img_height'], params['f_max'])
                y_end = dsp.freq_to_pixel_linear(ann['core:freq_lower_edge'], params['img_height'], params['f_max'])

                # Sécurité bornes (dans le repère original)
                y_start = max(y_start, 0)
                y_end   = min(y_end, params['img_height'])

                # Coordonnées X relatives à la fenêtre, clampées aux bords
                x_rel = max(ann_start - win_start, 0)
                x_end = min(ann_end,   win_end) - win_start
                w = x_end - x_rel
                h = y_end - y_start

                # Conversion vers le repère de l'image compressée
                cx = x_rel * scale_x
                cw = w * scale_x
                cy = y_start * scale_y
                ch = h * scale_y

                label = ann.get('core:description', 'GT')
                gt_boxes_pixels.append((cx, cy, cw, ch, label))

        # --- 3c. Lancement Évaluation ---
        if len(gt_boxes_pixels) > 0:
            evaluations.evaluate_coco_style(boxes, gt_boxes_pixels)
        else:
            print("Pas de Vérité Terrain disponible pour l'évaluation.")

    # Generate filename for the visualization
    if params['transform'] == 'cwt_rc':
        wavelet_name = "Raised_Cosine"
    elif params['transform'] == 'cwt':
        wavelet_name = params['wavelet'].replace('/', '_')
    else:
        wavelet_name = "Wavelet"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = os.path.basename(input_file).replace(".sigmf-data", "")
    filename = f"{input_name}_{wavelet_name}_start_{time_window.start}_length_{time_window.length}.png"
    filepath = os.path.join(output_dir, filename)

    save_viz_comparison(compressed_spec, gt_boxes_pixels, boxes, filepath, params)

    return None