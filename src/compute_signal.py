import os, time
import datetime

import cv2

from src.cwt_scheduler import TimeWindow
from src.processing.tools.dsp import compute_dual_linear_cwt, freq_to_pixel_linear
from src.processing.tools.dsp_rc import compute_dual_linear_cwt_rc
from src.processing.tools.evaluations import evaluate_coco_style
from src.processing.tools.loaders import load_iq_data
from src.processing.tools.raised_cosine import RaisedCosineWavelet
from src.processing.tools.vision import detect_box
from src.processing.tools.viz import save_viz_comparison, compress_spectrogram


def run_signal_processing_pipeline(input_file: str, meta: dict, output_dir: str, time_window: TimeWindow, params: dict) -> tuple[list, list]:
    # Absolute bounds of the current time window
    win_start = time_window.start
    win_end   = time_window.start + time_window.length
    sig = load_iq_data(input_file, num_samples=time_window.length, offset=time_window.start)

    if sig is None:
        return None

    if params.get('transform', 'cwt') == 'cwt':
        t = time.time()
        spec = compute_dual_linear_cwt(
            sig, params['wavelet'], params['img_height'],
            params['f_min'], params['f_max'], params['fs'],
        )
        print(f"CWT computation time: {time.time() - t:.2f} seconds")
    elif params['transform'] == 'cwt_rc':
        t = time.time()
        rc = RaisedCosineWavelet(
            fc=params['rc_fc'],
            B=params['rc_B'],
            beta=params['rc_beta'],
        )
        spec = compute_dual_linear_cwt_rc(
            sig, rc, params['img_height'],
            params['f_min'], params['f_max'], params['fs'],
        )
        print(f"Raised-cosine CWT computation time: {time.time() - t:.2f} seconds")
    else:
        raise ValueError("params['transform'] must be 'cwt' or 'cwt_rc'")


    
    compressed_spec = compress_spectrogram(spec, params['downsample_factor'])

    if params.get('saveRaw', False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spectrogram_{timestamp}.png"
        cv2.imwrite(os.path.join(output_dir, filename), compressed_spec)
        return None

    boxes = []
    gt_boxes_pixels = []
    if params.get('addPrediction', False):
        print("Adding prediction...")

        boxes = detect_box(compressed_spec, delta_t=100, smoothing_intensity=0.015, roll_off_threshold=0.30)
        print(f"-> {len(boxes)} objects detected.")

        img_h, img_w = compressed_spec.shape[:2]
        win_len = time_window.length

        scale_x = img_w / win_len
        scale_y = img_h / params['img_height']

        gt_boxes_pixels = []
        if meta:
            for ann in meta.get("annotations", []):
                ann_start = ann['core:sample_start']
                ann_end = ann_start + ann['core:sample_count']

                if ann_end <= win_start or ann_start >= win_end:
                    continue

                y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], params['img_height'], params['f_max'])
                y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], params['img_height'], params['f_max'])

                y_start = max(y_start, 0)
                y_end = min(y_end, params['img_height'])

                x_rel = max(ann_start - win_start, 0)
                x_end = min(ann_end, win_end) - win_start
                w = x_end - x_rel
                h = y_end - y_start

                cx = x_rel * scale_x
                cw = w * scale_x
                cy = y_start * scale_y
                ch = h * scale_y

                label = ann.get('core:description', 'GT')
                gt_boxes_pixels.append((cx, cy, cw, ch, label))

        if len(gt_boxes_pixels) > 0:
            evaluate_coco_style(boxes, gt_boxes_pixels)
        else:
            print("No ground truth available for evaluation.")

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

    return boxes, gt_boxes_pixels