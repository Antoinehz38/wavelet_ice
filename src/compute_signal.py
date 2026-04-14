import os, time
import datetime

import cv2

from src.cwt_scheduler import TimeWindow
from src.processing.tools.evaluations import evaluate_coco_style
from src.processing.tools.loaders import load_iq_data
from src.processing.tools.vision import detect_box
from src.processing.tools.viz import save_viz_comparison, compress_spectrogram

from src.processing.transformations.baseline.dsp_stft import compute_stft_scalogram
from src.processing.tools.raised_cosine import RaisedCosineWavelet
from src.processing.transformations.wavelet.dsp_rc import compute_dual_linear_cwt_rc
from src.processing.transformations.wavelet.dsp_dtcwt import compute_dual_dtcwt_scalogram_dyadic
from src.processing.transformations.wavelet.dsp import compute_dual_linear_cwt, freq_to_pixel_linear

from src.processing.transformations.wavelet.bump import BumpWavelet
from src.processing.transformations.wavelet.dsp_bump import compute_dual_linear_cwt_bump

from src.processing.transformations.wavelet.morse import MorseWavelet
from src.processing.transformations.wavelet.dsp_morse import compute_dual_linear_cwt_morse

from src.processing.tools.viz import build_output_dir_path, resolve_wavelet_name


def run_signal_processing_pipeline(input_file: str, meta: dict, output_dir: str, time_window: TimeWindow, params: dict) -> tuple[list, list, int, int]:
    # Absolute bounds of the current time window
    win_start = time_window.start
    win_end   = time_window.start + time_window.length
    sig = load_iq_data(input_file, num_samples=time_window.length, offset=time_window.start)

    if sig is None:
        return [], [], 0, 0

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
        
    elif params['transform'] == 'stft':
        spec = compute_stft_scalogram(
            iq_data=sig,
            total_height=params['img_height'],
            fs=params['fs'],
            f_min=params['f_min'],
            f_max=params['f_max'],
            nperseg=params['stft_nperseg'],
            noverlap=params['stft_noverlap'],
            nfft=params['stft_nfft'],
            window=params['stft_window'],
        )
    
    elif params['transform'] == 'dtcwt':
        spec = compute_dual_dtcwt_scalogram_dyadic(
            iq_data=sig,
            total_height=params['img_height'],
            nlevels=params['dtcwt_levels'],
            biort=params['dtcwt_biort'],
            qshift=params['dtcwt_qshift'],
            top_db_per_band=40.0,
            resize_to_signal_len=params['dtcwt_resize_to_signal_len'],
        )
    
    elif params['transform'] == 'cwt_bump':
        bump = BumpWavelet(
            fc=params['bump_fc'],
            B=params['bump_B']
        )
        spec = compute_dual_linear_cwt_bump(
            sig,
            bump,
            params['img_height'],
            params['f_min'],
            params['f_max'],
            params['fs']
        )

    elif params['transform'] == 'cwt_morse':
        morse = MorseWavelet(
            beta=params['morse_beta'],
            gamma=params['morse_gamma']
        )
        spec = compute_dual_linear_cwt_morse(
            sig,
            morse,
            params['img_height'],
            params['f_min'],
            params['f_max'],
            params['fs']
        )
    
    else:
        raise ValueError("params['transform'] must be 'cwt', 'cwt_rc', 'stft', 'dtcwt', 'cwt_bump', or 'cwt_morse'")


    
    compressed_spec = compress_spectrogram(spec, params['downsample_factor'])

    if params.get('saveRaw', False):
        output_path = build_output_dir_path(output_dir, params)
        wavelet_name = resolve_wavelet_name(params)
        filename = f"raw_{wavelet_name}_start_{time_window.start}_length_{time_window.length}.png"
        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, compressed_spec)

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

    
    output_path = build_output_dir_path(output_dir, params)
    wavelet_name = resolve_wavelet_name(params)
    filename = f"{wavelet_name}_start_{time_window.start}_length_{time_window.length}.png"
    filepath = os.path.join(output_path, filename)

    save_viz_comparison(compressed_spec, gt_boxes_pixels, boxes, filepath, params)

    img_w = compressed_spec.shape[1]
    img_h = compressed_spec.shape[0]
    return boxes, gt_boxes_pixels, img_w, img_h