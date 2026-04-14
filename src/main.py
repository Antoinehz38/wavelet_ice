import os, datetime
from pathlib import Path
import numpy as np 


from src.compute_signal import run_signal_processing_pipeline
from src.cwt_scheduler import build_transition_windows
from src.helpers.parser import parse_args
from src.processing.tools.loaders import ensure_dir, load_metadata
from src.processing.tools.evaluations import merge_boxes
from src.helpers.display import print_transformation_params


PARAMS = {
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 1_000_000,
    'f_min': 0.005,
    'f_max': 0.5,
    'rc_fc': 1.0,
    'rc_B': 0.12,
    'rc_beta': 0.25,
    'detect_db_range': 28,
    'detect_kernel': (200, 2),
    'downsample_factor': 500,
    'saveRaw': False,
    'addPrediction': False,
    'stft_nperseg': 256,
    'stft_noverlap': 192,
    'stft_nfft': 512,          
    'stft_window': 'hann',    
    'dtcwt_levels': 8,
    'dtcwt_biort': 'near_sym_a',
    'dtcwt_qshift': 'qshift_a',
    'dtcwt_resize_to_signal_len': False,
}

def main()->None:
    args = parse_args()
    input_file = str(args.input)
    meta_file = str(args.meta)

    if meta_file == 'None':
        meta_file = input_file.replace(".sigmf-data", ".sigmf-meta")
        print(f'meta_file = {meta_file}')


    PARAMS['duration'] = args.duration
    PARAMS['offset'] = args.offset
    PARAMS['transform'] = args.transfoType
    PARAMS['wavelet'] = args.waveletType
    PARAMS['downsample_factor'] = args.downSizeFactor
    PARAMS['saveRaw'] = args.saveRaw
    PARAMS['addPrediction'] = args.addPrediction
    PARAMS['points_per_window'] = args.pointsPerWindow
    PARAMS['input_file'] = input_file

    print_transformation_params(PARAMS)
    
    if args.runPipelineOnFolder:
        input_folder = str(args.runPipelineOnFolder)
        output_dir = str(args.output)
        ensure_dir(output_dir)
        for file in os.listdir(input_folder):
            if file.endswith(".sigmf-data"):
                input_file = os.path.join(input_folder, file)
                meta = load_metadata(input_file.replace(".sigmf-data", ".sigmf-meta"))
                annotations = meta.get("annotations", []) if meta else []
                windows = build_transition_windows(
                            annotations=annotations,
                            window_size=PARAMS['points_per_window'],
                            global_start=PARAMS['offset'],
                            global_end=PARAMS['offset'] + PARAMS['duration'],
                        )
    
                for w in windows:
                    boxes, gt_boxes, img_w, img_h = run_signal_processing_pipeline(input_file, meta, output_dir, 
                                                time_window=w, params=PARAMS)
                    

        return None 

    output_dir = str(args.output)
    ensure_dir(output_dir)

    meta = load_metadata(meta_file)

    annotations = meta.get("annotations", []) if meta else []
    windows = build_transition_windows(
        annotations=annotations,
        window_size=PARAMS['points_per_window'],
        global_start=PARAMS['offset'],
        global_end=PARAMS['offset'] + PARAMS['duration'],
    )
    print(f"{len(windows)} CWT windows to compute.")
    total_boxes = []
    for i, w in enumerate(windows):
        print(
            f"[{i}] start={w.start}, end={w.end}, len={w.length}, "
            f"active={w.descriptions}")
        boxes, gt_boxes, img_w, img_h = run_signal_processing_pipeline(input_file, meta, output_dir, 
                                                time_window=w, params=PARAMS)
        total_boxes.append((boxes, w, img_w, img_h))
    
    print(f'total boxes = {total_boxes}')

    # Merge all detected boxes into a single global prediction list
    merged = merge_boxes(total_boxes)
    print(f"\n=== Merged predictions: {len(merged['annotations'])} boxes ===")
    print(merged['annotations'])


if __name__ == "__main__":
    main()

