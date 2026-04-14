import os

from src.compute_signal import run_signal_processing_pipeline
from src.cwt_scheduler import build_transition_windows
from src.helpers.parser import parse_args
from src.processing.tools.loaders import ensure_dir, load_metadata


PARAMS = {
    'offset': 0,
    'duration': 20_000,
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 1_000_000,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': "cmor100.0-1.0",
    'transform': 'cwt',
    'rc_fc': 1.0,
    'rc_B': 0.12,
    'rc_beta': 0.25,
    'detect_db_range': 28,
    'detect_kernel': (200, 2),
    'downsample_factor': 500,
    'saveRaw': False,
    'addPrediction': False,
}

def main()->None:
    args = parse_args()
    input_file = str(args.input)
    meta_file = str(args.meta)

    if meta_file == 'None':
        meta_file = input_file.replace(".sigmf-data", ".sigmf-meta")
        print(f'meta_file = {meta_file}')

    if args.duration:
        PARAMS['duration'] = args.duration

    if args.offset:
        PARAMS['offset'] = args.offset

    if args.transfoType:
        PARAMS['transform'] = args.transfoType

    if args.waveletType:
        PARAMS['wavelet'] = args.waveletType
    
    if args.downSizeFactor:
        PARAMS['downsample_factor'] = args.downSizeFactor
    
    if args.saveRaw:
        PARAMS['saveRaw'] = True

    if args.addPrediction:
        PARAMS['addPrediction'] = True
    
    if args.pointsPerWindow:
        PARAMS['points_per_window'] = args.pointsPerWindow
    
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
                    run_signal_processing_pipeline(input_file, meta, output_dir, 
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
    for i, w in enumerate(windows):
        print(
            f"[{i}] start={w.start}, end={w.end}, len={w.length}, "
            f"active={w.descriptions}")
        run_signal_processing_pipeline(input_file, meta, output_dir,
                                       time_window=w, params=PARAMS)

if __name__ == "__main__":
    main()

