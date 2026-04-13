import os
import cv2
import datetime

from src.cwt_scheduler import build_cwt_windows_from_annotations
from src.helpers.parser import parse_args
from src.data_processing.tools import evaluations, dsp, loaders, viz, vision, dsp_rc
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet
from src.compute_signal import run_signal_processing_pipeline


PARAMS = {
    'offset': 0,
    'duration': 20_000,
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 100_000,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': "cmor100.0-1.0" , #"cmor100.0-1.0"  'fbsp10-0.01-2'

    'transform': 'cwt',      # cwt_rc ou cwt

    'rc_fc': 1.0,            # exemple (dans [0, 0.5] si fs=1)
    'rc_B': 0.12,            # bande utile
    'rc_beta': 0.25,         # roll-off
    

    'detect_db_range': 28, # réglage détection

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
    
    if args.runPipelineOnFolder:
        input_folder = str(args.runPipelineOnFolder)
        output_dir = str(args.output)
        loaders.ensure_dir(output_dir)
        for file in os.listdir(input_folder):
            if file.endswith(".sigmf-data"):
                input_file = os.path.join(input_folder, file)
                meta= loaders.load_metadata(input_file.replace(".sigmf-data", ".sigmf-meta"))
                windows = build_cwt_windows_from_annotations(
                            annotations=annotations,
                            points_per_window=PARAMS['points_per_window'],
                            global_start=PARAMS['offset'],
                            global_end=PARAMS['offset'] + PARAMS['duration'],
                        )
    
                for w in windows:
                    run_signal_processing_pipeline(input_file, meta, output_dir, 
                                                time_window=w, params=PARAMS)

        return None 

    output_dir = str(args.output)
    loaders.ensure_dir(output_dir)

    meta = loaders.load_metadata(meta_file)

    annotations = meta.get("annotations", []) if meta else []
    windows = build_cwt_windows_from_annotations(
        annotations=annotations,
        points_per_window=PARAMS['points_per_window'],
        global_start=PARAMS['offset'],
        global_end=PARAMS['offset'] + PARAMS['duration'],
    )
    print(f"{len(windows)} fenêtres CWT à calculer.")
    for i, w in enumerate(windows):
        print(
            f"[{i}] start={w.start}, end={w.end}, len={w.length}, "
            f"active={w.descriptions}")
        
        run_signal_processing_pipeline(input_file, meta, output_dir, 
                                       time_window=w, params=PARAMS)
if __name__ == "__main__":
    main()

