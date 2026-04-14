from pathlib import Path
import datetime

import numpy as np

from src.helpers.parser import parse_args
from src.data_processing.tools import evaluations, dsp, loaders, viz, vision, dsp_rc
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet


PARAMS = {
    'fs': 1.0,
    'img_height': 512,
    'f_min': 0.005,
    'f_max': 0.5,
    'rc_fc': 1.0,
    'rc_B': 0.12,
    'rc_beta': 0.25,
    'detect_db_range': 28,
    'detect_kernel': (200, 2),
}

def main() -> None:
    args = parse_args()
    input_file = str(args.input)
    meta_file = str(args.meta)

    if meta_file == 'None':
        meta_file = input_file.replace(".sigmf-data", ".sigmf-meta")
        print(f"meta_file = {meta_file}")

    PARAMS['duration'] = args.duration
    PARAMS['offset'] = args.offset
    PARAMS['transform'] = args.transfoType
    PARAMS['wavelet'] = args.waveletType

    output_dir = str(args.output)
    loaders.ensure_dir(output_dir)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    bytes_per_sample = np.dtype(np.complex64).itemsize
    load_offset_bytes = PARAMS['offset'] * bytes_per_sample
    sig = loaders.load_iq_data(input_file, PARAMS['duration'], offset=load_offset_bytes)
    meta = loaders.load_metadata(meta_file)

    if sig is None:
        return

    if PARAMS.get('transform', 'cwt') == 'cwt':
        spec = dsp.compute_dual_linear_cwt(
            sig,
            PARAMS['wavelet'],
            PARAMS['img_height'],
            PARAMS['f_min'],
            PARAMS['f_max'],
            PARAMS['fs'],
        )
    elif PARAMS['transform'] == 'cwt_rc':
        rc = RaisedCosineWavelet(
            fc=PARAMS['rc_fc'],
            B=PARAMS['rc_B'],
            beta=PARAMS['rc_beta'],
        )
        spec = dsp_rc.compute_dual_linear_cwt_rc(
            sig,
            rc,
            PARAMS['img_height'],
            PARAMS['f_min'],
            PARAMS['f_max'],
            PARAMS['fs'],
        )
    else:
        raise ValueError("PARAMS['transform'] doit être 'cwt' ou 'cwt_rc'")

    if args.saveRaw:
        viz.save_spectrogram_image(spec, output_dir, PARAMS, timestamp=run_timestamp)
        return

    boxes = []
    if args.addPrediction:
        boxes, _ = vision.detect_boxes(
            spec,
            min_db_range=PARAMS['detect_db_range'],
            morph_kernel_size=PARAMS['detect_kernel'],
        )
        print(f"-> {len(boxes)} objets détectés.")

        gt_boxes = []
        if meta:
            for ann in meta.get("annotations", []):
                gt_item = evaluations.clip_gt_box(ann, PARAMS)
                if gt_item is not None:
                    gt_boxes.append(gt_item)

        if gt_boxes:
            image_path = Path(viz.build_output_path(output_dir, PARAMS, run_timestamp, extension=".png"))
            json_path = image_path.with_suffix(".json")
            evaluations.evaluate_coco_style(
                boxes,
                gt_boxes,
                params=PARAMS,
                output_json_path=str(json_path),
            )
        else:
            print("Pas de Vérité Terrain disponible pour l'évaluation.")

    viz.save_viz_comparison(spec, meta, boxes, output_dir, PARAMS, timestamp=run_timestamp)

if __name__ == "__main__":
    main()
