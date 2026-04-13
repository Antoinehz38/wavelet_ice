from src.helpers.parser import parse_args
from src.data_processing.tools import evaluations, dsp, loaders, viz, vision, dsp_rc
from src.data_processing.tools.test import detect_signals_by_projections, tighten_box_2d,  tighten_box_with_energy
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet


PARAMS = {
    'offset': 0,
    'duration': 20_000,
    'fs': 1.0,
    'img_height': 512,
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

    output_dir = str(args.output)
    loaders.ensure_dir(output_dir)

    sig = loaders.load_iq_data(input_file, PARAMS['duration'], offset=PARAMS['offset'])
    meta = loaders.load_metadata(meta_file)
    
    if sig is None: return

    if PARAMS.get('transform', 'cwt') == 'cwt':
        spec = dsp.compute_dual_linear_cwt(
            sig, PARAMS['wavelet'], PARAMS['img_height'],
            PARAMS['f_min'], PARAMS['f_max'], PARAMS['fs']
        )

    elif PARAMS['transform'] == 'cwt_rc':
        rc = RaisedCosineWavelet(
            fc=PARAMS['rc_fc'],
            B=PARAMS['rc_B'],
            beta=PARAMS['rc_beta']
        )
        spec = dsp_rc.compute_dual_linear_cwt_rc(
            sig, rc, PARAMS['img_height'],
            PARAMS['f_min'], PARAMS['f_max'], PARAMS['fs']
        )
    else:
        raise ValueError("PARAMS['transform'] doit être 'cwt' ou 'cwt_rc'")



    if args.saveRaw:
        viz.save_spectrogram_image(spec, output_dir, PARAMS)
        return None
    
    compressed_spec = viz.compress_spectrogram(spec, PARAMS['downsample_factor'])

    boxes = []
    if args.addPrediction:
        print("adding prediction ... ")
       
        boxes, debug = detect_signals_by_projections(compressed_spec)
        z = debug["z"]
        boxes = [
            tighten_box_2d(
                z,
                box,
            )
            for box in boxes
            ]
        print(f"-> {len(boxes)} objets détectés.")

        # Facteurs de conversion spectrogramme original -> image compressée
        img_h = compressed_spec.shape[0]  # out_h_px (1500 par défaut)
        ds = PARAMS['downsample_factor']
        scale_y = img_h / PARAMS['img_height']

        gt_boxes_pixels = []
        if meta:
            for ann in meta.get("annotations", []):
                if ann['core:sample_start'] < PARAMS['duration']:

                    y_start = dsp.freq_to_pixel_linear(ann['core:freq_upper_edge'], PARAMS['img_height'], PARAMS['f_max'])
                    y_end = dsp.freq_to_pixel_linear(ann['core:freq_lower_edge'], PARAMS['img_height'], PARAMS['f_max'])

                    # Sécurité bornes (dans le repère original)
                    if y_start < 0: y_start = 0
                    if y_end > PARAMS['img_height']: y_end = PARAMS['img_height']

                    x = ann['core:sample_start']
                    w = min(ann['core:sample_count'], PARAMS['duration'] - x)
                    h = y_end - y_start

                    # Conversion vers le repère de l'image compressée
                    cx = x / ds
                    cw = w / ds
                    cy = y_start * scale_y
                    ch = h * scale_y

                    gt_boxes_pixels.append((cx, cy, cw, ch))

        # --- 3c. Lancement Évaluation ---
        if len(gt_boxes_pixels) > 0:
            evaluations.evaluate_coco_style(boxes, gt_boxes_pixels)
        else:
            print("Pas de Vérité Terrain disponible pour l'évaluation.")



    viz.save_viz_comparison(spec, meta, boxes, output_dir, PARAMS)

    return None

if __name__ == "__main__":
    main()

