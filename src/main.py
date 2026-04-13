from src.helpers.parser import parse_args
from src.data_processing.tools import evaluations, dsp, loaders, viz, vision, dsp_rc
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet
import numpy as np

PARAMS = {
    'offset': 606028,  # askip offset n’est pas en échantillons, il est en octets. (C’est à cause de loaders.py (line 9)). Dans numpy.fromfile, offset est un décalage en bytes. Or un échantillon complex64 vaut 8 octets. Donc :  offset_bytes = sample_start * 8
    # Autre point important : avec offset != 0, les annotations sont mal recalées dans main.py (line 89) et viz.py (line 56), parce que le code utilise encore ann['core:sample_start'] sans soustraire l’offset local. 
    # Donc : le signal affiché sera le bon ;
    # mais les boîtes GT risquent d’être absentes ou au mauvais endroit.

    'duration': 2_000_000,
    'fs': 1.0, # fréquence d'échantillonnage - nb d'echantillons par seconde (pour la conversion temps <-> échantillons) fs = 1.0 : on est en unités normalisées, pas en secondes physiques utiles. La conversion est : temps (s) = nombre_d'échantillons
    'img_height': 512,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': "cmor100.0-1.0" , #"cmor100.0-1.0"  'fbsp10-0.01-2'

    'transform': 'cwt_rc',      # cwt_rc ou cwt

    'rc_fc': 1.0,            # exemple (dans [0, 0.5] si fs=1)
    'rc_B': 0.12,            # bande utile
    'rc_beta': 0.25,         # roll-off
    

    'detect_db_range': 28, # réglage détection

    'detect_kernel': (200, 2)    
}

def main()->None:
    args = parse_args()
    input_file = str(args.input)
    meta_file = str(args.meta)

    if meta_file == 'None':
        meta_file = input_file.replace(".sigmf-data", ".sigmf-meta")
        print(f'meta_file = {meta_file}')

    if args.duration is not None:
        PARAMS['duration'] = args.duration

    if args.offset is not None:
        PARAMS['offset'] = args.offset

    if args.transfoType:
        PARAMS['transform'] = args.transfoType

    if args.waveletType:
        PARAMS['wavelet'] = args.waveletType

    output_dir = str(args.output)
    loaders.ensure_dir(output_dir)

    bytes_per_sample = np.dtype(np.complex64).itemsize
    load_offset_bytes = PARAMS['offset'] * bytes_per_sample
    sig = loaders.load_iq_data(input_file, PARAMS['duration'], offset=load_offset_bytes)
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

    boxes = []
    if args.addPrediction:
        boxes, _ = vision.detect_boxes(spec,
                                       min_db_range=PARAMS['detect_db_range'],
                                       morph_kernel_size=PARAMS['detect_kernel']
                                       )
        print(f"-> {len(boxes)} objets détectés.")
        gt_boxes_pixels = []
        if meta:
            for ann in meta.get("annotations", []):
                x = ann['core:sample_start'] - PARAMS['offset']
                if x < 0 or x >= PARAMS['duration']: 
                    continue  # Cette annotation est en dehors de la plage chargée, on l'ignore
                y_start = dsp.freq_to_pixel_linear(ann['core:freq_upper_edge'], PARAMS['img_height'], PARAMS['f_max'])
                y_end = dsp.freq_to_pixel_linear(ann['core:freq_lower_edge'], PARAMS['img_height'], PARAMS['f_max'])

                # Sécuité bornes
                if y_start < 0: y_start = 0
                if y_end > PARAMS['img_height']: y_end = PARAMS['img_height']

                x = ann['core:sample_start']
                w = min(ann['core:sample_count'], PARAMS['duration'] - x)
                h = y_end - y_start

                # Format (x, y, w, h)
                gt_boxes_pixels.append((x, y_start, w, h))

        # --- 3c. Lancement Évaluation ---
        if len(gt_boxes_pixels) > 0:
            evaluations.evaluate_coco_style(boxes, gt_boxes_pixels)
        else:
            print("Pas de Vérité Terrain disponible pour l'évaluation.")



    viz.save_viz_comparison(spec, meta, boxes, output_dir, PARAMS)

    return None

if __name__ == "__main__":
    main()

