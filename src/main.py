from src.helpers.parser import parse_args
from src.data_processing.tools import evaluations, dsp, loaders, viz, vision


PARAMS = {
    'duration': 200_000,
    'fs': 1.0,
    'img_height': 512,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': 'cmor100.0-1.0',
    

    'detect_db_range': 28, # réglage détection
    
    # (200, 2) : 
    # Largeur 200 -> Soude tout ce qui est fragmenté horizontalement (anti-carrés)
    # Hauteur 2   -> Garde les signaux empilés bien séparés
    'detect_kernel': (200, 2)    
}

def main():
    args = parse_args()
    input_file = str(args.input)
    meta_file = str(args.meta)

    if meta_file == 'None':
        meta_file = input_file.replace(".sigmf-data", ".sigmf-meta")
        print(f'meta_file = {meta_file}')

    output_dir = str(args.output)
    loaders.ensure_dir(output_dir)

    sig = loaders.load_iq_data(input_file, PARAMS['duration'])
    meta = loaders.load_metadata(meta_file)
    
    if sig is None: return

    spec = dsp.compute_dual_linear_cwt(
        sig, PARAMS['wavelet'], PARAMS['img_height'], 
        PARAMS['f_min'], PARAMS['f_max'], PARAMS['fs']
    )

    boxes, _ = vision.detect_boxes(
        spec, 
        min_db_range=PARAMS['detect_db_range'], 
        morph_kernel_size=PARAMS['detect_kernel']
    )
    print(f"-> {len(boxes)} objets détectés.")


    gt_boxes_pixels = []
    if meta:
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] < PARAMS['duration']:

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

if __name__ == "__main__":
    main()

