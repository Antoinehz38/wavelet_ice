import os
from tools import loaders, dsp, vision, viz, evaluations

# --- CONFIGURATION ---
INPUT_FILE = "C:\\Users\\bapti\\Documents\\projet central\\ondelette\\test_wavelet\\data\\sigmf\\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"
META_FILE = "C:\\Users\\bapti\\Documents\\projet central\\ondelette\\test_wavelet\\data\\sigmf\\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta"
OUTPUT_DIR = "C:\\Users\\bapti\\Documents\\projet central\\ondelette\\wavelet_ice\\src\\data_processing\\wavelet\\data\\wavelet_morlet"

PARAMS = {
    'duration': 20_000,
    'fs': 1.0,
    'img_height': 512,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': 'cmor100.0-1.0',
    
    # --- REGLAGES DE DETECTION ---
    # 30 dB : Bon compromis pour capter l'OFDM sans le bruit de fond
    'detect_db_range': 28,       
    
    # (200, 2) : 
    # Largeur 200 -> Soude tout ce qui est fragmenté horizontalement (anti-carrés)
    # Hauteur 2   -> Garde les signaux empilés bien séparés
    'detect_kernel': (200, 2)    
}

def main():
    loaders.ensure_dir(OUTPUT_DIR)
    
    # 1. Chargement
    print("--- 1. Chargement ---")
    sig = loaders.load_iq_data(INPUT_FILE, PARAMS['duration'])
    meta = loaders.load_metadata(META_FILE)
    
    if sig is None: return

    # 2. Traitement Signal
    print("--- 2. DSP ---")
    spec = dsp.compute_dual_linear_cwt(
        sig, PARAMS['wavelet'], PARAMS['img_height'], 
        PARAMS['f_min'], PARAMS['f_max'], PARAMS['fs']
    )
    
    # 3. Détection
    print("--- 3. Détection ---")
    boxes, _ = vision.detect_boxes(
        spec, 
        min_db_range=PARAMS['detect_db_range'], 
        morph_kernel_size=PARAMS['detect_kernel']
    )
    print(f"-> {len(boxes)} objets détectés.")

    # --- 3b. Conversion GT pour Évaluation ---
    gt_boxes_pixels = []
    if meta:
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] < PARAMS['duration']:
                # On réutilise la fonction de conversion de dsp.py
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
        print("⚠️ Pas de Vérité Terrain disponible pour l'évaluation.")


    
    # 4. Visualisation
    print("--- 4. Visualisation ---")
    viz.save_viz_comparison(spec, meta, boxes, OUTPUT_DIR, PARAMS)

if __name__ == "__main__":
    main()