import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .dsp import freq_to_pixel_linear

def save_viz_comparison(spectrogram, meta_data, detected_boxes, output_dir, params):
    """
    Sauvegarde l'image avec axes physiques (Hz) et comparaison BBox.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"viz_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    
    h, w = spectrogram.shape
    f_max = params['f_max']
    duration = params['duration']
    
    plt.figure(figsize=(16, 10))
    
    # --- Affichage Spectro ---
    vm = np.max(spectrogram)
    # Note : On n'utilise pas 'extent' ici pour garder la logique "pixel" pour les bbox,
    # on change juste les étiquettes (ticks) après.
    plt.imshow(spectrogram, aspect='auto', cmap='inferno', origin='upper',
               vmin=vm-40, vmax=vm) 
    
    ax = plt.gca()

    # --- Gestion des Axes (Le Fix) ---
    # On crée 11 points de repère sur l'axe Y (pixels)
    yticks_pixels = np.linspace(0, h-1, 11)
    # On calcule les valeurs Hz correspondantes : de +F_max à -F_max
    yticks_labels = np.linspace(f_max, -f_max, 11)
    # On formate pour n'avoir que 2 décimales
    labels_txt = [f"{val:.2f}" for val in yticks_labels]
    
    plt.yticks(yticks_pixels, labels_txt)
    plt.ylabel("Fréquence (Hz)")
    plt.xlabel("Temps (Échantillons)")

    # --- 1. Dessin Vérité Terrain (Cyan) ---
    if meta_data:
        for ann in meta_data.get("annotations", []):
            if ann['core:sample_start'] < duration:
                # Conversion Hz -> Pixels
                y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], h, f_max)
                y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], h, f_max)
                
                # Vérification pour éviter les crashs si hors image
                if y_start < 0: y_start = 0
                if y_end > h: y_end = h

                rect = patches.Rectangle(
                    (ann['core:sample_start'], y_start),
                    min(ann['core:sample_count'], duration - ann['core:sample_start']),
                    y_end - y_start,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                ax.add_patch(rect)
                plt.text(ann['core:sample_start'], y_start-5, ann.get('core:description',''), 
                         color='cyan', fontsize=9, fontweight='bold')

    # --- 2. Dessin Détection Auto (Vert) ---
    if detected_boxes:
        for (x, y, wb, hb) in detected_boxes:
            rect = patches.Rectangle((x, y), wb, hb, 
                                     linewidth=2, edgecolor='#00FF00', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            # Label discret
            plt.text(x+wb, y+hb+10, "Auto", color='#00FF00', fontsize=8, ha='right')

    title_suffix = " (Axe Hz Corrigé)"
    plt.title(f"Comparaison GT vs Auto - {timestamp} {title_suffix}")
    plt.grid(alpha=0.2, linestyle=':', color='white')
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"💾 Image sauvegardée : {save_path}")