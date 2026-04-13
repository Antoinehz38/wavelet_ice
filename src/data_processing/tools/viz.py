import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import cv2
from .dsp import freq_to_pixel_linear

def save_viz_comparison(spectrogram, meta_data, detected_boxes, output_dir, params):
    """
    Sauvegarde l'image compressée (uint8 grayscale) avec axes physiques (Hz / échantillons)
    et comparaison BBox ground-truth (cyan) vs détection auto (vert).

    spectrogram : np.ndarray uint8, image compressée (out_h_px, out_w_px)
    detected_boxes : liste de (x, y, w, h) dans le repère compressé
    """

    if params['transform'] == 'cwt_rc':
        wavelet_name = "Raised Cosine"
    elif params['transform'] == 'cwt':
        wavelet_name = params['wavelet']
    else:
        wavelet_name = "Wavelet"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{wavelet_name}_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)

    h, w = spectrogram.shape
    f_max = params['f_max']
    duration = params['duration']
    ds = params['downsample_factor']
    img_height_orig = params['img_height']
    scale_y = h / img_height_orig

    plt.figure(figsize=(16, 10))

    # --- Affichage image compressée (déjà en uint8 grayscale) ---
    plt.imshow(spectrogram, aspect='auto', cmap='gray', origin='upper',
               vmin=0, vmax=255)

    ax = plt.gca()

    # --- Axe Y : fréquences (Hz) ---
    yticks_pixels = np.linspace(0, h - 1, 11)
    yticks_labels = np.linspace(f_max, -f_max, 11)
    labels_y = [f"{val:.2f}" for val in yticks_labels]
    plt.yticks(yticks_pixels, labels_y)
    plt.ylabel("Fréquence (Hz)")

    # --- Axe X : temps en échantillons originaux ---
    xticks_pixels = np.linspace(0, w - 1, min(11, w))
    xticks_labels = xticks_pixels * ds
    labels_x = [f"{int(val)}" for val in xticks_labels]
    plt.xticks(xticks_pixels, labels_x)
    plt.xlabel("Temps (Échantillons)")

    # --- 1. Dessin Vérité Terrain (Cyan) — converti vers repère compressé ---
    if meta_data:
        for ann in meta_data.get("annotations", []):
            if ann['core:sample_start'] < duration:
                # Conversion Hz -> pixels dans le repère original
                y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], img_height_orig, f_max)
                y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], img_height_orig, f_max)

                if y_start < 0: y_start = 0
                if y_end > img_height_orig: y_end = img_height_orig

                x_orig = ann['core:sample_start']
                w_orig = min(ann['core:sample_count'], duration - x_orig)

                # Conversion vers le repère compressé
                cx = x_orig / ds
                cw = w_orig / ds
                cy = y_start * scale_y
                ch = (y_end - y_start) * scale_y

                rect = patches.Rectangle(
                    (cx, cy), cw, ch,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                ax.add_patch(rect)
                plt.text(cx, cy - 5, ann.get('core:description', ''),
                         color='cyan', fontsize=9, fontweight='bold')

    # --- 2. Dessin Détection Auto (Vert) — déjà dans le repère compressé ---
    if detected_boxes:
        for (x, y, wb, hb) in detected_boxes:
            rect = patches.Rectangle((x, y), wb, hb,
                                     linewidth=2, edgecolor='#00FF00', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            plt.text(x + wb, y + hb + 10, "Auto", color='#00FF00', fontsize=8, ha='right')

    plt.title(f"{wavelet_name} - {timestamp}")
    plt.grid(alpha=0.2, linestyle=':', color='white')

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Image sauvegardée : {save_path}")

def compress_spectrogram(spectrogram, downsample_factor, out_h_px=1500):
    """
    Reproduit en mémoire le résultat de save_spectrogram_image + cv2.imread(..., GRAYSCALE).
    Retourne un np.ndarray uint8 (niveaux de gris) de taille (out_h_px, ceil(w/ds)).
    """
    vm = np.max(spectrogram)
    h, w = spectrogram.shape
    out_w_px = int(np.ceil(w / downsample_factor))

    # 1. Clamp [vm-40, vm] puis normalise dans [0, 1]
    clipped = np.clip(spectrogram, vm - 40, vm)
    normed = (clipped - (vm - 40)) / 40.0

    # 2. Resize à la taille cible (même effet que aspect='auto' de imshow)
    #    cv2.resize attend (width, height)
    resized = cv2.resize(normed.astype(np.float32), (out_w_px, out_h_px),
                         interpolation=cv2.INTER_NEAREST)

    # 3. Appliquer la colormap inferno puis convertir en grayscale
    rgba = cm.inferno(resized)                      # float64 (H, W, 4), valeurs [0,1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)   # drop alpha, passe en uint8
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return gray


def save_spectrogram_image(spectrogram, output_dir, params):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spectrogram_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)

    vm = np.max(spectrogram)
    h, w = spectrogram.shape

    # Temps: 1 pixel = ds échantillons (plus ds est grand, plus c'est compressé)
    ds = params.get("downsample_factor", 500)  # ex: 500
    out_w_px = int(np.ceil(w / ds))

    # Hauteur FIXE (sinon h/w tue l'image)
    out_h_px = params.get("out_h_px", 1500)    

    dpi = params.get("dpi", 150)
    fig_w_in = out_w_px / dpi
    fig_h_in = out_h_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(
        spectrogram,
        cmap="inferno",
        origin="upper",
        aspect="auto",
        vmin=vm - 40,
        vmax=vm,
        interpolation="nearest",
    )
    ax.axis("off")

    # IMPORTANT: pas de bbox_inches='tight' sinon matplotlib recadre et casse la taille
    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    print(f"Spectrogramme sauvegardé : {save_path} ({out_w_px}x{out_h_px}px)")