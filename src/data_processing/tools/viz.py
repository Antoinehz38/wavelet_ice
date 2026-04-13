import os
import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import cv2
from .dsp import freq_to_pixel_linear


def _resolve_wavelet_output_dir(output_dir, params):
    if params.get('transform') != 'cwt':
        return output_dir

    wavelet_name = params.get('wavelet', '')
    match = re.match(r"[A-Za-z]+", wavelet_name)
    if not match:
        return output_dir

    wavelet_dir = os.path.join(output_dir, match.group(0))
    os.makedirs(wavelet_dir, exist_ok=True)
    return wavelet_dir

def _resolve_wavelet_name(params):
    if params['transform'] == 'cwt_rc':
        return "Raised Cosine"
    if params['transform'] == 'cwt':
        return params['wavelet']
    return "Wavelet"

def build_output_path(output_dir, params, timestamp, extension=".png"):
    wavelet_name = _resolve_wavelet_name(params)
    filename = f"{wavelet_name}_{timestamp}{extension}"
    target_output_dir = _resolve_wavelet_output_dir(output_dir, params)
    return os.path.join(target_output_dir, filename)

def save_viz_comparison(spectrogram, meta_data, detected_boxes, output_dir, params, timestamp=None):
    """
    Sauvegarde l'image avec axes physiques (Hz) et comparaison BBox.
    """
    wavelet_name = _resolve_wavelet_name(params)
    offset = params.get('offset', 0)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = build_output_path(output_dir, params, timestamp, extension=".png")

    h, w = spectrogram.shape
    f_max = params['f_max']
    duration = params['duration']

    plt.figure(figsize=(16, 10))

    vm = np.max(spectrogram)
    plt.imshow(
        spectrogram,
        aspect='auto',
        cmap='inferno',
        origin='upper',
        vmin=vm - 40,
        vmax=vm,
    )

    ax = plt.gca()

    yticks_pixels = np.linspace(0, h - 1, 11)
    yticks_labels = np.linspace(f_max, -f_max, 11)
    labels_txt = [f"{val:.2f}" for val in yticks_labels]

    plt.yticks(yticks_pixels, labels_txt)
    plt.ylabel("Fréquence (Hz)")
    plt.xlabel("Temps (Échantillons)")

    if meta_data:
        for ann in meta_data.get("annotations", []):
            x_start = ann['core:sample_start'] - offset
            if x_start < 0 or x_start >= duration:
                continue

            y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], h, f_max)
            y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], h, f_max)

            if y_start < 0:
                y_start = 0
            if y_end > h:
                y_end = h

            rect = patches.Rectangle(
                (x_start, y_start),
                min(ann['core:sample_count'], duration - x_start),
                y_end - y_start,
                linewidth=2,
                edgecolor='cyan',
                facecolor='none',
            )
            ax.add_patch(rect)
            plt.text(
                x_start,
                y_start - 5,
                ann.get('core:description', ''),
                color='cyan',
                fontsize=9,
                fontweight='bold',
            )

    if detected_boxes:
        for (x, y, wb, hb) in detected_boxes:
            rect = patches.Rectangle(
                (x, y),
                wb,
                hb,
                linewidth=2,
                edgecolor='#00FF00',
                facecolor='none',
                linestyle='--',
            )
            ax.add_patch(rect)
            plt.text(x + wb, y + hb + 10, "Auto", color='#00FF00', fontsize=8, ha='right')

    plt.title(f"{wavelet_name} - {timestamp}")
    plt.grid(alpha=0.2, linestyle=':', color='white')

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Image sauvegardée : {save_path}")
    return save_path

def save_spectrogram_image(spectrogram, output_dir, params, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = build_output_path(output_dir, params, timestamp, extension=".png")

    vm = np.max(spectrogram)
    h, w = spectrogram.shape

    ds = params.get("downsample_factor", 500)
    out_w_px = int(np.ceil(w / ds))

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

    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    print(f"Spectrogramme sauvegardé : {save_path} ({out_w_px}x{out_h_px}px)")
    return save_path
