import os, re
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import cv2

from src.processing.tools.dsp import freq_to_pixel_linear


def save_viz_comparison(spectrogram, gt_boxes_pixels, detected_boxes, filepath, params):
    """Save the compressed image (uint8 grayscale) with physical axes (Hz / samples)
    and a comparison of ground-truth bboxes (cyan) vs auto-detected bboxes (green).

    Args:
        spectrogram: uint8 ndarray, compressed image (out_h_px, out_w_px).
        gt_boxes_pixels: list of (x, y, w, h[, label]) ground truth in compressed coords.
        detected_boxes: list of (x, y, w, h) detections in compressed coords.
        filepath: Full output path.
        params: Dict containing f_max, downsample_factor, etc.
    """
    if params["transform"] == "cwt_rc":
        wavelet_name = "Raised Cosine"
    elif params["transform"] == "cwt":
        wavelet_name = params["wavelet"]
    else:
        wavelet_name = "Wavelet"

    h, w = spectrogram.shape
    f_max = params["f_max"]
    ds = params["downsample_factor"]

    plt.figure(figsize=(16, 10))

    plt.imshow(spectrogram, aspect="auto", cmap="gray", origin="upper", vmin=0, vmax=255)

    ax = plt.gca()

    yticks_pixels = np.linspace(0, h - 1, 11)
    yticks_labels = np.linspace(f_max, -f_max, 11)
    labels_y = [f"{val:.2f}" for val in yticks_labels]
    plt.yticks(yticks_pixels, labels_y)
    plt.ylabel("Frequency (Hz)")

    xticks_pixels = np.linspace(0, w - 1, min(11, w))
    xticks_labels = xticks_pixels * ds
    labels_x = [f"{int(val)}" for val in xticks_labels]
    plt.xticks(xticks_pixels, labels_x)
    plt.xlabel("Time (Samples)")

    if gt_boxes_pixels:
        for box in gt_boxes_pixels:
            if len(box) == 5:
                cx, cy, cw, ch, label = box
            else:
                cx, cy, cw, ch = box
                label = "GT"
            rect = patches.Rectangle(
                (cx, cy), cw, ch,
                linewidth=2, edgecolor="cyan", facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(cx, cy - 5, label, color="cyan", fontsize=9, fontweight="bold")

    if detected_boxes:
        for (x, y, wb, hb) in detected_boxes:
            rect = patches.Rectangle(
                (x, y), wb, hb,
                linewidth=2, edgecolor="#00FF00", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            plt.text(x + wb, y + hb + 10, "Auto", color="#00FF00", fontsize=8, ha="right")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.title(f"{wavelet_name} - {timestamp}")
    plt.grid(alpha=0.2, linestyle=":", color="white")

    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Image saved: {filepath}")


def compress_spectrogram(spectrogram, downsample_factor, out_h_px=1500):
    """Compress a spectrogram to a uint8 grayscale image.

    Reproduces in-memory the result of save_spectrogram_image + cv2.imread(GRAYSCALE).
    Returns a uint8 ndarray of shape (out_h_px, ceil(w/ds)).
    """
    vm = np.max(spectrogram)
    h, w = spectrogram.shape
    out_w_px = int(np.ceil(w / downsample_factor))

    clipped = np.clip(spectrogram, vm - 40, vm)
    normed = (clipped - (vm - 40)) / 40.0

    resized = cv2.resize(
        normed.astype(np.float32), (out_w_px, out_h_px),
        interpolation=cv2.INTER_NEAREST,
    )

    rgba = cm.inferno(resized)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return gray


def save_spectrogram_image(spectrogram, output_dir, params):
    """Save spectrogram as a raw PNG image."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spectrogram_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)

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

    print(f"Spectrogram saved: {save_path} ({out_w_px}x{out_h_px}px)")


def resolve_wavelet_name(params):
    if params['transform'] == 'cwt_rc':
        return "Raised_Cosine"
    if params['transform'] == 'cwt':
        return params['wavelet']
    return "Wavelet"


def _resolve_example_output_dir(output_dir, params):
    input_file = params.get('input_file', '')
    input_name = os.path.basename(input_file)
    match = re.search(r"(ex\d+)", input_name)

    if not match:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    example_dir = os.path.join(output_dir, match.group(1))
    os.makedirs(example_dir, exist_ok=True)
    return example_dir


def build_output_dir_path(output_dir, params):
    target_output_dir = _resolve_example_output_dir(output_dir, params)
    return target_output_dir
