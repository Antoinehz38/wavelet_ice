#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- params ----
META_PATH = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
NFFT = 512
MAX_FRAMES = 4096
DRAW_META = False   # <<< active / désactive les bounding boxes + labels

# ---- helpers ----
def freq_to_bin(f_norm, nfft):
    return (f_norm + 0.5) * nfft

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

if __name__ == "__main__":
    base = META_PATH[:-len(".sigmf-meta")]
    npy_path = os.path.join(
        "data/baseline",
        os.path.basename(base) + f"_logspec_n{NFFT}_cf32.npy"
    )
    out_png = os.path.join(
        "data/baseline",
        os.path.basename(base) + f"_spec_{MAX_FRAMES}f_META_{DRAW_META}.png"
    )

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    spec = np.load(npy_path, mmap_mode="r")  # (n_frames, NFFT)
    T = min(MAX_FRAMES, spec.shape[0])

    img = spec[:T, :].T  # (freq, time)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest")

    if DRAW_META:
        for a in meta.get("annotations", []):
            s0 = int(a["core:sample_start"])
            sc = int(a["core:sample_count"])
            label = str(a.get("core:description", ""))

            f_lo = float(a.get("core:freq_lower_edge", -0.5))
            f_hi = float(a.get("core:freq_upper_edge",  0.5))

            t0 = s0 // NFFT
            t1 = (s0 + sc) // NFFT

            if t1 <= 0 or t0 >= T:
                continue

            t0c = clamp(t0, 0, T - 1)
            t1c = clamp(t1, 0, T)

            y0 = freq_to_bin(f_lo, NFFT)
            y1 = freq_to_bin(f_hi, NFFT)
            y0c = clamp(min(y0, y1), 0, NFFT - 1)
            y1c = clamp(max(y0, y1), 0, NFFT)

            rect = Rectangle(
                (t0c, y0c),
                max(1, t1c - t0c),
                max(1, y1c - y0c),
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)

            ax.text(
                t0c + 3, y1c - 6, label,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.7)
            )

    ax.set_title(f"Spectrogram (NFFT={NFFT}, first {T} frames)")
    ax.set_xlabel("Time (FFT frame index)")
    ax.set_ylabel("Frequency bin (fftshifted)")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, NFFT - 1)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

    print(out_png)
