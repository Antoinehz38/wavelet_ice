#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------- params ---------
META_PATH = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"

CWT_DIR = "data/wavelet"
CWT_STEM = "west-wideband-modrec-ex110-tmpl13-20.04_cwtmorlet_cmor1.5-1.0_nf256_stride256_ns2097152"

DRAW_META = True
MAX_COLS = 4096
DPI = 150

# pour avoir le même axe temps que la FFT (index de frame FFT)
NFFT_FRAME = 512  # même NFFT que ta FFT baseline

# --------- helpers ---------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def intersect_1d(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return lo, hi

if __name__ == "__main__":
    cwt_path  = os.path.join(CWT_DIR, CWT_STEM + ".npy")
    freq_path = os.path.join(CWT_DIR, CWT_STEM + "_freq.npy")
    out_png   = os.path.join(CWT_DIR, CWT_STEM + f"_plot_{MAX_COLS}cols_frames.png")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    cwt = np.load(cwt_path, mmap_mode="r")   # (F, Tcols)
    freq_axis = np.load(freq_path)           # (F,)
    F, Tcols = cwt.shape

    # stride depuis le nom "*_stride###"
    stride = None
    for p in CWT_STEM.split("_"):
        if p.startswith("stride"):
            stride = int(p.replace("stride", ""))
            break
    if stride is None:
        raise ValueError("Impossible d'inférer STRIDE depuis CWT_STEM (attendu '*_stride###').")

    # crop affichage
    Tshow = min(MAX_COLS, Tcols)
    img = cwt[:, :Tshow]  # (F, Tshow)

    # ----- AXE TEMPS EN "FFT FRAME INDEX" -----
    # col -> sample = col*stride  -> frame = sample / NFFT_FRAME
    x0 = 0.0
    x1 = ((Tshow - 1) * stride) / float(NFFT_FRAME)  # frames (float)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[x0, x1, float(freq_axis[0]), float(freq_axis[-1])]
    )

    if DRAW_META:
        anns = meta.get("annotations", [])

        # fenêtre affichée en frames (pas en samples)
        crop_frames = (0.0, (Tshow * stride) / float(NFFT_FRAME))

        for a in anns:
            s0 = int(a["core:sample_start"])
            sc = int(a["core:sample_count"])
            s1 = s0 + sc
            label = str(a.get("core:description", ""))

            f_lo = float(a.get("core:freq_lower_edge", freq_axis[0]))
            f_hi = float(a.get("core:freq_upper_edge", freq_axis[-1]))

            # samples -> frames
            t0 = s0 / float(NFFT_FRAME)
            t1 = s1 / float(NFFT_FRAME)

            inter = intersect_1d(t0, t1, crop_frames[0], crop_frames[1])
            if inter is None:
                continue
            it0, it1 = inter

            flo = clamp(min(f_lo, f_hi), float(freq_axis[0]), float(freq_axis[-1]))
            fhi = clamp(max(f_lo, f_hi), float(freq_axis[0]), float(freq_axis[-1]))

            rect = Rectangle(
                (it0, flo),
                max(1e-9, it1 - it0),
                max(1e-9, fhi - flo),
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)

            ax.text(
                it0 + 0.01 * (it1 - it0 + 1e-9),
                fhi,
                label,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.7)
            )

    ax.set_title(f"CWT Morlet (stride={stride}) - first {Tshow} cols (time in FFT frame index)")
    ax.set_xlabel("Time (FFT frame index)")
    ax.set_ylabel("Frequency (cycles/sample)")
    ax.set_xlim(0, x1)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

    print(out_png)
