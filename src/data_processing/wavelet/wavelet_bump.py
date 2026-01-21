import os
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "/raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"
OUTPUT_DIR = "/home/cei_ice_2025/guido_leteurtre_hanachowicz/wavelet_ice/data/wavelet_bump"
SAMPLE_RATE = 1.0
DURATION_TO_READ = 200000
BOUNDING_BOX = True
BOUNDING_BOX_COLOR = "black"

# --- PARAMETRES WAVELET BUMP ---
# omega0: frequence centrale (rad/sample), sigma: largeur de bande (rad/sample)
OMEGA0 = 5.0
SIGMA = 2.0
# Si False, on garde les frequences negatives (utile pour signaux IQ)
ANALYTIC = False


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_sigmf_chunk(filepath, num_samples, offset=0):
    try:
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset)
        print(f"✅ Chargé {len(data)} échantillons.")
        return data
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable : {filepath}")
        return None


def get_sigmf_meta_path(data_path):
    if data_path.endswith(".sigmf-data"):
        return data_path[: -len(".sigmf-data")] + ".sigmf-meta"
    return data_path + ".sigmf-meta"


def load_sigmf_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Metadonnees introuvables : {meta_path}")
        return None


def extract_bounding_boxes(metadata, duration, f_min, f_max, time_offset=0):
    annotations = metadata.get("annotations", []) if metadata else []
    boxes = []
    for ann in annotations:
        t_start = ann.get("core:sample_start", 0) - time_offset
        t_end = t_start + ann.get("core:sample_count", 0)
        f_low = ann.get("core:freq_lower_edge", f_min)
        f_high = ann.get("core:freq_upper_edge", f_max)

        t0 = max(0, t_start)
        t1 = min(duration, t_end)
        f0 = max(f_min, f_low)
        f1 = min(f_max, f_high)

        if t1 <= t0 or f1 <= f0:
            continue

        boxes.append(
            {
                "x": float(t0),
                "y": float(f0),
                "w": float(t1 - t0),
                "h": float(f1 - f0),
            }
        )
    return boxes


def draw_bounding_boxes(ax, boxes, color="pink", linewidth=2):
    if not boxes:
        return
    import matplotlib.patches as patches

    for box in boxes:
        rect = patches.Rectangle(
            (box["x"], box["y"]),
            box["w"],
            box["h"],
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)


def bump_hat(omega, omega0=OMEGA0, sigma=SIGMA, analytic=ANALYTIC):
    """Bump wavelet en domaine frequentiel (support compact)."""
    x = (omega - omega0) / sigma
    inside = np.abs(x) < 1.0
    bump = np.zeros_like(omega, dtype=np.float64)
    bump[inside] = np.exp(1.0 - 1.0 / (1.0 - x[inside] ** 2))
    if analytic:
        bump[omega <= 0] = 0.0
    return bump


def get_frequency_range(analytic, fs):
    if analytic:
        return 0.0, fs / 2.0
    return -fs / 2.0, fs / 2.0


def compute_bump_scalogram(iq_data):
    print("⏳ Calcul CWT avec wavelet Bump...")
    n = iq_data.size
    scales = np.arange(2, 128)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / SAMPLE_RATE)
    x_fft = np.fft.fft(iq_data)

    coefs = np.empty((scales.size, n), dtype=np.complex128)
    for idx, scale in enumerate(scales):
        psi_hat = np.sqrt(scale) * bump_hat(scale * omega)
        coefs[idx] = np.fft.ifft(x_fft * np.conj(psi_hat))

    power = np.abs(coefs) ** 2
    power_db = 10 * np.log10(power + 1e-12)
    return power_db


def save_results(spectrogram, output_dir, source_file, wavelet_params, bounding_boxes=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"bump_{timestamp}"

    img_path = os.path.join(output_dir, f"{base_name}.png")
    meta_path = os.path.join(output_dir, f"{base_name}.json")

    scales = np.arange(
        wavelet_params["scales_start"],
        wavelet_params["scales_end"],
        wavelet_params["scales_step"],
    )

    f_min, f_max = get_frequency_range(
        wavelet_params["analytic"], wavelet_params["fs"]
    )

    extent = [0, wavelet_params["duration"], f_min, f_max]

    h, w = spectrogram.shape
    max_width = 5000
    if w > max_width:
        step = w // max_width
        img_to_plot = spectrogram[:, ::step]
    else:
        img_to_plot = spectrogram

    plt.figure(figsize=(16, 10))
    vm = np.max(img_to_plot)

    ax = plt.gca()
    im = ax.imshow(
        img_to_plot,
        extent=extent,
        aspect="auto",
        cmap="inferno",
        origin="upper",
        vmin=vm - 55,
        vmax=vm,
    )

    plt.colorbar(im, label="Puissance (dB)")
    plt.title(f"Scalogramme Bump - {timestamp}")
    plt.ylabel("Frequence Normalisee")
    plt.xlabel("Temps (Echantillons)")

    if not wavelet_params["analytic"]:
        ax.axhline(0, color="white", linestyle="--", alpha=0.5, linewidth=0.8)

    if bounding_boxes:
        draw_bounding_boxes(
            ax, bounding_boxes, color=BOUNDING_BOX_COLOR, linewidth=2.5
        )

    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"💾 Image sauvegardee : {img_path}")

    metadata = {
        "timestamp": timestamp,
        "source_file": os.path.basename(source_file),
        "type": "Bump CWT",
        "processing_parameters": {
            "duration": wavelet_params["duration"],
            "freq_range": [float(f_min), float(f_max)],
            "scales": [int(scales[0]), int(scales[-1]), int(scales[1] - scales[0])],
        },
        "wavelet_config": wavelet_params,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    sig = load_sigmf_chunk(INPUT_FILE, DURATION_TO_READ)
    if sig is not None:
        scalogram = compute_bump_scalogram(sig)
        wavelet_params = {
            "name": "bump",
            "omega0": OMEGA0,
            "sigma": SIGMA,
            "analytic": ANALYTIC,
            "fs": SAMPLE_RATE,
            "duration": DURATION_TO_READ,
            "scales_start": 2,
            "scales_end": 128,
            "scales_step": 1,
        }
        bounding_boxes = None
        if BOUNDING_BOX:
            meta_path = get_sigmf_meta_path(INPUT_FILE)
            metadata = load_sigmf_metadata(meta_path)
            f_min, f_max = get_frequency_range(ANALYTIC, SAMPLE_RATE)
            bounding_boxes = extract_bounding_boxes(
                metadata, DURATION_TO_READ, f_min, f_max
            )

        save_results(
            scalogram,
            OUTPUT_DIR,
            INPUT_FILE,
            wavelet_params,
            bounding_boxes=bounding_boxes,
        )
