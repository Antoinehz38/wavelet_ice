import numpy as np
from PIL import Image

# Configuration
NPY_PATH = "data/baseline/west-wideband-modrec-ex100-tmpl15-20.04_logspec_n512.npy"
OUTPUT_DIR = "data/baseline"
MAX_WIDTH = 4096  # Largeur max de l'image (temps)
MAX_HEIGHT = 512  # Hauteur max de l'image (fréquence)

arr = np.load(NPY_PATH, mmap_mode="r").astype(np.float32)
print(f"Original shape: {arr.shape}")
print(f"Stats: min={arr.min():.2f}, mean={arr.mean():.2f}, max={arr.max():.2f}, std={arr.std():.2f}")

# Crop (prendre les premières frames/fréquences)
n_frames, n_freq = arr.shape
crop_time = min(n_frames, MAX_WIDTH)
crop_freq = min(n_freq, MAX_HEIGHT)

arr = arr[:crop_time, :crop_freq]
print(f"Cropped shape: {arr.shape}")

# Clipping pour le contraste
lo, hi = np.percentile(arr, [2, 98])
arr = np.clip(arr, lo, hi)

# Normalisation 0–255
arr = (arr - lo) / (hi - lo + 1e-8)
arr = (arr * 255).astype(np.uint8)

# Transpose pour affichage (fréquence en Y, temps en X)
img = Image.fromarray(arr.T, mode="L")
output_path = f"{OUTPUT_DIR}/spectrogram_{arr.shape[0]}x{arr.shape[1]}.png"
img.save(output_path)
print(f"Saved: {output_path} (size: {img.size})")
