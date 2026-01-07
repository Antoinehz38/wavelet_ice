import numpy as np
from PIL import Image

arr = np.load("data/baseline/west-wideband-modrec-ex100-tmpl15-20.04_logspec_n512.npy", mmap_mode="r").astype(np.float32)

print(arr.min(), arr.mean(), arr.max(), arr.std())
# clipping pour le contraste
lo, hi = np.percentile(arr, [2, 98])
arr = np.clip(arr, lo, hi)

# normalisation 0–255
arr = (arr - lo) / (hi - lo + 1e-8)
arr = (arr * 255).astype(np.uint8)

# transpose pour affichage
Image.fromarray(arr.T, mode="L").save("data/baseline/west-wideband-modrec-ex100-tmpl15-20.04_logspec_n512.png")
