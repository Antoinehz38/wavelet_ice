#!/usr/bin/env python3
import os, json
import numpy as np

OUT_DIR = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline"
META_PATH = "/home/antoine/Downloads/west-wideband-modrec-ex100-tmpl15-20.04.sigmf-meta"

NFFT = 512
EPS = 1e-12
FRAMES_PER_CHUNK = 4096  # baisse si RAM faible

os.makedirs(OUT_DIR, exist_ok=True)

def normalize_memmap(m):
    mu = float(m.mean())
    sd = float(m.std())
    m[:] = (m - mu) / (sd + 1e-8)

if __name__ == "__main__":
    base = META_PATH[:-len(".sigmf-meta")]
    data_path = base + ".sigmf-data"

    # meta chargé juste pour cohérence (pas indispensable pour la FFT)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    file_bytes = os.path.getsize(data_path)
    n_i16 = file_bytes // 2
    n_complex = n_i16 // 2
    n_frames = n_complex // NFFT

    stem = os.path.basename(base) + f"_logspec_n{NFFT}"
    npy_path = os.path.join(OUT_DIR, stem + ".npy")

    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(n_frames, NFFT)
    )

    with open(data_path, "rb") as f:
        t = 0
        while t < n_frames:
            frames = min(FRAMES_PER_CHUNK, n_frames - t)
            n_complex_chunk = frames * NFFT
            n_i16_chunk = n_complex_chunk * 2  # IQ interleavé

            raw = np.frombuffer(f.read(n_i16_chunk * 2), dtype=np.int16)
            if raw.size < n_i16_chunk:
                break

            iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0
            x = iq.reshape(frames, NFFT)
            X = np.fft.fft(x, n=NFFT, axis=1)              # fidèle papier: pas de fftshift
            L = np.log(np.abs(X) + EPS).astype(np.float32) # log(|FFT|)

            out[t:t+frames, :] = L
            t += frames

    normalize_memmap(out)
    out.flush()

    print(f"saved {npy_path} shape={out.shape}")
