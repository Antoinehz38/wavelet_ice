#!/usr/bin/env python3
import os, json
import numpy as np

OUT_DIR = "data/baseline"
META_PATH = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"

NFFT = 512
EPS = 1e-12
FRAMES_PER_CHUNK = 4096

os.makedirs(OUT_DIR, exist_ok=True)

def normalize_memmap(m):
    mu = float(m.mean())
    sd = float(m.std())
    m[:] = (m - mu) / (sd + 1e-8)

if __name__ == "__main__":
    base = META_PATH[:-len(".sigmf-meta")]
    data_path = base + ".sigmf-data"

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    if meta["global"].get("core:datatype") != "cf32_le":
        raise ValueError("Ce script suppose core:datatype == cf32_le")

    file_bytes = os.path.getsize(data_path)
    n_complex = file_bytes // np.dtype(np.complex64).itemsize
    n_frames = n_complex // NFFT

    stem = os.path.basename(base) + f"_logspec_n{NFFT}_cf32"
    npy_path = os.path.join(OUT_DIR, stem + ".npy")

    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(n_frames, NFFT)
    )

    win = np.hanning(NFFT).astype(np.float32)

    with open(data_path, "rb") as f:
        t = 0
        while t < n_frames:
            frames = min(FRAMES_PER_CHUNK, n_frames - t)
            n_complex_chunk = frames * NFFT

            raw = np.frombuffer(
                f.read(n_complex_chunk * np.dtype(np.complex64).itemsize),
                dtype=np.complex64
            )
            if raw.size < n_complex_chunk:
                break

            x = raw.reshape(frames, NFFT) * win[None, :]
            X = np.fft.fftshift(np.fft.fft(x, n=NFFT, axis=1), axes=1)

            P = (np.abs(X) ** 2).astype(np.float32)
            L = np.log(P + EPS).astype(np.float32)   # ou 10*np.log10(P+EPS)

            out[t:t+frames, :] = L
            t += frames

    normalize_memmap(out)
    out.flush()
    print(f"saved {npy_path} shape={out.shape}")
