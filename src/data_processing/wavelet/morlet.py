#!/usr/bin/env python3
import os, json, math, gc
import numpy as np
import pywt

OUT_DIR = "data/wavelet"
META_PATH = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"

WAVELET = "cmor1.5-1.0"
NFREQ_POS = 256
FMIN = 1e-4
FMAX = 0.5
LOG_POWER = True
EPS = 1e-12

# ---- limiter au début: 4096 frames FFT ----
NFFT = 512
MAX_FRAMES = 4096
NSAMP_LIMIT = NFFT * MAX_FRAMES   # <<< 2,097,152 samples

CHUNK_SAMPLES = 250_000
STRIDE = 256

OVERLAP_FACTOR = 12
MIN_OVERLAP = 4096

os.makedirs(OUT_DIR, exist_ok=True)

def read_cf32_le_all(path):
    return np.memmap(path, dtype=np.complex64, mode="r")

def compute_scales(freqs_pos, wavelet, fs):
    dt = 1.0 / fs
    w = pywt.ContinuousWavelet(wavelet)
    f_hz = freqs_pos * fs
    scales = pywt.frequency2scale(w, f_hz * dt).astype(np.float64)
    return scales, w, dt

def first_multiple_in_range(a, b, m):
    return ((a + m - 1) // m) * m

if __name__ == "__main__":
    base = META_PATH[:-len(".sigmf-meta")]
    data_path = base + ".sigmf-data"

    with open(META_PATH, "r") as f:
        meta = json.load(f)
    if meta["global"].get("core:datatype") != "cf32_le":
        raise ValueError("Ce script suppose core:datatype == cf32_le")

    fs = float(meta["global"].get("core:sample_rate", 1.0))

    x = read_cf32_le_all(data_path)
    n_file = int(x.shape[0])
    n = min(n_file, NSAMP_LIMIT)   # <<< LIMIT HERE

    freqs_pos = np.linspace(FMIN, FMAX, NFREQ_POS).astype(np.float64)
    scales, w, dt = compute_scales(freqs_pos, WAVELET, fs)

    max_scale = float(np.max(scales))
    overlap = max(int(OVERLAP_FACTOR * max_scale), MIN_OVERLAP)

    n_out = int(math.ceil(n / STRIDE))
    n_freq_total = 2 * NFREQ_POS

    stem = os.path.basename(base) + f"_cwtmorlet_{WAVELET}_nf{NFREQ_POS}_stride{STRIDE}_ns{n}"
    npy_path = os.path.join(OUT_DIR, stem + ".npy")
    freq_path = os.path.join(OUT_DIR, stem + "_freq.npy")

    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(n_freq_total, n_out)
    )
    freq_axis = np.concatenate([-freqs_pos[::-1], freqs_pos]).astype(np.float32)
    np.save(freq_path, freq_axis)

    i = 0
    while i < n:
        i0 = max(0, i - overlap)
        i1 = min(n, i + CHUNK_SAMPLES + overlap)   # <<< clamp to n

        chunk = np.asarray(x[i0:i1]).astype(np.complex64, copy=False)

        coef_pos, _ = pywt.cwt(chunk, scales, w, sampling_period=dt)
        ppos = (coef_pos.real * coef_pos.real + coef_pos.imag * coef_pos.imag).astype(np.float32)
        del coef_pos
        gc.collect()

        coef_neg, _ = pywt.cwt(np.conj(chunk), scales, w, sampling_period=dt)
        pneg = (coef_neg.real * coef_neg.real + coef_neg.imag * coef_neg.imag).astype(np.float32)
        del coef_neg, chunk
        gc.collect()

        p = np.vstack([pneg[::-1, :], ppos]).astype(np.float32)
        del pneg, ppos
        gc.collect()

        if LOG_POWER:
            p = np.log(p + EPS).astype(np.float32)

        valid_start = 0 if i0 == 0 else overlap
        valid_end = p.shape[1] if i1 == n else (p.shape[1] - overlap)
        if valid_end <= valid_start:
            del p
            i += CHUNK_SAMPLES
            continue

        p_valid = p[:, valid_start:valid_end]
        del p
        gc.collect()

        global_start = i0 + valid_start
        global_end = i0 + valid_end

        first = first_multiple_in_range(global_start, global_end, STRIDE)
        if first < global_end:
            idx = np.arange(first, global_end, STRIDE, dtype=np.int64)
            local = idx - global_start
            cols = (idx // STRIDE).astype(np.int64)
            out[:, cols] = p_valid[:, local]

        del p_valid
        gc.collect()

        i += CHUNK_SAMPLES

    out.flush()
    print(npy_path)
    print(freq_path)
