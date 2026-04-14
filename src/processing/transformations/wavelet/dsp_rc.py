import numpy as np

from src.processing.tools.raised_cosine import RaisedCosineWavelet


def generate_linear_scales_rc(f_min, f_max, num_pixels, wavelet: RaisedCosineWavelet, fs=1.0):
    """Generate scales for a linear frequency axis using raised-cosine wavelet."""
    freqs_linear = np.linspace(f_max, f_min, num_pixels)
    fc = wavelet.central_frequency
    scales = (fc * fs) / freqs_linear
    return scales

def cwt_fft(x: np.ndarray, scales: np.ndarray, wavelet: RaisedCosineWavelet, fs=1.0):
    x = np.asarray(x)
    N = x.shape[0]

    # --- 1. ZERO PADDING ---
    pad = N // 2  # simple et efficace
    xpad = np.pad(x, (pad, pad), mode="constant")

    Npad = xpad.shape[0]

    # Nouvelle grille fréquentielle
    fgrid = np.fft.fftfreq(Npad, d=1.0 / fs)

    # FFT du signal paddé
    X = np.fft.fft(xpad)

    coefs = np.empty((len(scales), N), dtype=np.complex64)

    for i, s in enumerate(scales):
        Psi_s = wavelet.psi_scaled_on_grid(fgrid, scale=float(s))

        conv = np.fft.ifft(X * np.conj(Psi_s))

        # --- 2. RECADRAGE ---
        coefs[i, :] = conv[pad:pad + N]

    return coefs


def compute_dual_linear_cwt_rc(iq_data, wavelet: RaisedCosineWavelet, total_height, f_min, f_max, fs=1.0):
    """Compute dual-band (positive + negative) scalogram in dB using raised-cosine wavelet."""
    print(f"Computing raised-cosine CWT ({total_height} px)...")

    nb_rows_per_band = total_height // 2
    scales = generate_linear_scales_rc(f_min, f_max, nb_rows_per_band, wavelet, fs)

    coefs_pos = cwt_fft(iq_data, scales, wavelet, fs=fs)
    power_pos = np.abs(coefs_pos) ** 2

    coefs_neg = cwt_fft(np.conj(iq_data), scales, wavelet, fs=fs)
    power_neg = np.abs(coefs_neg) ** 2

    full_spec = np.vstack((power_pos, np.flipud(power_neg)))
    return 10.0 * np.log10(full_spec + 1e-12)