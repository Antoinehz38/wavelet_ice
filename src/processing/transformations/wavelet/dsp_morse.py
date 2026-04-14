import numpy as np
from src.processing.transformations.wavelet.morse import MorseWavelet


def generate_linear_scales_morse(f_min, f_max, num_pixels, wavelet: MorseWavelet, fs=1.0):
    """
    Même logique que pour les autres ondelettes :
      freqs_linear = linspace(f_max -> f_min)
      scales = (fc * fs) / freqs_linear

    où fc est la fréquence centrale de l'ondelette mère.
    """
    freqs_linear = np.linspace(f_max, f_min, num_pixels)

    if np.any(freqs_linear <= 0):
        raise ValueError("f_min et f_max doivent être strictement positifs pour les échelles.")

    fc = wavelet.central_frequency
    scales = (fc * fs) / freqs_linear
    return scales


def cwt_fft(x: np.ndarray, scales: np.ndarray, wavelet: MorseWavelet, fs=1.0):
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


def compute_dual_linear_cwt_morse(iq_data, wavelet: MorseWavelet, total_height, f_min, f_max, fs=1.0):
    """
    Calcule un scalogramme double bande (positif + négatif) en dB,
    avec ondelette de Morse analytique.
    """
    print(f"⏳ Calcul CWT Morse Analytique Linéaire ({total_height} px)...")

    nb_rows_per_band = total_height // 2
    scales = generate_linear_scales_morse(f_min, f_max, nb_rows_per_band, wavelet, fs)

    # Bande positive
    coefs_pos = cwt_fft(iq_data, scales, wavelet, fs=fs)
    power_pos = np.abs(coefs_pos) ** 2

    # Bande négative via le conjugué
    coefs_neg = cwt_fft(np.conj(iq_data), scales, wavelet, fs=fs)
    power_neg = np.abs(coefs_neg) ** 2

    full_spec = np.vstack((power_pos, np.flipud(power_neg)))
    return 10.0 * np.log10(full_spec + 1e-12)