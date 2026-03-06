import numpy as np
from src.data_processing.tools.raised_cosine import RaisedCosineWavelet

def generate_linear_scales_rc(f_min, f_max, num_pixels, wavelet: RaisedCosineWavelet, fs=1.0):
    """
    Identique à votre logique:
      freqs_linear = linspace(f_max -> f_min)
      scales = (fc * fs) / freqs_linear
    """
    freqs_linear = np.linspace(f_max, f_min, num_pixels)
    fc = wavelet.central_frequency()
    scales = (fc * fs) / freqs_linear
    return scales

def cwt_fft(x: np.ndarray, scales: np.ndarray, wavelet: RaisedCosineWavelet, fs=1.0):
    """
    CWT via FFT: W(s,t) = ifft( FFT(x) * conj(Psi_s(f)) )
    Retourne coefs (n_scales, N) complex.
    """
    x = np.asarray(x)
    N = x.shape[0]

    # Grille fréquentielle (Hz) compatible FFT
    # d = 1/fs car dt = 1/fs
    fgrid = np.fft.fftfreq(N, d=1.0/fs)

    X = np.fft.fft(x)
    coefs = np.empty((len(scales), N), dtype=np.complex64)

    for i, s in enumerate(scales):
        Psi_s = wavelet.psi_scaled_on_grid(fgrid, scale=float(s))  # complex
        # Produit fréquentiel et retour temps
        coefs[i, :] = np.fft.ifft(X * np.conj(Psi_s)).astype(np.complex64)

    return coefs

def compute_dual_linear_cwt_rc(iq_data, wavelet: RaisedCosineWavelet, total_height, f_min, f_max, fs=1.0):
    """
    Calcule scalogramme double bande (Pos + Neg) en dB,
    avec ondelette raised-cosine analytique.
    """
    print(f"⏳ Calcul CWT RC Analytique Linéaire ({total_height} px)...")

    nb_rows_per_band = total_height // 2
    scales = generate_linear_scales_rc(f_min, f_max, nb_rows_per_band, wavelet, fs)

    # Bande positive
    coefs_pos = cwt_fft(iq_data, scales, wavelet, fs=fs)
    power_pos = np.abs(coefs_pos) ** 2

    # Bande négative (sur le conjugué)
    coefs_neg = cwt_fft(np.conj(iq_data), scales, wavelet, fs=fs)
    power_neg = np.abs(coefs_neg) ** 2

    full_spec = np.vstack((power_pos, np.flipud(power_neg)))
    return 10.0 * np.log10(full_spec + 1e-12)