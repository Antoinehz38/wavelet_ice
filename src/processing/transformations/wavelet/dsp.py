import numpy as np
import pywt


def generate_linear_scales(f_min, f_max, num_pixels, wavelet_name, fs=1.0):
    """Generate scales for a linear frequency axis."""
    freqs_linear = np.linspace(f_max, f_min, num_pixels)
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * fs) / freqs_linear
    return scales


def compute_dual_linear_cwt(iq_data, wavelet_name, total_height, f_min, f_max, fs=1.0):
    """Compute the dual-band (positive + negative) linearised scalogram.

    Returns a (total_height, time) matrix in dB.
    """
    print(f"Computing linear CWT ({total_height} px)...")

    nb_rows_per_band = total_height // 2
    scales = generate_linear_scales(f_min, f_max, nb_rows_per_band, wavelet_name, fs)

    coefs_pos, _ = pywt.cwt(iq_data, scales, wavelet_name, sampling_period=1.0)
    power_pos = np.abs(coefs_pos) ** 2

    coefs_neg, _ = pywt.cwt(np.conj(iq_data), scales, wavelet_name, sampling_period=1.0)
    power_neg = np.abs(coefs_neg) ** 2

    full_spectrogram = np.vstack((power_pos, np.flipud(power_neg)))

    return 10 * np.log10(full_spectrogram + 1e-12)


def freq_to_pixel_linear(target_freq, total_height, f_max=0.5):
    """Convert a frequency value to a pixel Y coordinate (linear axis)."""
    if target_freq > f_max:
        target_freq = f_max
    if target_freq < -f_max:
        target_freq = -f_max

    y_pixel = total_height * (f_max - target_freq) / (2 * f_max)
    return int(y_pixel)