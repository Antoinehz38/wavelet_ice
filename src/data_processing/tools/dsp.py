import numpy as np
import pywt

def generate_linear_scales(f_min, f_max, num_pixels, wavelet_name, fs=1.0):
    """Génère les échelles pour obtenir un axe fréquentiel linéaire."""
    # Axe fréquentiel cible (décroissant pour avoir les HF en haut de l'image)
    freqs_linear = np.linspace(f_max, f_min, num_pixels)
    
    # Conversion Freq -> Scale
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * fs) / freqs_linear
    return scales

def compute_dual_linear_cwt(iq_data, wavelet_name, total_height, f_min, f_max, fs=1.0):
    """
    Calcule le scalogramme double bande (Positif + Négatif) linéarisé.
    Retourne une matrice (total_height, time) en dB.
    """
    print(f"⏳ Calcul CWT Linéaire ({total_height} px)...")
    
    # On divise l'image en deux : moitié haute (Pos), moitié basse (Neg)
    nb_rows_per_band = total_height // 2
    
    scales = generate_linear_scales(f_min, f_max, nb_rows_per_band, wavelet_name, fs)
    
    # 1. Bande Positive
    coefs_pos, _ = pywt.cwt(iq_data, scales, wavelet_name, sampling_period=1.0)
    power_pos = np.abs(coefs_pos)**2
    
    # 2. Bande Négative (sur le conjugué)
    coefs_neg, _ = pywt.cwt(np.conj(iq_data), scales, wavelet_name, sampling_period=1.0)
    power_neg = np.abs(coefs_neg)**2
    
    # 3. Assemblage (Positif en haut, Négatif inversé en bas)
    full_spectrogram = np.vstack((power_pos, np.flipud(power_neg)))
    
    # Conversion dB
    return 10 * np.log10(full_spectrogram + 1e-12)

def freq_to_pixel_linear(target_freq, total_height, f_max=0.5):
    """Convertit Hz -> Pixel Y (Axe Linéaire)."""
    # Clamp
    if target_freq > f_max: target_freq = f_max
    if target_freq < -f_max: target_freq = -f_max
        
    # Formule linéaire : y = H * (0.5 - f)
    y_pixel = total_height * (f_max - target_freq) / (2 * f_max)
    return int(y_pixel)