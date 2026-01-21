import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import json
import datetime

# --- CONFIGURATION ---
INPUT_FILE = "/raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex2-tmpl3-20.04.sigmf-data"
META_FILE = "/raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex2-tmpl3-20.04.sigmf-meta"
OUTPUT_DIR = "/home/cei_ice_2025/guido_leteurtre_hanachowicz/wavelet_ice/data/wavelet_morlet"

DURATION_TO_READ = 20_000      # Largeur temporelle pour la visualisation
IMG_HEIGHT = 512               # Hauteur totale de l'image (Puissance de 2 recommandée pour CNN)
SAMPLE_RATE = 1.0              # Fs normalisée

# Choix Ondelette : 'cmor1.5-1.0' pour précision temporelle (BBox précises)
# ou 'cmor6.0-1.0' pour précision fréquentielle (mais flou temporel)
WAVELET_NAME = 'cmor100.0-1.0'  #$B = 100.0$ (La largeur de bande / Bandwidth)
                                #$C = 1.0$ (La fréquence centrale / Center Frequency) : askip pas utile de le changer

# Plage Fréquentielle à visualiser
F_MAX = 0.5   # +Fs/2
F_MIN = 0.005 # On s'arrête juste avant 0 pour éviter la division par zéro (singularité)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_sigmf_chunk(filepath, num_samples, offset=0):
    try:
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset)
        return data
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable : {filepath}")
        return None

def compute_linear_cwt(iq_data, wavelet_name, total_height, f_min, f_max):
    """
    Génère un scalogramme à axe fréquentiel LINEAIRE.
    """
    print(f"⏳ Calcul CWT Linéarisée ({total_height} px de haut)...")
    
    # 1. Définir la grille de fréquences cibles (Linéaire)
    # On divise l'image en deux : moitié haute (Pos), moitié basse (Neg)
    nb_rows_per_band = total_height // 2
    
    # On génère les fréquences de f_max (haut) vers f_min (centre)
    freqs_linear = np.linspace(f_max, f_min, nb_rows_per_band)
    
    # 2. Convertir ces fréquences cibles en ÉCHELLES (Scales)
    # Formule : scale = (center_freq * fs) / freq
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * SAMPLE_RATE) / freqs_linear
    
    # 3. Calcul CWT sur les deux bandes avec ces échelles précises
    # Bande Positive
    coefs_pos, _ = pywt.cwt(iq_data, scales, wavelet_name, sampling_period=1.0)
    power_pos = np.abs(coefs_pos)**2
    
    # Bande Négative (sur le conjugué)
    coefs_neg, _ = pywt.cwt(np.conj(iq_data), scales, wavelet_name, sampling_period=1.0)
    power_neg = np.abs(coefs_neg)**2
    
    # 4. Assemblage "STFT-like"
    # power_pos[0] correspond à F_MAX (+0.5) -> Haut de l'image
    # power_pos[-1] correspond à F_MIN (~0.0) -> Milieu
    
    # Pour la partie négative :
    # power_neg[0] correspond à F_MAX (donc -0.5 en physique)
    # power_neg[-1] correspond à F_MIN (donc -0.0 en physique)
    # On veut que le -0.0 soit au milieu et le -0.5 en bas.
    # Donc on doit inverser la partie négative (flipud)
    
    full_spectrogram = np.vstack((power_pos, np.flipud(power_neg)))
    
    # Conversion dB
    full_spectrogram_db = 10 * np.log10(full_spectrogram + 1e-12)
    
    return full_spectrogram_db

def freq_to_pixel_linear(target_freq, total_height, f_max=0.5):
    """
    Convertit Hz -> Pixel Y pour un axe purement linéaire.
    L'image couvre [+f_max ... 0 ... -f_max].
    """
    # L'axe va de +0.5 (Y=0) à -0.5 (Y=total_height)
    # Relation linéaire simple : Y = m * freq + p
    
    # Si freq = +0.5 -> y = 0
    # Si freq = -0.5 -> y = total_height
    
    # Formule de normalisation :
    # ratio = (freq - f_min_total) / (f_max_total - f_min_total)
    # Ici f_max_total = 0.5, f_min_total = -0.5 (range = 1.0)
    
    # On inverse car l'axe Y image descend (0 en haut)
    # pixel = total_height * (0.5 - freq) / 1.0
    
    # Clamp pour éviter de sortir de l'image
    if target_freq > f_max: target_freq = f_max
    if target_freq < -f_max: target_freq = -f_max
        
    y_pixel = total_height * (f_max - target_freq) / (2 * f_max)
    
    return int(y_pixel)

def save_linear_viz(spectrogram, output_dir, meta_file, duration_view):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"linear_cwt_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    
    h, w = spectrogram.shape
    
    plt.figure(figsize=(16, 10))
    
    # --- Affichage ---
    # Ici, l'extent reflète la réalité physique LINEAIRE !
    # Y va de +0.5 à -0.5
    extent = [0, duration_view, -F_MAX, F_MAX]
    
    vm = np.max(spectrogram)
    
    # Note: origin='upper' met le premier pixel en haut.
    # Dans notre matrice, la ligne 0 est +0.5 Hz.
    # Donc si on met origin='upper', l'axe Y doit afficher +0.5 en haut.
    # Matplotlib gère ça avec extent=[..., -0.5, 0.5] mais il faut inverser l'ordre des labels ou des données.
    # Le plus simple pour le debug : utiliser les pixels et customiser les ticks,
    # OU faire confiance à imshow avec les données bien ordonnées.
    
    plt.imshow(spectrogram, 
               aspect='auto', 
               # extent=extent, # On peut remettre l'extent physique car c'est linéaire !
               cmap='inferno', 
               origin='upper',
               vmin=vm-40, vmax=vm) # Contraste durci pour nettoyer le bruit (-40 dB ici)

    # --- Bounding Boxes ---
    if os.path.exists(meta_file):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
            
        ax = plt.gca()
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] < duration_view:
                # Récupération Hz
                f_lower = ann['core:freq_lower_edge']
                f_upper = ann['core:freq_upper_edge']
                
                # Conversion Hz -> Pixels
                y_start = freq_to_pixel_linear(f_upper, h, F_MAX) # Freq haute = Pixel petit (Haut)
                y_end = freq_to_pixel_linear(f_lower, h, F_MAX)   # Freq basse = Pixel grand (Bas)
                
                # Coords X
                x_start = ann['core:sample_start']
                w_box = min(ann['core:sample_count'], duration_view - x_start)
                h_box = y_end - y_start
                
                # Dessin
                rect = patches.Rectangle((x_start, y_start), w_box, h_box, 
                                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                plt.text(x_start, y_start-5, ann.get('core:description', ''), 
                         color='cyan', fontsize=10, fontweight='bold')

    # Customisation Axes Pixels -> Hz pour vérification humaine
    # On ajoute des ticks manuels pour prouver que c'est linéaire
    yticks_pixels = np.linspace(0, h, 11)
    yticks_labels = np.round(np.linspace(F_MAX, -F_MAX, 11), 2)
    plt.yticks(yticks_pixels, yticks_labels)
    
    plt.ylabel("Fréquence (Hz) - Axe Linéaire")
    plt.xlabel("Temps (Samples)")
    plt.title(f"Visualisation Dataset 'Deep Learning Ready' (Axe Linéaire) - {timestamp}")
    plt.grid(True, color='white', alpha=0.1, linestyle='--')
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Image Linéaire sauvegardée : {save_path}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    
    # 1. Chargement
    sig = load_sigmf_chunk(INPUT_FILE, DURATION_TO_READ)
    
    if sig is not None:
        # 2. Calcul CWT Linéaire (Le cœur de la Solution 2)
        # On demande explicitement une image de IMG_HEIGHT pixels de haut
        spec_linear = compute_linear_cwt(sig, WAVELET_NAME, IMG_HEIGHT, F_MIN, F_MAX)
        
        # 3. Sauvegarde avec Bbox (vérification)
        save_linear_viz(spec_linear, OUTPUT_DIR, META_FILE, DURATION_TO_READ)