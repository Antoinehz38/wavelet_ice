import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import json
import datetime

# --- CONFIGURATION ---
INPUT_FILE = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-data"
META_FILE = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
OUTPUT_DIR = "data/wavelet_morlet"

DURATION_TO_READ = 600_000      # Temporal width for visualization
IMG_HEIGHT = 512               # Total image height (power of 2 recommended for CNN)
SAMPLE_RATE = 1.0              # Normalized Fs

# Wavelet choice: 'cmor1.5-1.0' for temporal precision (precise BBoxes)
# or 'cmor6.0-1.0' for frequency precision (but temporal blur)
WAVELET_NAME = 'cmor100.0-1.0'  # B = 100.0 (Bandwidth)
                                # C = 1.0 (Center Frequency)

# Frequency range to visualize
F_MAX = 0.5   # +Fs/2
F_MIN = 0.005 # Stop just before 0 to avoid division by zero (singularity)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_sigmf_chunk(filepath, num_samples, offset=0):
    try:
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None

def compute_linear_cwt(iq_data, wavelet_name, total_height, f_min, f_max):
    """
    Generate a scalogram with a LINEAR frequency axis.
    """
    print(f"Computing linearized CWT ({total_height} px height)...")
    
    # 1. Define the target frequency grid (linear)
    # Split the image in two: top half (positive), bottom half (negative)
    nb_rows_per_band = total_height // 2
    
    # Generate frequencies from f_max (top) to f_min (center)
    freqs_linear = np.linspace(f_max, f_min, nb_rows_per_band)
    
    # 2. Convert target frequencies to scales
    # Formula: scale = (center_freq * fs) / freq
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * SAMPLE_RATE) / freqs_linear
    
    # 3. Compute CWT on both bands with these exact scales
    # Positive band
    coefs_pos, _ = pywt.cwt(iq_data, scales, wavelet_name, sampling_period=1.0)
    power_pos = np.abs(coefs_pos)**2
    
    # Negative band (on conjugate)
    coefs_neg, _ = pywt.cwt(np.conj(iq_data), scales, wavelet_name, sampling_period=1.0)
    power_neg = np.abs(coefs_neg)**2
    
    # 4. STFT-like assembly
    # power_pos[0] corresponds to F_MAX (+0.5) -> top of image
    # power_pos[-1] corresponds to F_MIN (~0.0) -> middle
    # For the negative part, flip so -0.0 is at center and -0.5 at bottom
    full_spectrogram = np.vstack((power_pos, np.flipud(power_neg)))
    
    # Convert to dB
    full_spectrogram_db = 10 * np.log10(full_spectrogram + 1e-12)
    
    return full_spectrogram_db

def freq_to_pixel_linear(target_freq, total_height, f_max=0.5):
    """
    Convert Hz -> pixel Y for a purely linear axis.
    Image covers [+f_max ... 0 ... -f_max].
    """
    # Axis goes from +0.5 (Y=0) to -0.5 (Y=total_height)
    # Clamp to stay within the image
    if target_freq > f_max:
        target_freq = f_max
    if target_freq < -f_max:
        target_freq = -f_max
        
    y_pixel = total_height * (f_max - target_freq) / (2 * f_max)
    
    return int(y_pixel)

def save_linear_viz(spectrogram, output_dir, meta_file, duration_view):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"linear_cwt_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    
    h, w = spectrogram.shape
    
    plt.figure(figsize=(16, 10))
    
    extent = [0, duration_view, -F_MAX, F_MAX]
    
    vm = np.max(spectrogram)
    
    plt.imshow(spectrogram, 
               aspect='auto', 
               cmap='inferno', 
               origin='upper',
               vmin=vm-40, vmax=vm)

    # --- Bounding Boxes ---
    if os.path.exists(meta_file):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
            
        ax = plt.gca()
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] < duration_view:
                f_lower = ann['core:freq_lower_edge']
                f_upper = ann['core:freq_upper_edge']
                
                # Convert Hz -> pixels
                y_start = freq_to_pixel_linear(f_upper, h, F_MAX)
                y_end = freq_to_pixel_linear(f_lower, h, F_MAX)
                
                x_start = ann['core:sample_start']
                w_box = min(ann['core:sample_count'], duration_view - x_start)
                h_box = y_end - y_start
                
                rect = patches.Rectangle((x_start, y_start), w_box, h_box, 
                                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-')
                ax.add_patch(rect)
                plt.text(x_start, y_start-5, ann.get('core:description', ''), 
                         color='cyan', fontsize=10, fontweight='bold')

    yticks_pixels = np.linspace(0, h, 11)
    yticks_labels = np.round(np.linspace(F_MAX, -F_MAX, 11), 2)
    plt.yticks(yticks_pixels, yticks_labels)
    
    plt.ylabel("Frequency (Hz) - Linear Axis")
    plt.xlabel("Time (Samples)")
    plt.title(f"Deep Learning Ready Dataset Visualization (Linear Axis) - {timestamp}")
    plt.grid(True, color='white', alpha=0.1, linestyle='--')
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Linear image saved: {save_path}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    
    # 1. Load
    sig = load_sigmf_chunk(INPUT_FILE, DURATION_TO_READ)
    
    if sig is not None:
        # 2. Compute linear CWT
        spec_linear = compute_linear_cwt(sig, WAVELET_NAME, IMG_HEIGHT, F_MIN, F_MAX)
        
        # 3. Save with bounding boxes
        save_linear_viz(spec_linear, OUTPUT_DIR, META_FILE, DURATION_TO_READ)