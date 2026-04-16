import cv2
from matplotlib.pylab import det
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, find_objects
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit

from src.processing.tools.dsp import freq_to_pixel_linear
from src.processing.tools.loaders import load_metadata



def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def temporal_mean_spectrogram(image, delta_t):
    F, T = image.shape
    n_windows = T // delta_t

    if n_windows == 0:
        raise ValueError("delta_t is larger than the temporal dimension T of the image.")
    image_truncated = image[:, :n_windows * delta_t]

    image_reshaped = image_truncated.reshape(F, n_windows, delta_t)

    mean_spectrogram = image_reshaped.mean(axis=2)

    return mean_spectrogram

def save_window_visualisation(image, mean_spectrogram, delta_t, i, output_path):
    F = image.shape[0]
    n_windows = mean_spectrogram.shape[1]

    if i < 0 or i >= n_windows:
        raise ValueError(f"Index i={i} is invalid. Must be between 0 and {n_windows - 1}.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(image, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title(f"Original spectrogram (delta_t={delta_t})")
    axes[0].set_xlabel("Time (pixels)")
    axes[0].set_ylabel("Frequency (pixels)")

    x_start = i * delta_t
    y_start = 0

    rect = patches.Rectangle((x_start, y_start), delta_t, F,
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)

    window_intensity = mean_spectrogram[:, i]

    axes[1].plot(window_intensity, color='blue')
    axes[1].set_title(f"Mean intensity (Window i={i})")
    axes[1].set_xlabel("Frequency (pixels)")
    axes[1].set_ylabel("Intensity (0-255)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[1].set_ylim(0, 260)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    plt.close()


def save_visualisation_with_boxes(image, mean_spectrogram, delta_t, i, boxes, output_path, gt_boxes_pixels=None):
    """Generate a window visualisation with detected boxes and optional ground-truth boxes (cyan)."""
    F = image.shape[0]
    n_windows = mean_spectrogram.shape[1]

    if i < 0 or i >= n_windows:
        raise ValueError(f"Index i={i} is invalid. Must be between 0 and {n_windows - 1}.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(image, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title(f"Original spectrogram (Window i={i})")
    axes[0].set_xlabel("Time (pixels)")
    axes[0].set_ylabel("Frequency (pixels)")

    t_start_window = i * delta_t
    t_end_window = (i + 1) * delta_t

    rect = patches.Rectangle((t_start_window, 0), delta_t, F,
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)

    window_intensity = mean_spectrogram[:, i]

    axes[1].plot(window_intensity, color='blue', zorder=3)
    axes[1].set_title("Mean intensity and detected signals")
    axes[1].set_xlabel("Frequency (pixels)")
    axes[1].set_ylabel("Intensity (0-255)")
    axes[1].grid(True, linestyle='--', alpha=0.7, zorder=0)
    axes[1].set_ylim(0, 260)

    for (bx0, by0, w, h) in boxes:
        bx1 = bx0 + w
        by1 = by0 + h

        if bx0 < t_end_window and bx1 > t_start_window:
            axes[1].axvline(x=by0, color='green', linestyle='-', linewidth=2, zorder=2)
            axes[1].axvline(x=by1, color='green', linestyle='-', linewidth=2, zorder=2)
            axes[1].axvspan(by0, by1, color='green', alpha=0.15, zorder=1)

            box_rect = patches.Rectangle((bx0, by0), w, h,
                                         linewidth=1, edgecolor='lime', facecolor='none')
            axes[0].add_patch(box_rect)

    if gt_boxes_pixels is not None:
        for (bx0, by0, w, h) in gt_boxes_pixels:
            bx1 = bx0 + w
            by1 = by0 + h

            if bx0 < t_end_window and bx1 > t_start_window:
                axes[1].axvline(x=by0, color='cyan', linestyle='--', linewidth=2, zorder=2)
                axes[1].axvline(x=by1, color='cyan', linestyle='--', linewidth=2, zorder=2)
                axes[1].axvspan(by0, by1, color='cyan', alpha=0.10, zorder=1)

                gt_rect = patches.Rectangle((bx0, by0), w, h,
                                             linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
                axes[0].add_patch(gt_rect)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def apply_lowpass_filter(data, cutoff_freq, order=4, axis=0):
    if not 0 < cutoff_freq < 1:
        raise ValueError("Cutoff frequency must be strictly between 0 and 1.")
    b, a = butter(order, cutoff_freq, btype='low')
    filtered_data = filtfilt(b, a, data, axis=axis)
    filtered_data = np.clip(filtered_data, 0, 255)
    return filtered_data

def detect_robust_signals(smoothed_spectrogram,
                          min_prominence=5,
                          min_width=5,
                          min_distance=10,
                          max_intensity_ratio=1.5,
                          rolloff_threshold=0.90):
    """Detect signals with intelligent merging/splitting and roll-off trimming.
    Includes padding to accurately detect signals on the frequency boundaries."""
    _, T_reduced = smoothed_spectrogram.shape
    detections_list = []

    pad_width = 50 

    for i in range(T_reduced):
        freq_profile = smoothed_spectrogram[:, i]

        padded_profile = np.pad(freq_profile, (pad_width, pad_width), mode='constant', constant_values=0)

        peaks_padded, properties_padded = find_peaks(
            padded_profile,
            prominence=min_prominence,
            width=min_width,
            rel_height=0.85,
            distance=min_distance,
        )

        raw_bands = []
        if len(peaks_padded) > 0:
            peaks = peaks_padded - pad_width
            left_edges = properties_padded["left_ips"] - pad_width
            right_edges = properties_padded["right_ips"] - pad_width

            for peak_idx, f_min, f_max in zip(peaks, left_edges, right_edges):
                if 0 <= peak_idx < len(freq_profile):
                    f_min_clipped = max(0, int(f_min))
                    f_max_clipped = min(len(freq_profile) - 1, int(f_max))
                    
                    raw_bands.append({
                        'f_min': f_min_clipped,
                        'f_max': f_max_clipped,
                        'peak_idx': int(peak_idx),
                        'intensity': freq_profile[int(peak_idx)],
                    })

        if len(raw_bands) > 0:
            raw_bands.sort(key=lambda x: x['f_min'])
            merged_bands = [raw_bands[0]]

            for current_band in raw_bands[1:]:
                last_band = merged_bands[-1]
                touching = current_band['f_min'] <= last_band['f_max'] + 5

                if touching:
                    intensity_prev = last_band['intensity']
                    intensity_curr = current_band['intensity']
                    ratio = max(intensity_prev, intensity_curr) / max(min(intensity_prev, intensity_curr), 1)

                    idx_start = last_band['peak_idx']
                    idx_end = current_band['peak_idx']
                    valley_idx = np.argmin(freq_profile[idx_start:idx_end]) + idx_start if idx_end > idx_start else idx_start
                    valley_intensity = freq_profile[valley_idx]
                    relative_depth = valley_intensity / max(min(intensity_prev, intensity_curr), 1)

                    if ratio < max_intensity_ratio and relative_depth > 0.6:
                        last_band['f_max'] = max(last_band['f_max'], current_band['f_max'])
                        if intensity_curr > intensity_prev:
                            last_band['intensity'] = intensity_curr
                            last_band['peak_idx'] = current_band['peak_idx']
                    else:
                        last_band['f_max'] = valley_idx
                        current_band['f_min'] = valley_idx + 1
                        merged_bands.append(current_band)
                else:
                    merged_bands.append(current_band)

            trimmed_bands = []
            for b in merged_bands:
                f_min = b['f_min']
                f_max = min(b['f_max'], len(freq_profile) - 1)

                if f_max > f_min:
                    segment = freq_profile[f_min:f_max + 1]
                    trim_threshold = np.max(segment) * rolloff_threshold

                    valid_indices = np.where(segment >= trim_threshold)[0]

                    if len(valid_indices) > 0:
                        new_f_min = f_min + valid_indices[0]
                        new_f_max = f_min + valid_indices[-1]
                        trimmed_bands.append((new_f_min, new_f_max))
                    else:
                        trimmed_bands.append((f_min, f_max))
                else:
                    trimmed_bands.append((f_min, f_max))

            signals_in_window = trimmed_bands
        else:
            signals_in_window = []

        detections_list.append(signals_in_window)

    return detections_list

def gaussian(x, amplitude, mean, stddev):
    """Équation d'une seule courbe en cloche."""
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def double_gaussian(x, a1, m1, s1, a2, m2, s2):
    """Somme de deux courbes en cloche qui se chevauchent."""
    return gaussian(x, a1, m1, s1) + gaussian(x, a2, m2, s2)


def extract_signals_curvefit(freq_profile, base_f_min, base_f_max, current_f_min, current_f_max, rolloff_threshold=0.3):
    """
    Méthode 2: Ajustement mathématique (Curve Fitting).
    Force la reconnaissance de 2 cloches distinctes dans le profil fusionné.
    """
    # --- 1. SÉCURITÉ : Garantir que les variables sont dans le bon ordre ---
    base_f_min, base_f_max = min(base_f_min, base_f_max), max(base_f_min, base_f_max)
    current_f_min, current_f_max = min(current_f_min, current_f_max), max(current_f_min, current_f_max)

    pad = 20
    c_min = max(0, current_f_min - pad)
    c_max = min(len(freq_profile), current_f_max + pad)
    
    x_data = np.arange(c_min, c_max)
    y_data = freq_profile[c_min:c_max]
    
    if len(y_data) < 10:
        # --- 2. SÉCURITÉ : Ne pas retourner de valeurs inversées ---
        return min(base_f_max, current_f_max), max(base_f_max, current_f_max)

    # --- Hypothèses initiales pour aider l'algorithme ---
    m1_guess = (base_f_min + base_f_max) / 2.0
    s1_guess = max(2.0, (base_f_max - base_f_min) / 4.0)
    a1_guess = np.max(freq_profile[base_f_min:base_f_max]) if base_f_max > base_f_min else 100.0

    # Cloche 2 (Le nouveau signal)
    if current_f_max > base_f_max: # Empilement par le HAUT
        m2_guess = (base_f_max + current_f_max) / 2.0
        s2_guess = max(2.0, (current_f_max - base_f_max) / 4.0)
        search_area = freq_profile[base_f_max:current_f_max]
        a2_guess = np.max(search_area) if len(search_area) > 0 else 100.0
        
        bounds = (
            [0, base_f_min - 10, 1, 0, base_f_max, 1], 
            [np.inf, base_f_max + 10, np.inf, np.inf, current_f_max + 10, np.inf] 
        )
    else: # Empilement par le BAS (ou signal englobé)
        m2_guess = (current_f_min + base_f_min) / 2.0
        s2_guess = max(2.0, (base_f_min - current_f_min) / 4.0)
        search_area = freq_profile[current_f_min:base_f_min]
        a2_guess = np.max(search_area) if len(search_area) > 0 else 100.0
        
        lower_m2 = min(current_f_min - 10, base_f_min - 1)
        
        bounds = (
            [0, base_f_min - 10, 1, 0, lower_m2, 1],
            [np.inf, base_f_max + 10, np.inf, np.inf, base_f_min, np.inf]
        )

    # --- 3. SÉCURITÉ ABSOLUE POUR SCIPY ---
    # Convertir en tableau float et forcer lower < upper
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    
    for k in range(len(lb)):
        if lb[k] >= ub[k]:
            lb[k] = ub[k] - 1e-5  # Oblige la borne inférieure à être strictement plus petite
            
    bounds = (lb, ub)
    p0 = [a1_guess, m1_guess, s1_guess, a2_guess, m2_guess, s2_guess]
    p0 = np.clip(p0, bounds[0], bounds[1])

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit de scipy
            popt, _ = curve_fit(double_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=1000)
            
        a1, m1, s1, a2, m2, s2 = popt
        
        spread = s2 * np.sqrt(-2 * np.log(max(rolloff_threshold, 1e-5)))
        new_f_min = max(0, int(m2 - spread))
        new_f_max = min(len(freq_profile)-1, int(m2 + spread))
        
        # --- 4. SÉCURITÉ : Protéger la sortie contre une inversion due aux limites ---
        return min(new_f_min, new_f_max), max(new_f_min, new_f_max)

    except RuntimeError:
        if current_f_max > base_f_max:
            return min(base_f_max, current_f_max), max(base_f_max, current_f_max)
        else:
            return min(current_f_min, base_f_min), max(current_f_min, base_f_min)


def merge_precise_detections(refined_detections, smoothed_spectrogram, tolerance=15, split_threshold=30):
    """Merge detections, with mathematical footprint subtraction for overlapping signals."""
    finished_boxes = []
    active_boxes = []
    
    for i, window_detections in enumerate(refined_detections):
        next_active_boxes = []
        used_detections = set()
        
        # Profil actuel pour les ajouts, et profil PRÉCÉDENT pour les disparitions
        freq_profile = smoothed_spectrogram[:, i]
        prev_freq_profile = smoothed_spectrogram[:, max(0, i-1)] 
        
        # Copie locale des dictionnaires pour pouvoir les découper (modifier) en direct
        current_window_dets = [dict(d) for d in window_detections]
        for active_box in active_boxes:
            match_found = False
            
            for j in range(len(current_window_dets)):
                if j in used_detections:
                    continue
                    
                det = current_window_dets[j]
                last_f_min = active_box['last_f_min']
                last_f_max = active_box['last_f_max']
                
                bottom_matches = abs(det['f_min'] - last_f_min) <= tolerance
                top_matches = abs(det['f_max'] - last_f_max) <= tolerance
                
                if bottom_matches and top_matches:
                    # Match parfait : Le signal continue normalement
                    active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                    active_box['global_f_min'] = min(active_box['global_f_min'], det['f_min'])
                    active_box['global_f_max'] = max(active_box['global_f_max'], det['f_max'])
                    active_box['last_f_min'] = det['f_min']
                    active_box['last_f_max'] = det['f_max']
                    next_active_boxes.append(active_box)
                    used_detections.add(j)
                    match_found = True
                    break

                elif bottom_matches and not top_matches:
                    diff_top = det['f_max'] - last_f_max

                    if diff_top >= split_threshold:
                        new_f_min, new_f_max = extract_signals_curvefit(
                                freq_profile, last_f_min, last_f_max, det['f_min'], det['f_max']
                                )
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_min'] = min(active_box['global_f_min'], det['f_min'])
                        active_box['last_f_min'] = det['f_min']
                        active_box['last_f_max'] = last_f_max
                        next_active_boxes.append(active_box)
                        match_found = True
                        det['f_min'] = new_f_min
                        det['f_max'] = new_f_max
                        break

                    elif diff_top <= -split_threshold:
                        dropped_f_min, dropped_f_max = extract_signals_curvefit(
                            prev_freq_profile, det['f_min'], det['f_max'], last_f_min, last_f_max
                        )
                        finished_boxes.append({
                            'global_t_min': active_box['global_t_min'],
                            'global_t_max': active_box['global_t_max'],
                            'global_f_min': dropped_f_min,
                            'global_f_max': active_box['global_f_max'],
                            'last_f_min': dropped_f_min,
                            'last_f_max': active_box['last_f_max']
                        })
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_max'] = det['f_max']
                        active_box['last_f_min'] = det['f_min']
                        active_box['last_f_max'] = det['f_max']
                        next_active_boxes.append(active_box)
                        used_detections.add(j)
                        match_found = True
                        break

                    else:
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_max'] = max(active_box['global_f_max'], det['f_max'])
                        active_box['last_f_min'] = det['f_min']
                        active_box['last_f_max'] = det['f_max']
                        next_active_boxes.append(active_box)
                        used_detections.add(j)
                        match_found = True
                        break


                elif top_matches and not bottom_matches:
                    diff_bottom = last_f_min - det['f_min']
                    
                    if diff_bottom >= split_threshold:
                        new_f_min, new_f_max = extract_signals_curvefit(
                            freq_profile, last_f_min, last_f_max, det['f_min'], det['f_max']
                        )
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_max'] = max(active_box['global_f_max'], det['f_max'])
                        active_box['last_f_max'] = det['f_max']
                        active_box['last_f_min'] = last_f_min
                        next_active_boxes.append(active_box)
                        match_found = True
                        # On découpe la détection en direct
                        det['f_min'] = new_f_min
                        det['f_max'] = new_f_max
                        break

                    elif diff_bottom <= -split_threshold:
                        # --- SIGNAL S'ARRÊTE EN HAUT ---
                        # On cherche la bordure dans la frame PRÉCÉDENTE
                        dropped_f_min, dropped_f_max = extract_signals_curvefit(
                            prev_freq_profile, det['f_min'], det['f_max'], last_f_min, last_f_max
                        )
                        finished_boxes.append({
                            'global_t_min': active_box['global_t_min'],
                            'global_t_max': active_box['global_t_max'],
                            'global_f_min': active_box['global_f_min'],
                            'global_f_max': dropped_f_max,
                            'last_f_min': active_box['last_f_min'],
                            'last_f_max': dropped_f_max
                        })
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_min'] = det['f_min']
                        active_box['last_f_min'] = det['f_min']
                        active_box['last_f_max'] = det['f_max']
                        next_active_boxes.append(active_box)
                        used_detections.add(j)
                        match_found = True
                        break

                    else:
                        active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                        active_box['global_f_min'] = min(active_box['global_f_min'], det['f_min'])
                        active_box['last_f_min'] = det['f_min']
                        active_box['last_f_max'] = det['f_max']
                        next_active_boxes.append(active_box)
                        used_detections.add(j)
                        match_found = True
                        break
            if not match_found:
                finished_boxes.append(active_box)

        # Les détections restantes (incluant celles qui viennent d'être "découpées") démarrent de nouvelles boîtes
        for j in range(len(current_window_dets)):
            if j not in used_detections:
                det = current_window_dets[j]
                next_active_boxes.append({
                    'global_t_min': det['t_min'],
                    'global_t_max': det['t_max'],
                    'global_f_min': det['f_min'],
                    'global_f_max': det['f_max'],
                    'last_f_min': det['f_min'],
                    'last_f_max': det['f_max'],
                })
        
        # --- CORRECTION INDENTATION : Mise à jour globale pour la prochaine fenêtre ---
        active_boxes = next_active_boxes
        
    # Fin de la boucle sur toutes les fenêtres temporelles
    finished_boxes.extend(active_boxes)
                    
    final_boxes = []
    for box in finished_boxes:
        bx0 = box['global_t_min']
        bx1 = box['global_t_max']
        by0 = box['global_f_min']
        by1 = box['global_f_max']
        final_boxes.append((bx0, by0, bx1 - bx0, by1 - by0))

    return final_boxes



def refine_temporal_borders(original_image, detections_list, delta_t, min_amplitude=5, margin=4, debug=False):
    refined_detections = []
    
    # --- NOUVEAU : Fonction utilitaire pour vérifier la continuité ---
    def has_overlap_in_adjacent_window(f_min, f_max, window_idx, direction):
        target_idx = window_idx + direction
        if target_idx < 0 or target_idx >= len(detections_list):
            return False
        
        # On vérifie si la bande de fréquence chevauche une détection de la fenêtre voisine
        for (other_f_min, other_f_max) in detections_list[target_idx]:
            if max(f_min, other_f_min) < min(f_max, other_f_max):
                return True
        return False

    for i, window_detections in enumerate(detections_list):
        refined_window = []
        block_t_start = i * delta_t
        block_t_end = min((i + 1) * delta_t, original_image.shape[1])
        window_width = block_t_end - block_t_start
        
        for (f_min, f_max) in window_detections:
            roi = original_image[f_min:f_max, block_t_start:block_t_end]
            temporal_profile = roi.mean(axis=0)
            
            kernel_size = min(5, len(temporal_profile)) 
            if kernel_size > 0:
                pad_w = kernel_size // 2
                padded_profile = np.pad(temporal_profile, (pad_w, pad_w), mode='edge')
                kernel = np.ones(kernel_size) / kernel_size
                temporal_profile = np.convolve(padded_profile, kernel, mode='valid')
           
            min_val = np.min(temporal_profile)
            max_val = np.max(temporal_profile)
            amplitude = max_val - min_val
            
            detected_start = None
            detected_end = None 
            threshold_used = None

            if amplitude > min_amplitude: 
                dynamic_threshold = min_val + (amplitude * 0.20)
                threshold_used = dynamic_threshold
                
                mask = temporal_profile > dynamic_threshold
                true_indices = np.where(mask)[0]
                
                if len(true_indices) > 0:
                    t_offset_min = true_indices[0]
                    t_offset_max = true_indices[-1] + 1
                    
                    # --- CORRECTION : Aimantation (Snapping) Contextuelle ---
                    # On n'aimante à gauche QUE si le signal vient de la fenêtre précédente
                    signal_comes_from_left = has_overlap_in_adjacent_window(f_min, f_max, i, -1)
                    if t_offset_min <= margin and signal_comes_from_left:
                        t_offset_min = 0
                        
                    # On n'aimante à droite QUE si le signal continue dans la fenêtre suivante
                    signal_continues_right = has_overlap_in_adjacent_window(f_min, f_max, i, 1)
                    if (window_width - t_offset_max) <= margin and signal_continues_right:
                        t_offset_max = window_width
                    
                    detected_start = t_offset_min
                    detected_end = t_offset_max if t_offset_max < window_width else window_width - 1
                    
                    t_exact_min = block_t_start + t_offset_min
                    t_exact_max = block_t_start + t_offset_max
                    
                    refined_window.append({
                        'f_min': f_min,
                        'f_max': f_max,
                        't_min': t_exact_min,
                        't_max': t_exact_max,
                    })
            else:
                # Si l'amplitude est trop faible mais qu'il y a du signal, on garde la fenêtre.
                # (Attention, si un début de signal est TRÈS faible, il passera par ici et ne sera pas affiné)
                if np.max(temporal_profile) > 15: 
                    refined_window.append({
                        'f_min': f_min,
                        'f_max': f_max,
                        't_min': block_t_start,
                        't_max': block_t_end,
                    })
            
            # --- BLOC DEBUG ---
            has_real_start = (detected_start is not None) and (detected_start > 0)
            has_real_end = (detected_end is not None) and (detected_end < window_width - 1)
            
            if debug and (has_real_start or has_real_end):
                plt.figure(figsize=(10, 4))
                plt.plot(temporal_profile, color='blue', label='Profil temporel lissé')
                
                if has_real_start:
                    plt.axvline(x=detected_start, color='red', linestyle='-', linewidth=2, 
                                label=f'Début signal (t={detected_start})')
                
                if has_real_end:
                    plt.axvline(x=detected_end, color='orange', linestyle='-', linewidth=2, 
                                label=f'Fin signal (t={detected_end})')
                            
                if threshold_used is not None:
                    plt.axhline(y=threshold_used, color='green', linestyle='--', alpha=0.7, 
                                label=f'Seuil dyn. ({threshold_used:.1f})')
                
                plt.title(f"Temporal profile for window {i}")
                plt.legend(loc='lower right')
                plt.tight_layout()
                # Remets ton chemin d'enregistrement ici si besoin
                plt.savefig(f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/temporal_profile_window_{i}_{f_min}_{f_max}.png", dpi=150) 
                plt.close()
                
        refined_detections.append(refined_window)
        
    return refined_detections

def refine_one_detection(roi, window_width, min_amplitude, margin, noise_floor=15):
    """Single-detection refinement logic."""
    temporal_profile = roi.mean(axis=0)
    
    kernel_size = min(5, len(temporal_profile)) 
    if kernel_size > 0:
        pad_w = kernel_size // 2
        padded_profile = np.pad(temporal_profile, (pad_w, pad_w), mode='edge')
        kernel = np.ones(kernel_size) / kernel_size
        temporal_profile = np.convolve(padded_profile, kernel, mode='valid')
   
    min_val = np.min(temporal_profile)
    max_val = np.max(temporal_profile)
    amplitude = max_val - min_val
    
    if amplitude > min_amplitude: 
        dynamic_threshold = min_val + (amplitude * 0.50)
        # --- LA CORRECTION EST ICI AUSSI ---
        final_threshold = max(dynamic_threshold, noise_floor)
        
        mask = temporal_profile > final_threshold
        true_indices = np.where(mask)[0]
        
        if len(true_indices) > 0:
            t_offset_min = true_indices[0]
            t_offset_max = true_indices[-1] + 1
            
            if t_offset_min <= margin:
                t_offset_min = 0
            if (window_width - t_offset_max) <= margin:
                t_offset_max = window_width
            
            return t_offset_min, t_offset_max
            
    if np.max(temporal_profile) > noise_floor: 
        return 0, window_width
    return None, None

PARAMS = {
    'offset': 0,
    'duration': 2_000_000,
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 1_000_000,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': "cmor100.0-1.0",
    'transform': 'cwt',
    'rc_fc': 1.0,
    'rc_B': 0.12,
    'rc_beta': 0.25,
    'detect_db_range': 28,
    'detect_kernel': (200, 2),
    'downsample_factor': 500,
    'saveRaw': False,
    'addPrediction': False,
}

if __name__ == "__main__":
    file_path = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/spectrogram_20260414_093653.png"
    compressed_spec = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    filename = file_path.split("/")[-1].replace(".png", "")
    mean_spec = temporal_mean_spectrogram(compressed_spec, delta_t=100)

    i = 19
    smoothing_intensity = 0.03
    delta_t = 100

    smoothed_spec = apply_lowpass_filter(
        mean_spec,
        cutoff_freq=smoothing_intensity,
        axis=0,
    )

    save_window_visualisation(
        image=compressed_spec,
        mean_spectrogram=mean_spec,
        delta_t=delta_t,
        i=i,
        output_path=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/{filename}_window_{delta_t}_{i}_raw.png",
    )

    save_window_visualisation(
        image=compressed_spec,
        mean_spectrogram=smoothed_spec,
        delta_t=delta_t,
        i=i,
        output_path=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/{filename}_window_{delta_t}_{i}_smooth_{smoothing_intensity}.png",
    )
    gt_boxes_pixels = []

    img_h = compressed_spec.shape[0]
    ds = 500
    scale_y = img_h / PARAMS['img_height']

    meta = load_metadata("/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta")

    gt_boxes_pixels = []
    if meta:
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] <2_000_000:
                y_start = freq_to_pixel_linear(ann['core:freq_upper_edge'], PARAMS['img_height'], PARAMS['f_max'])
                y_end = freq_to_pixel_linear(ann['core:freq_lower_edge'], PARAMS['img_height'], PARAMS['f_max'])

                if y_start < 0:
                    y_start = 0
                if y_end > PARAMS['img_height']:
                    y_end = PARAMS['img_height']

                x = ann['core:sample_start']
                w = min(ann['core:sample_count'], PARAMS['duration'] - x)
                h = y_end - y_start

                cx = x / ds
                cw = w / ds
                cy = y_start * scale_y
                ch = h * scale_y

                gt_boxes_pixels.append((cx, cy, cw, ch))

    roll_off_threshold = 0.30
    detections_list = detect_robust_signals(smoothed_spec, rolloff_threshold=roll_off_threshold, 
                                            min_prominence=3, min_width=2, min_distance=1, max_intensity_ratio=3
                                            )
    
    refined = refine_temporal_borders(compressed_spec, detections_list, delta_t, debug=True)

    boxes = merge_precise_detections(refined, tolerance=15, smoothed_spectrogram=smoothed_spec, split_threshold=5)

    save_visualisation_with_boxes(
        boxes=boxes,
        gt_boxes_pixels=gt_boxes_pixels,
        image=compressed_spec,
        mean_spectrogram=smoothed_spec,
        delta_t=delta_t,
        i=i,
        output_path=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/{filename}_window_{delta_t}_{i}_with_boxes_roth_{roll_off_threshold}.png",
    )

    out = draw_boxes(compressed_spec, boxes)
    cv2.imwrite(f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/hp/{filename}_merged_detections.png", out)


