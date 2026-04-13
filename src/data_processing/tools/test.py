import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, label, find_objects
from scipy.signal import butter, filtfilt


def smooth_1d(x, k):
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def segments_from_profile(profile, rel_thr=0.2, smooth_k=21, min_len=10):
    p = smooth_1d(profile, smooth_k)
    if p.max() <= 0:
        return [], p

    thr = rel_thr * p.max()
    m = p > thr

    segs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(m) - start >= min_len:
        segs.append((start, len(m)))

    return segs, p


def merge_close_segments(segs, gap=5):
    if not segs:
        return []
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def robust_row_z(gray):
    x = gray.astype(np.float32)
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    mad = np.maximum(mad, 1.0)
    z = (x - med) / mad
    z = np.clip(z, 0, None)
    return z


def detect_signals_by_projections(gray):
    H, W = gray.shape
    z = robust_row_z(gray)

    # léger écrêtage pour éviter que quelques pixels dominent
    z = np.clip(z, 0, 8)

    # -------- 1) segmentation temporelle globale --------
    time_profile = z.sum(axis=0)
    time_segs, time_profile_s = segments_from_profile(
        time_profile,
        rel_thr=0.15,
        smooth_k=max(31, (W // 40) | 1),
        min_len=max(20, W // 100)
    )
    time_segs = merge_close_segments(time_segs, gap=max(8, W // 200))

    boxes = []

    # -------- 2) segmentation fréquentielle dans chaque plage de temps --------
    for x0, x1 in time_segs:
        roi = z[:, x0:x1]
        freq_profile = roi.sum(axis=1)

        freq_segs, freq_profile_s = segments_from_profile(
            freq_profile,
            rel_thr=0.20,
            smooth_k=max(9, (H // 40) | 1),
            min_len=max(6, H // 80)
        )
        freq_segs = merge_close_segments(freq_segs, gap=max(2, H // 150))

        for y0, y1 in freq_segs:
            sub = z[y0:y1, x0:x1]

            # resserrage fin avec énergie locale
            col_energy = sub.sum(axis=0)
            row_energy = sub.sum(axis=1)

            col_segs, _ = segments_from_profile(
                col_energy,
                rel_thr=0.25,
                smooth_k=max(11, ((x1 - x0) // 20) | 1),
                min_len=max(8, (x1 - x0) // 30)
            )
            row_segs, _ = segments_from_profile(
                row_energy,
                rel_thr=0.25,
                smooth_k=max(5, ((y1 - y0) // 4) | 1),
                min_len=max(4, (y1 - y0) // 4)
            )

            if not col_segs:
                col_segs = [(0, x1 - x0)]
            if not row_segs:
                row_segs = [(0, y1 - y0)]

            # en général on garde seulement le segment principal
            cx0, cx1 = max(col_segs, key=lambda t: t[1] - t[0])
            ry0, ry1 = max(row_segs, key=lambda t: t[1] - t[0])

            bx0 = max(0, x0 + cx0 - 4)
            bx1 = min(W, x0 + cx1 + 4)
            by0 = max(0, y0 + ry0 - 2)
            by1 = min(H, y0 + ry1 + 2)

            if bx1 - bx0 >= 10 and by1 - by0 >= 4:
                boxes.append((bx0, by0, bx1 - bx0, by1 - by0))

    return boxes, {
        "z": z,
        "time_profile": time_profile,
        "time_segments": time_segs,
    }


def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

def tighten_box_with_energy(z, box, qx=(0.01, 0.99), qy=(0.01, 0.99), pad_x=0, pad_y=0):
    x, y, w, h = box
    sub = z[y:y+h, x:x+w]

    col_energy = sub.sum(axis=0)
    row_energy = sub.sum(axis=1)

    cx0, cx1 = tight_interval_from_energy(col_energy, *qx)
    ry0, ry1 = tight_interval_from_energy(row_energy, *qy)

    nx0 = max(0, x + cx0 - pad_x)
    nx1 = min(z.shape[1], x + cx1 + pad_x)
    ny0 = max(0, y + ry0 - pad_y)
    ny1 = min(z.shape[0], y + ry1 + pad_y)

    return (nx0, ny0, nx1 - nx0, ny1 - ny0)


from scipy.ndimage import gaussian_filter, label

def tighten_box_2d(z, box,
                   sigma=2,
                   thr_rel=0.3,
                   min_area_ratio=0.01,
                   pad=0):
    x, y, w, h = box
    sub = z[y:y+h, x:x+w]

    # lissage 2D
    sm = gaussian_filter(sub, sigma=sigma)

    # seuil relatif au max local (beaucoup plus stable ici)
    thr = sm.max() * thr_rel
    mask = sm > thr

    # composantes connexes
    lbl, n = label(mask)

    if n == 0:
        return box

    # garder la plus grosse composante
    best = None
    best_area = 0

    for i in range(1, n+1):
        ys, xs = np.where(lbl == i)
        area = len(xs)

        if area > best_area:
            best_area = area
            best = (xs, ys)

    xs, ys = best

    # bbox serrée
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()

    # coords globales
    nx0 = max(0, x + left - pad)
    nx1 = min(z.shape[1], x + right + pad)
    ny0 = max(0, y + top - pad)
    ny1 = min(z.shape[0], y + bottom + pad)

    return (nx0, ny0, nx1 - nx0, ny1 - ny0)

import numpy as np

def tight_interval_from_energy(profile, q_low=0.01, q_high=0.99):
    p = np.asarray(profile, dtype=np.float64)
    p = np.maximum(p, 0)

    s = p.sum()
    if s <= 0:
        return 0, len(p)

    c = np.cumsum(p) / s

    i0 = np.searchsorted(c, q_low)
    i1 = np.searchsorted(c, q_high)

    i0 = max(0, min(i0, len(p)-1))
    i1 = max(i0 + 1, min(i1 + 1, len(p)))

    return i0, i1


def moyenne_temporelle_spectrogramme(image, delta_t):
    F, T = image.shape
    n_fenetres = T // delta_t
    
    if n_fenetres == 0:
        raise ValueError("delta_t est plus grand que la dimension temporelle T de l'image.")
    image_tronquee = image[:, :n_fenetres * delta_t]
    
    image_remodelle = image_tronquee.reshape(F, n_fenetres, delta_t)
    
    spectrogramme_moyen = image_remodelle.mean(axis=2)
    
    return spectrogramme_moyen

def sauvegarder_visualisation_fenetre(image, spectrogramme_moyen, delta_t, i, chemin_sortie):
    F = image.shape[0]
    n_fenetres = spectrogramme_moyen.shape[1]
    
    if i < 0 or i >= n_fenetres:
        raise ValueError(f"L'index i={i} est invalide. Il doit être entre 0 et {n_fenetres - 1}.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))


    axes[0].imshow(image, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title(f"Spectrogramme original (delta_t={delta_t})")
    axes[0].set_xlabel("Temps (pixels)")
    axes[0].set_ylabel("Fréquence (pixels)")

    x_debut = i * delta_t
    y_debut = 0
    
    rect = patches.Rectangle((x_debut, y_debut), delta_t, F, 
                             linewidth=2, edgecolor='red', facecolor='none')

    axes[0].add_patch(rect)
   
    intensite_fenetre = spectrogramme_moyen[:, i]
    
    axes[1].plot(intensite_fenetre, color='blue')
    axes[1].set_title(f"Intensité moyenne (Fenêtre i={i})")
    axes[1].set_xlabel("Fréquence (pixels)")
    axes[1].set_ylabel("Intensité (0-255)")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].set_ylim(0, 260) 

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    
    plt.close()



def creer_et_appliquer_passe_bas(donnees, frequence_coupure, ordre=4, axe=0):
    if not 0 < frequence_coupure < 1:
        raise ValueError("La fréquence de coupure doit être strictement comprise entre 0 et 1.")
    b, a = butter(ordre, frequence_coupure, btype='low')
    donnees_filtrees = filtfilt(b, a, donnees, axis=axe)
    donnees_filtrees = np.clip(donnees_filtrees, 0, 255)
    return donnees_filtrees


def detecter_signaux_toutes_fenetres(spectrogramme_lisse, seuil=25):

    _, T_reduit = spectrogramme_lisse.shape
    liste_detections = []

    for i in range(T_reduit):
        profil_frequence = spectrogramme_lisse[:, i]
        masque = profil_frequence > seuil

        labels_regions, num_signaux = label(masque)
        tranches = find_objects(labels_regions)

        signaux_dans_fenetre = []
        for tranche in tranches:
            freq_min = tranche[0].start
            freq_max = tranche[0].stop
            signaux_dans_fenetre.append((freq_min, freq_max))

        liste_detections.append(signaux_dans_fenetre)
        
    return liste_detections

def fusionner_detections_precises(detections_affinees, tolerance=3):
    """
    Fusionne les détections en utilisant les temps exacts calculés.
    """
    finished_boxes = []
    active_boxes = []

    for i, detections_fenetre in enumerate(detections_affinees):
        next_active_boxes = []
        used_detections = set()

        for active_box in active_boxes:
            match_found = False
            
            for j, det in enumerate(detections_fenetre):
                if j in used_detections:
                    continue

                last_f_min = active_box['last_f_min']
                last_f_max = active_box['last_f_max']

                # Si on est dans la tolérance fréquentielle
                if (abs(det['f_min'] - last_f_min) <= tolerance and 
                    abs(det['f_max'] - last_f_max) <= tolerance):
                    
                    # --- MISE À JOUR DE LA BOÎTE ---
                    # On met à jour le temps maximum avec le t_max de ce bloc
                    active_box['global_t_max'] = max(active_box['global_t_max'], det['t_max'])
                    
                    # On élargit les fréquences si ça ondule
                    active_box['global_f_min'] = min(active_box['global_f_min'], det['f_min'])
                    active_box['global_f_max'] = max(active_box['global_f_max'], det['f_max'])
                    
                    active_box['last_f_min'] = det['f_min']
                    active_box['last_f_max'] = det['f_max']
                    
                    next_active_boxes.append(active_box)
                    used_detections.add(j)
                    match_found = True
                    break 
            
            if not match_found:
                finished_boxes.append(active_box)

        # Nouveaux signaux
        for j, det in enumerate(detections_fenetre):
            if j not in used_detections:
                next_active_boxes.append({
                    'global_t_min': det['t_min'], # Temps de départ exact
                    'global_t_max': det['t_max'], # Temps de fin exact
                    'global_f_min': det['f_min'],
                    'global_f_max': det['f_max'],
                    'last_f_min': det['f_min'],
                    'last_f_max': det['f_max']
                })
        
        active_boxes = next_active_boxes

    finished_boxes.extend(active_boxes)

    # --- FORMATAGE FINAL ---
    boxes_finales = []
    for box in finished_boxes:
        bx0 = box['global_t_min']
        bx1 = box['global_t_max']
        by0 = box['global_f_min']
        by1 = box['global_f_max']

        w = bx1 - bx0
        h = by1 - by0

        boxes_finales.append((bx0, by0, w, h))

    return boxes_finales

def affiner_bordures_temporelles(image_originale, liste_detections, delta_t, seuil_t=20, lissage_t=0.1):
    """
    Affine les bordures temporelles de chaque détection en regardant le signal d'origine.
    """
    detections_affinees = []
    
    for i, detections_fenetre in enumerate(liste_detections):
        fenetre_affinee = []
        # Limites théoriques du bloc actuel
        t_debut_bloc = i * delta_t
        t_fin_bloc = min((i + 1) * delta_t, image_originale.shape[1])
        
        for (f_min, f_max) in detections_fenetre:
            # 1. On extrait la "tranche" exacte du signal dans l'image d'origine
            # (seulement les fréquences du signal, et seulement pour ce delta_t)
            roi = image_originale[f_min:f_max, t_debut_bloc:t_fin_bloc]
            
            # 2. On fait la moyenne sur l'axe des fréquences (axe 0)
            # On obtient un profil temporel 1D de longueur delta_t
            profil_temporel = roi.mean(axis=0)
            
            # 3. Lissage (optionnel mais recommandé, on réutilise ta fonction)
            # On met un try/except au cas où delta_t serait trop petit pour le filtre
            try:
                profil_temporel = creer_et_appliquer_passe_bas(profil_temporel, lissage_t, axe=0)
            except ValueError:
                pass # Si le bloc est trop petit, on garde le profil brut
                
            # 4. Détection par seuil
            masque = profil_temporel > seuil_t
            
            # S'il y a du signal dans ce bloc
            if np.any(masque):
                indices_vrais = np.where(masque)[0]
                
                # Le premier et le dernier index qui dépassent le seuil
                t_offset_min = indices_vrais[0]
                t_offset_max = indices_vrais[-1] + 1 # +1 pour englober le pixel
                
                # On repasse en coordonnées globales (sur toute l'image)
                t_exact_min = t_debut_bloc + t_offset_min
                t_exact_max = t_debut_bloc + t_offset_max
                
                fenetre_affinee.append({
                    'f_min': f_min,
                    'f_max': f_max,
                    't_min': t_exact_min,
                    't_max': t_exact_max
                })
                
        detections_affinees.append(fenetre_affinee)
        
    return detections_affinees

if __name__ == "__main__":
    compressed_spec = cv2.imread("/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/debug.png", cv2.IMREAD_GRAYSCALE)
    spectrogramme_moyen = moyenne_temporelle_spectrogramme(compressed_spec, delta_t=100)

    i = 20
    intensite_lissage = 0.015
    delta_t = 100
    # sauvegarder_visualisation_fenetre(
    #     image=compressed_spec,
    #     spectrogramme_moyen=spectrogramme_moyen,
    #     delta_t=delta_t,
    #     i=i,
    #     chemin_sortie=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/fenetre_{delta_t}_{i}.png"
    # )   



    spectrogramme_lisse = creer_et_appliquer_passe_bas(
        spectrogramme_moyen, 
        frequence_coupure=intensite_lissage, 
        axe=0
    )

    sauvegarder_visualisation_fenetre(
        image=compressed_spec,
        spectrogramme_moyen=spectrogramme_lisse,
        delta_t=delta_t,
        i=i,
        chemin_sortie=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/fenetre_{delta_t}_{i}_lisse_{intensite_lissage}.png"
    )  

    detections_list = detecter_signaux_toutes_fenetres(spectrogramme_lisse, seuil=25)

    detection_affinees = affiner_bordures_temporelles(compressed_spec, detections_list, delta_t, seuil_t=20, lissage_t=0.1)
    

    boxes = fusionner_detections_precises(detection_affinees, delta_t)

    out = draw_boxes(compressed_spec, boxes)
    cv2.imwrite("/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/detections_fusionnees.png", out)



