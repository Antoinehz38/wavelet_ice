import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, find_objects
from scipy.signal import butter, filtfilt, find_peaks

from src.data_processing.tools import evaluations, dsp, loaders, viz, dsp_rc



def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


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


def sauvegarder_visualisation_avec_boxes(image, spectrogramme_moyen, delta_t, i, boxes, chemin_sortie, gt_boxes_pixels=None):
    """
    Génère la visualisation d'une fenêtre avec les délimitations des boîtes détectées
    et les ground truth boxes (gt_boxes_pixels) en cyan.
    """
    F = image.shape[0]
    n_fenetres = spectrogramme_moyen.shape[1]
    
    if i < 0 or i >= n_fenetres:
        raise ValueError(f"L'index i={i} est invalide. Il doit être entre 0 et {n_fenetres - 1}.")
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ==========================================
    # GAUCHE : Spectrogramme + Rectangle
    # ==========================================
    axes[0].imshow(image, cmap='gray', aspect='auto', origin='lower')
    axes[0].set_title(f"Spectrogramme original (Fenêtre i={i})")
    axes[0].set_xlabel("Temps (pixels)")
    axes[0].set_ylabel("Fréquence (pixels)")

    # Limites temporelles théoriques de la fenêtre actuelle
    t_debut_fenetre = i * delta_t
    t_fin_fenetre = (i + 1) * delta_t
    
    # Rectangle rouge indiquant la zone qu'on observe
    rect = patches.Rectangle((t_debut_fenetre, 0), delta_t, F, 
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
   
    # ==========================================
    # DROITE : Intensité en fonction de la fréquence + Traits verts
    # ==========================================
    intensite_fenetre = spectrogramme_moyen[:, i]
    
    axes[1].plot(intensite_fenetre, color='blue', zorder=3)
    axes[1].set_title(f"Intensité moyenne et Signaux détectés")
    axes[1].set_xlabel("Fréquence (pixels)")
    axes[1].set_ylabel("Intensité (0-255)")
    axes[1].grid(True, linestyle='--', alpha=0.7, zorder=0)
    axes[1].set_ylim(0, 260) 

    # --- AJOUT DES LIGNES VERTES BASÉES SUR LES BOXES ---
    for (bx0, by0, w, h) in boxes:
        bx1 = bx0 + w
        by1 = by0 + h
        
        # On vérifie si la boîte traverse notre fenêtre temporelle actuelle
        # Condition d'intersection : le début de la boîte est avant la fin de la fenêtre 
        # ET la fin de la boîte est après le début de la fenêtre
        if bx0 < t_fin_fenetre and bx1 > t_debut_fenetre:
            
            # Dessin des traits verts pour la fréquence min (by0) et max (by1)
            axes[1].axvline(x=by0, color='green', linestyle='-', linewidth=2, zorder=2)
            axes[1].axvline(x=by1, color='green', linestyle='-', linewidth=2, zorder=2)
            
            # Bonus : on colorie légèrement la zone du signal en vert pour que ce soit très lisible
            axes[1].axvspan(by0, by1, color='green', alpha=0.15, zorder=1)
            
            # Optionnel : On peut aussi dessiner les boxes sur l'image de gauche 
            # pour voir comment elles se superposent au signal brut
            box_rect = patches.Rectangle((bx0, by0), w, h, 
                                         linewidth=1, edgecolor='lime', facecolor='none')
            axes[0].add_patch(box_rect)

    # --- AJOUT DES LIGNES CYAN BASÉES SUR LES GT BOXES ---
    if gt_boxes_pixels is not None:
        for (bx0, by0, w, h) in gt_boxes_pixels:
            bx1 = bx0 + w
            by1 = by0 + h
            
            # On vérifie si la boîte GT traverse notre fenêtre temporelle actuelle
            if bx0 < t_fin_fenetre and bx1 > t_debut_fenetre:
                
                # Dessin des traits cyan pour la fréquence min (by0) et max (by1)
                axes[1].axvline(x=by0, color='cyan', linestyle='--', linewidth=2, zorder=2)
                axes[1].axvline(x=by1, color='cyan', linestyle='--', linewidth=2, zorder=2)
                
                # Zone du signal GT en cyan
                axes[1].axvspan(by0, by1, color='cyan', alpha=0.10, zorder=1)
                
                # Dessiner les GT boxes sur l'image de gauche en cyan
                gt_rect = patches.Rectangle((bx0, by0), w, h, 
                                             linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
                axes[0].add_patch(gt_rect)

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


def detecter_signaux_robustes(spectrogramme_lisse, 
                              proeminence_min=5, 
                              largeur_min=5, 
                              distance_min=10, 
                              ratio_intensite_max=1.5, 
                              seuil_rolloff=0.90):
    """
    Détecte les signaux, gère les fusions/séparations complexes, et 
    rogne les bords (roll-off) pour des boîtes parfaitement ajustées à l'énergie.
    """
    _, T_reduit = spectrogramme_lisse.shape
    liste_detections = []
    
    for i in range(T_reduit):
        profil_frequence = spectrogramme_lisse[:, i]
        
        # 1. Détection initiale (volontairement large au niveau de la base)
        pics, proprietes = find_peaks(
            profil_frequence, 
            prominence=proeminence_min, 
            width=largeur_min,
            rel_height=0.85, 
            distance=distance_min
        )
        
        bandes_brutes = []
        if len(pics) > 0:
            bords_gauches = proprietes["left_ips"]
            bords_droits = proprietes["right_ips"]
            
            for pic_idx, f_min, f_max in zip(pics, bords_gauches, bords_droits):
                bandes_brutes.append({
                    'f_min': int(f_min),
                    'f_max': int(f_max),
                    'pic_idx': pic_idx,
                    'intensite': profil_frequence[pic_idx]
                })
                
        # 2. FUSION INTELLIGENTE PAR VALLÉE
        if len(bandes_brutes) > 0:
            bandes_brutes.sort(key=lambda x: x['f_min'])
            bandes_fusionnees = [bandes_brutes[0]]
            
            for bande_actuelle in bandes_brutes[1:]:
                derniere_bande = bandes_fusionnees[-1]
                se_touchent = bande_actuelle['f_min'] <= derniere_bande['f_max'] + 5
                
                if se_touchent:
                    intensite_prec = derniere_bande['intensite']
                    intensite_act = bande_actuelle['intensite']
                    ratio = max(intensite_prec, intensite_act) / max(min(intensite_prec, intensite_act), 1)
                    
                    idx_debut = derniere_bande['pic_idx']
                    idx_fin = bande_actuelle['pic_idx']
                    idx_vallee = np.argmin(profil_frequence[idx_debut : idx_fin]) + idx_debut if idx_fin > idx_debut else idx_debut
                    intensite_vallee = profil_frequence[idx_vallee]
                    profondeur_relative = intensite_vallee / max(min(intensite_prec, intensite_act), 1)
                    
                    if ratio < ratio_intensite_max and profondeur_relative > 0.6:
                        # Fusion du plateau
                        derniere_bande['f_max'] = max(derniere_bande['f_max'], bande_actuelle['f_max'])
                        if intensite_act > intensite_prec:
                            derniere_bande['intensite'] = intensite_act
                            derniere_bande['pic_idx'] = bande_actuelle['pic_idx']
                    else:
                        # Séparation par la vallée
                        derniere_bande['f_max'] = idx_vallee
                        bande_actuelle['f_min'] = idx_vallee + 1
                        bandes_fusionnees.append(bande_actuelle)
                else:
                    bandes_fusionnees.append(bande_actuelle)
            
            # ---------------------------------------------------------
            # 3. NOUVEAU : ROGNAGE DU ROLL-OFF (RÉDUCTION DE LA LARGEUR)
            # ---------------------------------------------------------
            bandes_trimmees = []
            for b in bandes_fusionnees:
                f_min = b['f_min']
                f_max = min(b['f_max'], len(profil_frequence) - 1) # Sécurité bordure
                
                if f_max > f_min:
                    # On isole le profil d'intensité uniquement pour cette bande
                    segment = profil_frequence[f_min : f_max + 1]
                    
                    # Calcul du seuil absolu pour CETTE bande (ex: 90% de son maximum)
                    seuil_coupure = np.max(segment) * seuil_rolloff
                    
                    # On cherche tous les pixels de la bande qui sont au-dessus du seuil
                    indices_valides = np.where(segment >= seuil_coupure)[0]
                    
                    if len(indices_valides) > 0:
                        # On redéfinit les bords sur le premier et le dernier pixel valide
                        nouveau_f_min = f_min + indices_valides[0]
                        nouveau_f_max = f_min + indices_valides[-1]
                        bandes_trimmees.append((nouveau_f_min, nouveau_f_max))
                    else:
                        # Fallback (ne devrait mathématiquement pas arriver)
                        bandes_trimmees.append((f_min, f_max))
                else:
                    bandes_trimmees.append((f_min, f_max))
                    
            signaux_dans_fenetre = bandes_trimmees
            
        else:
            signaux_dans_fenetre = []
            
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


params = {
    'offset': 0,
    'duration': 2_000_000,
    'fs': 1.0,
    'img_height': 512,
    'points_per_window': 1_000_000,
    'f_min': 0.005,
    'f_max': 0.5,
    'wavelet': "cmor100.0-1.0" , #"cmor100.0-1.0"  'fbsp10-0.01-2'

    'transform': 'cwt',      # cwt_rc ou cwt

    'rc_fc': 1.0,            # exemple (dans [0, 0.5] si fs=1)
    'rc_B': 0.12,            # bande utile
    'rc_beta': 0.25,         # roll-off
    

    'detect_db_range': 28, # réglage détection

    'detect_kernel': (200, 2), 
    'downsample_factor': 500,

    'saveRaw': False,

    'addPrediction': False,
}

if __name__ == "__main__":
    file_path = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/debug.png"
    compressed_spec = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    filename = file_path.split("/")[-1].replace(".png", "")
    spectrogramme_moyen = moyenne_temporelle_spectrogramme(compressed_spec, delta_t=100)

    i = 20
    intensite_lissage = 0.015
    delta_t = 100
   


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
        chemin_sortie=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/test_detection/{filename}_fenetre_{delta_t}_{i}_lisse_{intensite_lissage}.png"
    )  
    gt_boxes_pixels = []

    img_h = compressed_spec.shape[0]  # out_h_px (1500 par défaut)
    ds = 500
    scale_y = img_h / params['img_height']

    meta = loaders.load_metadata("/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta")

    gt_boxes_pixels = []
    if meta:
        for ann in meta.get("annotations", []):
            if ann['core:sample_start'] < 2_000_000:  

                y_start = dsp.freq_to_pixel_linear(ann['core:freq_upper_edge'], params['img_height'], params['f_max'])
                y_end = dsp.freq_to_pixel_linear(ann['core:freq_lower_edge'], params['img_height'], params['f_max'])

                # Sécurité bornes (dans le repère original)
                if y_start < 0: y_start = 0
                if y_end > params['img_height']: y_end = params['img_height']

                x = ann['core:sample_start']
                w = min(ann['core:sample_count'], params['duration'] - x)
                h = y_end - y_start

                # Conversion vers le repère de l'image compressée
                cx = x / ds
                cw = w / ds
                cy = y_start * scale_y
                ch = h * scale_y

                gt_boxes_pixels.append((cx, cy, cw, ch))

    
    roll_off_threshold = 0.30
    detections_list = detecter_signaux_robustes(spectrogramme_lisse, seuil_rolloff=roll_off_threshold)

    detection_affinees = affiner_bordures_temporelles(compressed_spec, detections_list, delta_t, seuil_t=20, lissage_t=0.1)
    

    boxes = fusionner_detections_precises(detection_affinees, delta_t)

    sauvegarder_visualisation_avec_boxes(boxes=boxes,
                                        gt_boxes_pixels=gt_boxes_pixels,
                                        image=compressed_spec,
                                        spectrogramme_moyen=spectrogramme_lisse,
                                        delta_t=delta_t,
                                        i=i,
                                        chemin_sortie=f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/test_detection/{filename}_fenetre_with_boxes_roth_{roll_off_threshold}.png"
                                        )

    out = draw_boxes(compressed_spec, boxes)
    cv2.imwrite(f"/home/antoine/Documents/ICE/projet/wavelet_ice/data/test_detection/{filename}_detections_fusionnees.png", out)



