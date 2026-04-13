import cv2
from src.data_processing.tools.test import detect_signals_by_projections, tighten_box_2d,  tighten_box_with_energy
                                            
import numpy as np

def detect_boxes(spectrogram_db, min_db_range=30, morph_kernel_size=(200, 2)):
    """
    Détecte les zones d'énergie avec une forte cohésion horizontale.
    """
    # 1. Clipping
    v_max = np.max(spectrogram_db)
    threshold = v_max - min_db_range
    
    img_clean = spectrogram_db.copy()
    img_clean[img_clean < threshold] = threshold
    
    # 2. Normalisation 0-255
    norm_img = (img_clean - threshold) / (v_max - threshold) * 255
    norm_img = norm_img.astype(np.uint8)
    
    # 3. Prétraitement : FLOU GAUSSIEN RENFORCÉ
    # Un noyau (7, 7) ou (9, 9) va "baver" les pixels ensemble avant même le seuillage.
    # Cela réduit drastiquement la sensibilité à la variance locale.
    blur = cv2.GaussianBlur(norm_img, (7, 7), 0)
    
    # 4. Binarisation OTSU
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphologie (La "Super Colle")
    # On force la fusion horizontale.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    
    # Closing = Dilate (bouche les trous) + Erode (restore la taille)
    # On fait 2 itérations pour être sûr de bien souder les blocs fragmentés
    final_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 6. Extraction
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtre anti-bruit : 
        # On ignore les boîtes trop petites (moins de 100px de large ou 3px de haut)
        if w > 100 and h > 3:
            detected_boxes.append((x, y, w, h))
            
    return detected_boxes, final_mask

def simple_binary_th(gray_image, min_area=50):
    _, bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    bboxes = []
    H, W = bw.shape
    for i in range(1, num):
        x0, y0, w, h, area = stats[i]
        if area < min_area:
            continue
        # filtres optionnels (souvent utiles)
        if w < 10 or h < 10:
            continue
        if w * h > 0.8 * H * W:
            continue
        bboxes.append((x0, y0, w, h))

    return bboxes

def binary_th_v2(gray_image, min_area=100):
    H, W = gray_image.shape

    # 1) suppression du fond
    bg = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=15, sigmaY=15)
    enh = cv2.subtract(gray_image, bg)

    # 2) normalisation optionnelle
    enh = cv2.normalize(enh, None, 0, 255, cv2.NORM_MINMAX)

    # 3) seuillage statistique
    mu = enh.mean()
    sigma = enh.std()
    bw = (enh > mu + 2.0 * sigma).astype(np.uint8) * 255

    # 4) morphologie
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k1)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k2)

    # 5) composantes connexes
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    bboxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue
        if w < 8 or h < 4:
            continue
        if w * h == 0:
            continue

        density = area / float(w * h)
        aspect = w / float(h)

        # filtres raisonnables mais pas trop agressifs
        if density < 0.15:
            continue
        if w * h > 0.9 * H * W:
            continue

        bboxes.append((x, y, w, h, area, density, aspect))

    return bw, bboxes


if __name__ == "__main__":
    file_path = "data/metrics/spectrogram_20260305_103527.png"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    boxes, debug = detect_signals_by_projections(
        img
    )
    z = debug["z"]

    tight_boxes = [
        tighten_box_with_energy(
            z,
            box,
            qx=(0.02, 0.98),
            qy=(0.02, 0.98),
            pad_x=1,
            pad_y=1
        )
        for box in boxes
    ]

    tight_boxes_2 = [
        tighten_box_2d(
            z,
            box,
            
        )
        for box in boxes
    ]
       

  
    # Affichage pour vérification
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color_2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for (x, y, w, h) in tight_boxes:
    #     cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # for (x, y, w, h) in boxes:
    #     cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    for (x, y, w, h) in tight_boxes_2:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imwrite("data/metrics/detected_boxes_tight.png", img_color)
