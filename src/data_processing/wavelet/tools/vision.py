import cv2
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