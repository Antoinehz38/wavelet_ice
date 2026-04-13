import cv2
from src.data_processing.tools.test import moyenne_temporelle_spectrogramme, creer_et_appliquer_passe_bas, detecter_signaux_toutes_fenetres, affiner_bordures_temporelles, fusionner_detections_precises
                                            
import numpy as np


def detect_box(image, delta_t=100, intensite_lissage=0.015, seuil_detection=25, seuil_affinage=20, lissage_affinage=0.1):
    spectrogramme_moyen = moyenne_temporelle_spectrogramme(image, delta_t=delta_t)


    spectrogramme_lisse = creer_et_appliquer_passe_bas(
        spectrogramme_moyen, 
        frequence_coupure=intensite_lissage, 
        axe=0
    )

    detections_list = detecter_signaux_toutes_fenetres(spectrogramme_lisse, seuil=seuil_detection)

    detection_affinees = affiner_bordures_temporelles(image, detections_list, delta_t, seuil_t=seuil_affinage, lissage_t=lissage_affinage)
    

    boxes = fusionner_detections_precises(detection_affinees, delta_t)

    return boxes
if __name__ == "__main__":
    file_path = "data/metrics/spectrogram_20260305_103527.png"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    boxes = detect_box(img)
    print("Boxes détectées :", boxes)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imwrite("data/metrics/detected_boxes_tight.png", out)