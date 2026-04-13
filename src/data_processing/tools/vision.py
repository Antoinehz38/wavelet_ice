import cv2
from src.data_processing.tools.detection_helpers import moyenne_temporelle_spectrogramme, creer_et_appliquer_passe_bas 
from src.data_processing.tools.detection_helpers import sauvegarder_visualisation_avec_boxes, detecter_signaux_robustes, affiner_bordures_temporelles, fusionner_detections_precises
                                            
import numpy as np


def detect_box(image, delta_t=100, intensite_lissage=0.015, seuil_detection=25, seuil_affinage=20, lissage_affinage=0.1):
    F, T = image.shape
    delta_t = min(delta_t, T//10)
    spectrogramme_moyen = moyenne_temporelle_spectrogramme(image, delta_t=delta_t)


    spectrogramme_lisse = creer_et_appliquer_passe_bas(
        spectrogramme_moyen, 
        frequence_coupure=intensite_lissage, 
        axe=0
    )

    detections_list = detecter_signaux_robustes(spectrogramme_lisse, proeminence_min=10)

    detection_affinees = affiner_bordures_temporelles(image, detections_list, delta_t, seuil_t=seuil_affinage, lissage_t=lissage_affinage)
    

    boxes = fusionner_detections_precises(detection_affinees, delta_t)

    return boxes
if __name__ == "__main__":
    file_path = "data/test_detection/spectrogram_20260413_150146.png"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    boxes = detect_box(img)

    sauvegarder_visualisation_avec_boxes(
        image=img,
        spectrogramme_moyen=moyenne_temporelle_spectrogramme(img, delta_t=100),
        delta_t=100,
        i=25,
        boxes=boxes,
        chemin_sortie="data/metrics/visu.png"
    )

    print("Boxes détectées :", boxes)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imwrite("data/metrics/detected_boxes_tight.png", out)