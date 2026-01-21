import numpy as np
import cv2

def mark_pixels_above_noise_energy(gray, x_percent=10.0, out_bgr=True):
    """
    gray: image N&B (uint8 ou float), shape (H,W)
    x_percent: pourcentage des pixels les plus faibles utilisés pour estimer le bruit
    Renvoie: (overlay_bgr, noise_energy, threshold)
      - overlay_bgr: image couleur avec pixels > seuil en bleu
      - noise_energy: énergie (somme des carrés) des x% plus faibles
      - threshold: valeur seuil (quantile x%)
    """
    g = gray.astype(np.float32)
    vals = g.reshape(-1)

    # seuil = quantile correspondant aux x% plus faibles
    x = np.clip(float(x_percent), 0.0, 100.0)
    threshold = float(np.percentile(vals, x))


    noise_mask = g >= threshold
    noise_energy = float(np.sum(g[noise_mask] ** 2))

    # sortie couleur + marquage en bleu des pixels au-dessus du seuil
    if out_bgr:
        overlay = cv2.cvtColor(gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR)
        above = ~noise_mask
        overlay[above] = (255, 0, 0)  # bleu en BGR
    else:
        overlay = None

    return overlay, noise_energy, threshold

def extract_dark_zones(img_bgr, blur_ksize=7, min_area=200, blue=(255,0,0)):
    """
    Naif: lisse -> seuil inverse (zones sombres) -> nettoyage morpho -> supprime petites taches -> overlay bleu
    Retour: (mask, overlay)
      - mask: uint8 {0,255} des zones sombres
      - overlay: image BGR avec zones sombres en bleu
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) débruitage léger
    g_blur = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)

    # 2) seuil inverse automatique (Otsu) => sombre = 255
    _, mask = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) nettoyage (enlève bruit, remplit un peu)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 5) overlay bleu
    overlay = img_bgr.copy()
    overlay[mask == 255] = blue  # bleu en BGR
    return mask, overlay



if __name__ == "__main__":
    img = cv2.imread("data/baseline/spectrogram_4096x512.png", cv2.IMREAD_GRAYSCALE)
    overlay, noise_energy, threshold = mark_pixels_above_noise_energy(img, x_percent=90.0, out_bgr=True)

    mask, overlay2 = extract_dark_zones(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), blur_ksize=7, blue=(0,0,255))
    cv2.imwrite("data/baseline/spectrogram_marked2big.png", overlay2)
    print(f"Noise energy (90% lowest): {noise_energy:.2f}, threshold: {threshold:.2f}")
    cv2.imwrite("data/baseline/spectrogram_markedbig.png", overlay)
