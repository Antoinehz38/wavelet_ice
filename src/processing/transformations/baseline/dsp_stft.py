import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom


def _resize_freq_axis(spec: np.ndarray, target_height: int) -> np.ndarray:
    """
    Redimensionne uniquement l'axe fréquentiel d'une matrice 2D (freq, temps).
    """
    if spec.ndim != 2:
        raise ValueError("spec doit être une matrice 2D (freq, temps).")

    h, w = spec.shape
    if h == target_height:
        return spec

    if h <= 0 or w <= 0:
        raise ValueError("Dimensions invalides pour le redimensionnement.")

    return zoom(spec, (target_height / h, 1.0), order=1)


def _power_to_db(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convertit une puissance linéaire en décibels.
    """
    return 10.0 * np.log10(np.maximum(power, eps))


def compute_stft_scalogram(
    iq_data: np.ndarray,
    total_height: int,
    fs: float,
    f_min: float,
    f_max: float,
    nperseg: int = 256,
    noverlap: int = 192,
    nfft: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Calcule un spectrogramme STFT complet avec un seul calcul.

    Convention de sortie :
    - haut de l'image   : fréquences positives les plus élevées (+f_max)
    - milieu            : 0 Hz
    - bas de l'image    : fréquences négatives les plus basses (-f_max)

    La bande conservée est [-f_max, -f_min] U [+f_min, +f_max], avec la zone
    centrale autour de 0 naturellement incluse si présente dans la discrétisation.

    Retour :
        matrice 2D (total_height, temps) en dB
    """
    if total_height < 2:
        raise ValueError("total_height doit être >= 2.")

    if fs <= 0:
        raise ValueError("fs doit être strictement positif.")

    if not (0 <= f_min < f_max <= fs / 2):
        raise ValueError(
            f"Il faut respecter 0 <= f_min < f_max <= fs/2. "
            f"Ici : f_min={f_min}, f_max={f_max}, fs/2={fs/2}."
        )

    iq_data = np.asarray(iq_data)
    if iq_data.ndim != 1:
        raise ValueError("iq_data doit être un vecteur 1D.")

    if nfft is None:
        nfft = nperseg

    if noverlap >= nperseg:
        raise ValueError("noverlap doit être strictement inférieur à nperseg.")

    # STFT complète : fréquences positives et négatives
    freqs, _, Zxx = stft(
        iq_data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    # Recentrage fréquentiel : [-fs/2, ..., 0, ..., +fs/2]
    freqs = np.fft.fftshift(freqs)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Garde uniquement la bande utile signée
    band_mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
    freqs_band = freqs[band_mask]
    Zxx_band = Zxx[band_mask, :]

    if freqs_band.size == 0:
        raise ValueError(
            f"Aucune fréquence dans la bande demandée : "
            f"[{-f_max}, {-f_min}] U [{f_min}, {f_max}]."
        )

    # On veut : positif en haut, négatif en bas
    # Après fftshift, l'ordre est croissant : négatif -> positif
    # Donc on inverse l'axe vertical.
    power = np.abs(Zxx_band) ** 2
    spec_db = _power_to_db(power)
    spec_db = np.flipud(spec_db)

    # Redimensionnement à la hauteur voulue
    spec_db = _resize_freq_axis(spec_db, total_height)

    return spec_db