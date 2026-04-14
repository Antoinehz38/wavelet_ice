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


def _resize_time_axis(spec: np.ndarray, target_width: int) -> np.ndarray:
    """
    Redimensionne uniquement l'axe temporel d'une matrice 2D (freq, temps).
    """
    if spec.ndim != 2:
        raise ValueError("spec doit être une matrice 2D (freq, temps).")

    h, w = spec.shape
    if w == target_width:
        return spec

    if h <= 0 or w <= 0:
        raise ValueError("Dimensions invalides pour le redimensionnement.")

    return zoom(spec, (1.0, target_width / w), order=1)


def _power_to_db(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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

    Sortie :
    - haut : fréquences positives
    - milieu : 0 Hz
    - bas : fréquences négatives

    La largeur finale est ramenée à len(iq_data) pour rester compatible
    avec un pipeline où l'axe horizontal correspond aux échantillons.
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

    target_width = len(iq_data)

    if nfft is None:
        nfft = nperseg

    if noverlap >= nperseg:
        raise ValueError("noverlap doit être strictement inférieur à nperseg.")

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

    # Spectre centré : fréquences croissantes de -fs/2 à +fs/2
    freqs = np.fft.fftshift(freqs)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Garde la bande utile
    band_mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
    freqs_band = freqs[band_mask]
    Zxx_band = Zxx[band_mask, :]

    if freqs_band.size == 0:
        raise ValueError(
            f"Aucune fréquence dans la bande demandée : "
            f"[{-f_max}, {-f_min}] U [{f_min}, {f_max}]."
        )

    power = np.abs(Zxx_band) ** 2
    spec_db = _power_to_db(power)

    # On veut positif en haut, négatif en bas
    spec_db = np.flipud(spec_db)

    # Resize fréquence puis temps
    spec_db = _resize_freq_axis(spec_db, total_height)
    spec_db = _resize_time_axis(spec_db, target_width)

    return spec_db