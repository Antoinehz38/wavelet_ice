import numpy as np
import dtcwt


def _resize_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x).squeeze()
    if len(x) == target_len:
        return x
    if len(x) == 0:
        return np.zeros(target_len, dtype=float)

    old_idx = np.linspace(0.0, 1.0, len(x))
    new_idx = np.linspace(0.0, 1.0, target_len)

    if np.iscomplexobj(x):
        real = np.interp(new_idx, old_idx, np.real(x))
        imag = np.interp(new_idx, old_idx, np.imag(x))
        return real + 1j * imag

    return np.interp(new_idx, old_idx, x)


def _normalize_row_db(row_db: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """
    Normalisation visuelle par bande.
    0 dB = maximum de la bande.
    On tronque à -top_db.
    """
    row_db = row_db - np.max(row_db)
    row_db = np.clip(row_db, -top_db, 0.0)
    return row_db


def _allocate_dyadic_heights(total_height: int, nlevels: int, include_approx: bool = False):
    """
    Attribue à chaque niveau une hauteur proportionnelle à sa largeur fréquentielle.
    D1: 1/2, D2: 1/4, ..., DL: 1/2^L
    Optionnellement, A_L: 1/2^L
    """
    weights = [1 / (2 ** k) for k in range(1, nlevels + 1)]  # D1, D2, ..., DL
    if include_approx:
        weights.append(1 / (2 ** nlevels))  # A_L

    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    heights = np.floor(weights * total_height).astype(int)
    heights = np.maximum(heights, 1)

    diff = total_height - int(np.sum(heights))
    while diff != 0:
        if diff > 0:
            idx = np.argmax(weights)
            heights[idx] += 1
            diff -= 1
        else:
            idx = np.argmax(heights)
            if heights[idx] > 1:
                heights[idx] -= 1
                diff += 1
            else:
                break

    return heights


def _dtcwt_forward_complex_signal(x, nlevels, biort, qshift):
    """
    Applique la DT-CWT 1D à un signal complexe x en utilisant l'API réelle :
      - DT-CWT sur Re(x)
      - DT-CWT sur Im(x)
    puis recombinaison complexe des sous-bandes.

    Retourne une liste [W1, W2, ..., WL] avec Wk vecteur complexe.
    """
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Le signal doit être 1D, reçu shape={x.shape}")

    xr = np.real(x)
    xi = np.imag(x)

    tfm = dtcwt.Transform1d(biort=biort, qshift=qshift)

    # La doc indique que Transform1d.forward attend une entrée réelle 1D/2D
    pyr_r = tfm.forward(xr, nlevels=nlevels)
    pyr_i = tfm.forward(xi, nlevels=nlevels)

    # La doc indique que highpasses est un tuple de sous-bandes complexes,
    # de l'échelle la plus fine à la plus grossière.
    if len(pyr_r.highpasses) != len(pyr_i.highpasses):
        raise RuntimeError("Incohérence entre les pyramides DT-CWT réelle et imaginaire.")

    bands = []
    for level_idx, (wr, wi) in enumerate(zip(pyr_r.highpasses, pyr_i.highpasses), start=1):
        wr = np.asarray(wr).squeeze()
        wi = np.asarray(wi).squeeze()

        if wr.ndim != 1 or wi.ndim != 1:
            raise ValueError(
                f"Sous-bande DT-CWT non 1D au niveau {level_idx}: "
                f"wr.shape={wr.shape}, wi.shape={wi.shape}"
            )

        # recombinaison IQ simple
        w = wr + 1j * wi
        bands.append(w)

    return bands


def _build_dyadic_half_image(
    bands,
    half_height: int,
    signal_len: int,
    top_db_per_band: float = 40.0,
    resize_to_signal_len: bool = True,
    pad_value: float | None = None,
):
    """
    Construit une demi-image dyadique à partir d'une liste de sous-bandes complexes.

    Si resize_to_signal_len=True :
        chaque niveau est interpolé à signal_len.

    Si resize_to_signal_len=False :
        chaque niveau garde sa longueur native, et on ajoute un padding à droite
        pour pouvoir empiler les bandes dans une même image.
    """
    nlevels = len(bands)
    heights = _allocate_dyadic_heights(half_height, nlevels, include_approx=False)

    if pad_value is None:
        pad_value = -top_db_per_band

    rows = []
    target_widths = []

    # 1) déterminer les largeurs cibles
    for w in bands:
        power = np.abs(w) ** 2
        if resize_to_signal_len:
            target_widths.append(signal_len)
        else:
            target_widths.append(len(power))

    max_width = max(target_widths) if len(target_widths) > 0 else 0

    # 2) construire chaque bande
    for w, band_h in zip(bands, heights):
        power = np.abs(w) ** 2

        if resize_to_signal_len:
            power_disp = _resize_1d(power, signal_len)
        else:
            power_disp = power

        row_db = 10.0 * np.log10(power_disp + 1e-12)
        row_db = _normalize_row_db(row_db, top_db=top_db_per_band)

        band_img = np.tile(row_db[None, :], (band_h, 1))

        # 3) padding horizontal à droite si nécessaire
        cur_width = band_img.shape[1]
        if cur_width < max_width:
            pad = np.full((band_h, max_width - cur_width), pad_value, dtype=band_img.dtype)
            band_img = np.hstack((band_img, pad))

        rows.append(band_img)

    # 4) empilement vertical
    half_spec = np.vstack(rows)

    # sécurité sur la hauteur
    if half_spec.shape[0] != half_height:
        if half_spec.shape[0] > half_height:
            half_spec = half_spec[:half_height, :]
        else:
            pad = np.full((half_height - half_spec.shape[0], half_spec.shape[1]), pad_value, dtype=half_spec.dtype)
            half_spec = np.vstack((half_spec, pad))

    return half_spec


def compute_dual_dtcwt_scalogram_dyadic(
    iq_data,
    total_height: int,
    nlevels: int = 8,
    biort: str = "near_sym_a",
    qshift: str = "qshift_a",
    top_db_per_band: float = 40.0,
    resize_to_signal_len: bool = True,
):
    """
    Visualisation dyadique DT-CWT double bande :
      - moitié haute  : DT-CWT sur x
      - moitié basse : DT-CWT sur conj(x), retournée verticalement

    Remarque importante :
      - ce n'est PAS une vraie carte temps-fréquence linéaire ;
      - chaque niveau correspond à une sous-bande dyadique ;
      - l'axe vertical ne doit pas être annoté comme un axe fréquentiel linéaire.
    """
    x = np.asarray(iq_data).squeeze()
    if x.ndim != 1:
        raise ValueError(f"iq_data doit être 1D, reçu shape={x.shape}")

    signal_len = len(x)
    half_height = total_height // 2

    # 1) moitié haute : x
    bands_pos = _dtcwt_forward_complex_signal(
        x=x,
        nlevels=nlevels,
        biort=biort,
        qshift=qshift,
    )
    spec_pos = _build_dyadic_half_image(
    bands=bands_pos,
    half_height=half_height,
    signal_len=signal_len,
    top_db_per_band=top_db_per_band,
    resize_to_signal_len=resize_to_signal_len,
    )

    # 2) moitié basse : conj(x)
    bands_neg = _dtcwt_forward_complex_signal(
        x=np.conj(x),
        nlevels=nlevels,
        biort=biort,
        qshift=qshift,
    )
    

    spec_neg = _build_dyadic_half_image(
        bands=bands_neg,
        half_height=half_height,
        signal_len=signal_len,
        top_db_per_band=top_db_per_band,
        resize_to_signal_len=resize_to_signal_len,
    )

    # 3) assemblage : positif en haut, négatif en bas retourné
    full_spec = np.vstack((spec_pos, np.flipud(spec_neg)))

    # si total_height est impair, on complète d'une ligne
    if full_spec.shape[0] < total_height:
        pad = np.full((total_height - full_spec.shape[0], full_spec.shape[1]), -top_db_per_band)
        full_spec = np.vstack((full_spec, pad))
    elif full_spec.shape[0] > total_height:
        full_spec = full_spec[:total_height, :]

    return full_spec