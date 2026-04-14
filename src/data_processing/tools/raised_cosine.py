import numpy as np



class RaisedCosineWavelet:

    def __init__(self, fc: float, B: float, beta: float = 0.25):
        """
        Ondelette analytique définie en fréquence (f >= 0).
        Paramètres:
        fc   : fréquence centrale (Hz, dans le domaine normalisé si fs=1)
        B    : largeur de bande utile (Hz)
        beta : roll-off (0 < beta <= 1)
        """
        self.fc: float = fc
        self.B: float = B
        self.beta: float = beta

    @property
    def central_frequency(self) -> float:
        # Pour notre construction, la fréquence centrale est un paramètre.
        return float(self.fc)

    def psi_positive(self, f: np.ndarray) -> np.ndarray:
        """
        Enveloppe raised-cosine band-pass sur f >= 0.
        Retourne Psi(f) (complexe) mais ici amplitude réelle (phase nulle).
        """
        f = np.asarray(f, dtype=float)

        # Bords de bande
        f1 = self.fc - self.B / 2.0
        f2 = self.fc + self.B / 2.0
        # Largeur de transition
        delta = self.beta * self.B / 2.0

        if delta <= 0:
            raise ValueError("beta et B doivent donner delta > 0")
        if f1 - delta <= 0:
            # Important: ondelette "wavelet-like" => éviter DC
            # sinon Psi(0) != 0 et vous aurez beaucoup de basse fréquence.
            raise ValueError("Choisir fc, B, beta tels que (f1 - delta) > 0 pour éviter DC.")

        A = np.zeros_like(f)

        # Zones
        left0 = f <= (f1 - delta)
        leftT = (f > (f1 - delta)) & (f < (f1 + delta))
        mid   = (f >= (f1 + delta)) & (f <= (f2 - delta))
        rightT= (f > (f2 - delta)) & (f < (f2 + delta))
        right0= f >= (f2 + delta)

        # Transition basse: montée cosinus de 0 -> 1
        # x in [0, 1] sur [f1-delta, f1+delta]
        x = (f[leftT] - (f1 - delta)) / (2.0 * delta)
        A[leftT] = 0.5 * (1.0 - np.cos(np.pi * x))

        # Plateau
        A[mid] = 1.0

        # Transition haute: descente cosinus de 1 -> 0
        # x in [0, 1] sur [f2-delta, f2+delta]
        x2 = (f[rightT] - (f2 - delta)) / (2.0 * delta)
        A[rightT] = 0.5 * (1.0 + np.cos(np.pi * x2))

        # Hors bande déjà 0
        A[left0] = 0.0
        A[right0] = 0.0

        # Psi(f) = A(f) (réel) ; ondelette "complexe" car analytique => support unilatéral
        return A.astype(np.complex64)

    def psi(self, f: np.ndarray) -> np.ndarray:
        """
        Psi(f) sur f réel: analytique => 0 pour f<0.
        """
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=np.complex64)
        pos = f >= 0
        out[pos] = self.psi_positive(f[pos])
        out[~pos] = 0.0
        return out

    def psi_scaled_on_grid(self, fgrid: np.ndarray, scale: float) -> np.ndarray:
        """
        Construit Psi_s(fgrid) = sqrt(scale) * Psi(scale * fgrid)
        sur une grille fréquentielle fgrid (Hz), typiquement np.fft.fftfreq(N, d=1/fs).
        """
        f_scaled = scale * fgrid

        # Psi(scale*f) ; analytique par construction
        base = self.psi(f_scaled)

        # facteur de normalisation énergie type CWT
        return (np.sqrt(scale) * base).astype(np.complex64)
    

if __name__ == "__main__":
    rc = RaisedCosineWavelet(fc=0.1, B=0.05, beta=0.25)
    print("Central frequency:", rc.central_frequency)