import numpy as np


class BumpWavelet:
    def __init__(self, fc: float, B: float):
        """
        Ondelette bump analytique définie en fréquence pour f >= 0.

        Paramètres
        ----------
        fc : float
            Fréquence centrale de la bump (Hz, ou fréquence normalisée si fs=1).
        B : float
            Demi-largeur du support fréquentiel.
            Le support vaut alors ]fc - B, fc + B[.

        Remarques
        ---------
        La définition implémentée est :

            Psi(f) = exp(1 - 1 / (1 - ((f - fc)/B)^2))   si |f - fc| < B
                     0                                    sinon

        Cette ondelette est strictement analytique si fc > B,
        car alors tout son support fréquentiel est strictement dans f > 0.
        """
        self.fc: float = fc
        self.B: float = B

        if self.B <= 0:
            raise ValueError("B doit être strictement positif.")
        if self.fc <= 0:
            raise ValueError("fc doit être strictement positif.")
        if self.fc <= self.B:
            raise ValueError(
                "Pour une ondelette strictement analytique, il faut fc > B "
                "(ainsi le support [fc-B, fc+B] reste dans les fréquences positives)."
            )

    @property
    def central_frequency(self) -> float:
        return float(self.fc)

    def psi_positive(self, f: np.ndarray) -> np.ndarray:
        """
        Calcule Psi(f) pour f >= 0.

        Retourne un tableau complexe64, même si l'amplitude est réelle,
        afin de rester compatible avec le reste de la chaîne CWT.
        """
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=np.float32)

        x = (f - self.fc) / self.B
        mask = np.abs(x) < 1.0

        # Formule bump exacte sur le support
        # exp(1 - 1 / (1 - x^2))
        denom = 1.0 - x[mask] ** 2
        out[mask] = np.exp(1.0 - 1.0 / denom).astype(np.float32)

        return out.astype(np.complex64)

    def psi(self, f: np.ndarray) -> np.ndarray:
        """
        Ondelette analytique sur f réel :
        - Psi(f) = 0 pour f < 0
        - Psi(f) = bump pour f >= 0
        """
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=np.complex64)

        pos = f >= 0
        out[pos] = self.psi_positive(f[pos])
        return out

    def psi_scaled_on_grid(self, fgrid: np.ndarray, scale: float) -> np.ndarray:
        """
        Construit Psi_s(fgrid) = sqrt(scale) * Psi(scale * fgrid)

        Paramètres
        ----------
        fgrid : np.ndarray
            Grille fréquentielle en Hz, typiquement np.fft.fftfreq(N, d=1/fs)
        scale : float
            Échelle CWT

        Retour
        ------
        np.ndarray complex64
        """
        f_scaled = scale * np.asarray(fgrid, dtype=float)
        base = self.psi(f_scaled)
        return (np.sqrt(scale) * base).astype(np.complex64)


if __name__ == "__main__":
    bump = BumpWavelet(fc=0.1, B=0.03)
    print("Central frequency:", bump.central_frequency)