import numpy as np


class MorseWavelet:
    def __init__(self, beta: float, gamma: float):
        """
        Ondelette de Morse analytique définie en fréquence :

            Psi_{beta,gamma}(omega)
            = U(omega) * a_{beta,gamma} * omega^beta * exp(-omega^gamma)

        avec omega = 2*pi*f dans cette implémentation.

        Paramètres
        ----------
        beta : float
            Paramètre de forme (> 0)
        gamma : float
            Paramètre de forme (> 0)

        Remarques
        ---------
        Le maximum spectral est atteint pour :
            omega_peak = (beta / gamma)^(1/gamma)

        On normalise ici pour que Psi(omega_peak) = 1.
        """
        self.beta = float(beta)
        self.gamma = float(gamma)

        if self.beta <= 0:
            raise ValueError("beta doit être strictement positif.")
        if self.gamma <= 0:
            raise ValueError("gamma doit être strictement positif.")

        self._omega_peak = (self.beta / self.gamma) ** (1.0 / self.gamma)
        self._f_peak = self._omega_peak / (2.0 * np.pi)

        # Normalisation : Psi(omega_peak) = 1
        # log(a) = -beta*log(omega_peak) + omega_peak^gamma
        self._log_a = -self.beta * np.log(self._omega_peak) + (self._omega_peak ** self.gamma)
        self._a = float(np.exp(self._log_a))

    @property
    def central_frequency(self) -> float:
        """
        Fréquence centrale de l'ondelette mère (pic spectral),
        en Hz si fs est en Hz, ou fréquence normalisée si fs=1.
        """
        return float(self._f_peak)

    @property
    def omega_peak(self) -> float:
        return float(self._omega_peak)

    @property
    def normalization(self) -> float:
        return float(self._a)

    def psi_positive(self, f: np.ndarray) -> np.ndarray:
        """
        Calcule Psi(f) pour f >= 0 de manière numériquement stable.
        """
        f = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f

        out = np.zeros_like(omega, dtype=np.float64)
        mask = omega > 0.0

        if np.any(mask):
            om = omega[mask]

            # Calcul stable en log :
            # log(Psi) = log(a) + beta*log(omega) - omega^gamma
            log_psi = self._log_a + self.beta * np.log(om) - (om ** self.gamma)

            # Evite overflow/underflow dans exp
            # exp(709) ~ limite float64
            log_psi = np.clip(log_psi, -745.0, 80.0)

            out[mask] = np.exp(log_psi)

        return out.astype(np.complex64)

    def psi(self, f: np.ndarray) -> np.ndarray:
        """
        Ondelette analytique :
        - Psi(f) = 0 pour f < 0
        - Psi(f) = Morse pour f >= 0
        """
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=np.complex64)

        pos = f >= 0
        out[pos] = self.psi_positive(f[pos])
        return out

    def psi_scaled_on_grid(self, fgrid: np.ndarray, scale: float) -> np.ndarray:
        """
        Construit :
            Psi_s(fgrid) = sqrt(scale) * Psi(scale * fgrid)
        """
        f_scaled = scale * np.asarray(fgrid, dtype=float)
        base = self.psi(f_scaled)
        return (np.sqrt(scale) * base).astype(np.complex64)


if __name__ == "__main__":
    morse = MorseWavelet(beta=180.0, gamma=3.0)
    print("Central frequency:", morse.central_frequency)
    print("Omega peak:", morse.omega_peak)
    print("Normalization:", morse.normalization)