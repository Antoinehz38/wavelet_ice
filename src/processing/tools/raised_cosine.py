import numpy as np


class RaisedCosineWavelet:

    def __init__(self, fc: float, B: float, beta: float = 0.25):
        """Analytic wavelet defined in the frequency domain (f >= 0).

        Args:
            fc: Centre frequency (Hz, normalised if fs=1).
            B: Useful bandwidth (Hz).
            beta: Roll-off factor (0 < beta <= 1).
        """
        self.fc: float = fc
        self.B: float = B
        self.beta: float = beta

    @property
    def central_frequency(self) -> float:
        return float(self.fc)

    def psi_positive(self, f: np.ndarray) -> np.ndarray:
        """Raised-cosine band-pass envelope for f >= 0.

        Returns Psi(f) as a complex array (real amplitude, zero phase).
        """
        f = np.asarray(f, dtype=float)

        f1 = self.fc - self.B / 2.0
        f2 = self.fc + self.B / 2.0
        delta = self.beta * self.B / 2.0

        if delta <= 0:
            raise ValueError("beta and B must yield delta > 0")
        if f1 - delta <= 0:
            raise ValueError("Choose fc, B, beta such that (f1 - delta) > 0 to avoid DC.")

        A = np.zeros_like(f)

        left0 = f <= (f1 - delta)
        leftT = (f > (f1 - delta)) & (f < (f1 + delta))
        mid = (f >= (f1 + delta)) & (f <= (f2 - delta))
        rightT = (f > (f2 - delta)) & (f < (f2 + delta))
        right0 = f >= (f2 + delta)

        x = (f[leftT] - (f1 - delta)) / (2.0 * delta)
        A[leftT] = 0.5 * (1.0 - np.cos(np.pi * x))

        A[mid] = 1.0

        x2 = (f[rightT] - (f2 - delta)) / (2.0 * delta)
        A[rightT] = 0.5 * (1.0 + np.cos(np.pi * x2))

        A[left0] = 0.0
        A[right0] = 0.0

        return A.astype(np.complex64)

    def psi(self, f: np.ndarray) -> np.ndarray:
        """Psi(f) over real f: analytic => 0 for f < 0."""
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=np.complex64)
        pos = f >= 0
        out[pos] = self.psi_positive(f[pos])
        out[~pos] = 0.0
        return out

    def psi_scaled_on_grid(self, fgrid: np.ndarray, scale: float) -> np.ndarray:
        """Build Psi_s(fgrid) = sqrt(scale) * Psi(scale * fgrid).

        Args:
            fgrid: Frequency grid (Hz), typically np.fft.fftfreq(N, d=1/fs).
            scale: CWT scale.
        """
        f_scaled = scale * fgrid
        base = self.psi(f_scaled)
        return (np.sqrt(scale) * base).astype(np.complex64)


if __name__ == "__main__":
    rc = RaisedCosineWavelet(fc=0.1, B=0.05, beta=0.25)
    print("Central frequency:", rc.central_frequency)