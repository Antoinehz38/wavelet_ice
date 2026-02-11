import argparse
from pathlib import Path

from sympy import false


def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet detection pipeline")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin vers le fichier .sigmf-data"
    )

    parser.add_argument(
        "--meta",
        type=Path,
        required=False,
        default=None,
        help="Chemin vers le fichier .sigmf-meta"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dossier de sortie"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20_000,
        help="Nombre d'échantillons à charger"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset in the document"
    )

    parser.add_argument(
        "--transfoType",
        type=str,
        default="morlet",
        help="Type of transformation you want in [morlet, fft]"
    )

    return parser.parse_args()
