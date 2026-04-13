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
        default=None,
        help="Nombre d'échantillons à charger"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Offset in the document"
    )

    parser.add_argument(
        "--transfoType",
        type=str,
        default="cwt",
        help="Type of transformation you want in [cwt, fft, cwt_rc]"
    )

    parser.add_argument(
        "--waveletType",
        type=str,
        default="cmor100.0-1.0",
        help="Type of wavelet for CWT (e.g., 'cmor100.0-1.0' or 'fbsp10-0.01-2')"
    )

    parser.add_argument(
        "--addPrediction",
        type=bool,
        default=False,
        help="Whether to add prediction to the output (True/False)"
    )

    parser.add_argument(
        "--saveRaw",
        type=bool,
        default=False,
        help="Whether to save the raw spectrogram image (without annotations) in the output directory"
    )

    parser.add_argument(
        "--downSizeFactor",
        type= int,
        default = 500,
        help = "to down size the png picture"
    )

    return parser.parse_args()
