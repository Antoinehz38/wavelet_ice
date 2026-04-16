import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet detection pipeline")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the .sigmf-data file",
    )

    parser.add_argument(
        "--meta",
        type=Path,
        required=False,
        default=None,
        help="Path to the .sigmf-meta file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=100_000_000,
        help="Number of samples to load",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset in the document",
    )

    parser.add_argument(
        "--transfoType",
        type=str,
        default="cwt",
        help="Type of transformation in [cwt, fft, cwt_rc]",
    )

    parser.add_argument(
        "--waveletType",
        type=str,
        default="cmor220.0-1.0",
        help="Wavelet type for CWT (e.g. 'cmor220.0-1.0' or 'fbsp10-0.01-2')",
    )

    parser.add_argument(
        "--addPrediction",
        type=bool,
        default=False,
        help="Whether to add prediction to the output",
    )

    parser.add_argument(
        "--saveRaw",
        type=bool,
        default=False,
        help="Whether to save the raw spectrogram image (without annotations)",
    )

    parser.add_argument(
        "--downSizeFactor",
        type=int,
        default=250,
        help="Downsample factor for the output image",
    )

    parser.add_argument(
        "--pointsPerWindow",
        type=int,
        default=1_000_000,
        help="Number of points per window for CWT calculation",
    )

    parser.add_argument(
        "--runPipelineOnFolder",
        type=Path,
        default=None,
        help="Input folder containing .sigmf-data files to process (optional, overrides --input)",
    )

    parser.add_argument(
        "--runOnlyAnalyzeOnFolder",
        type=Path,
        default=None,
        help="Input folder containing .sigmf-data files to analyze (optional, overrides --input)",
    )

    parser.add_argument(
        "--dataFolder",
        type=Path,
        default=None,
        help="Input folder containing .sigmf-data files (optional, used for both --runPipelineOnFolder and --runOnlyAnalyzeOnFolder)",
    )

    parser.add_argument(
        "--analyzeEnergyOnFolder",
        type=Path,
        default=None,
        help="Output folder for analysis reports (optional, default is 'reports' in the current directory)",
    )

    return parser.parse_args()
