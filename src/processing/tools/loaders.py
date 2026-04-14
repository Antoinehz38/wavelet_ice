import os
import json
import numpy as np


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_iq_data(filepath, num_samples, offset=0):
    """Load raw I/Q samples (complex64).

    Args:
        filepath: Path to the .sigmf-data file.
        num_samples: Number of samples to read.
        offset: Offset in samples (not bytes).
    """
    try:
        offset_bytes = offset * np.dtype(np.complex64).itemsize
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset_bytes)
        print(f"Data loaded: {len(data)} samples (offset={offset} samples, {offset_bytes} bytes).")
        return data
    except FileNotFoundError:
        print(f"Error: data file not found: {filepath}")
        return None


def load_metadata(filepath):
    """Load JSON metadata file."""
    if not os.path.exists(filepath):
        print(f"Warning: metadata file not found: {filepath}")
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return {}