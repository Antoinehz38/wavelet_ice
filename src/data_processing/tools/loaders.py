import os
import json
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_iq_data(filepath, num_samples, offset=0):
    """Charge les échantillons bruts I/Q (complex64).
    
    Args:
        filepath: chemin vers le fichier .sigmf-data
        num_samples: nombre d'échantillons à lire
        offset: offset en **échantillons** (pas en bytes)
    """
    try:
        # np.fromfile attend un offset en BYTES, pas en échantillons
        offset_bytes = offset * np.dtype(np.complex64).itemsize  # complex64 = 8 bytes
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset_bytes)
        print(f"✅ Data chargée : {len(data)} échantillons (offset={offset} samples, {offset_bytes} bytes).")
        return data
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier data introuvable : {filepath}")
        return None

def load_metadata(filepath):
    """Charge le fichier JSON de métadonnées."""
    if not os.path.exists(filepath):
        print(f"⚠️ Warning : Fichier meta introuvable : {filepath}")
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erreur lecture meta : {e}")
        return {}