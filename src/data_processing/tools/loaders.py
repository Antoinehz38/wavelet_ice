import os
import json
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_iq_data(filepath, num_samples, offset=0):
    """Charge les échantillons bruts I/Q (complex64)."""
    try:
        data = np.fromfile(filepath, dtype=np.complex64, count=num_samples, offset=offset)
        print(f"✅ Data chargée : {len(data)} échantillons.")
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