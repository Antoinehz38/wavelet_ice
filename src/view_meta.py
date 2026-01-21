import json
import json
import cv2
import numpy as np

if __name__ == "__main__":
    META_PATH = "data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=4))
