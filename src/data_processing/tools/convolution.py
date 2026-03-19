import numpy as np
import cv2

import matplotlib.pyplot as plt

from src.data_processing.tools.vision import detect_boxes, simple_binary_th


def convolve2d(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros_like(image, dtype=np.float32)

    for y in range(i_h):
        for x in range(i_w):
            region = padded[y:y+k_h, x:x+k_w]
            out[y, x] = np.sum(region * kernel)

    return out


def normalize_uint8(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max() + 1e-8
    img *= 255
    return img.astype(np.uint8)



   

if __name__ == "__main__":

    input = "/home/antoine/Downloads/spectrogram_20260306_083825.png"
    output = input.replace(".png", "_predict.png")
    image_gray = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(input, cv2.IMREAD_COLOR)


    bboxes = simple_binary_th(image_gray)


    for (x0, y0, w, h) in bboxes:
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

    cv2.imwrite(input.replace(".png", "_bboxes2.png"), image)
    print("bboxes:", bboxes)
    print("saved:", input.replace(".png", "_bboxes2.png"))


       