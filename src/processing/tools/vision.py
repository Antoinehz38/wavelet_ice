import cv2
import numpy as np

from src.processing.tools.detection_helpers import (
    temporal_mean_spectrogram,
    apply_lowpass_filter,
    save_visualisation_with_boxes,
    detect_robust_signals,
    refine_temporal_borders,
    merge_precise_detections,
)


def detect_box(image, delta_t=100,
               smoothing_intensity=0.015,
               detection_threshold=25,
               refinement_threshold=5,
               refinement_smoothing=0.1,
               roll_off_threshold=0.10):
    F, T = image.shape
    delta_t = min(delta_t, T // 10)
    mean_spec = temporal_mean_spectrogram(image, delta_t=delta_t)

    smoothed_spec = apply_lowpass_filter(
        mean_spec,
        cutoff_freq=smoothing_intensity,
        axis=0,
    )

    detections_list = detect_robust_signals(smoothed_spec, min_prominence=10, rolloff_threshold=roll_off_threshold)

    refined = refine_temporal_borders(image, detections_list, delta_t, time_threshold=refinement_threshold, time_smoothing=refinement_smoothing)

    boxes = merge_precise_detections(refined, smoothed_spectrogram=smoothed_spec, tolerance=15)

    return boxes


if __name__ == "__main__":
    file_path = "data/test_detection/spectrogram_20260413_150146.png"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    boxes = detect_box(img)

    save_visualisation_with_boxes(
        image=img,
        mean_spectrogram=temporal_mean_spectrogram(img, delta_t=100),
        delta_t=100,
        i=25,
        boxes=boxes,
        output_path="data/metrics/visu.png",
    )

    print("Detected boxes:", boxes)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("data/metrics/detected_boxes_tight.png", out)