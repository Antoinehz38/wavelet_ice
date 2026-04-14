from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from src.processing.tools.loaders import load_metadata


@dataclass
class TimeWindow:
    start: int
    end: int
    active_annotations: List[int]
    descriptions: List[str]

    @property
    def length(self) -> int:
        return self.end - self.start


def _get_centered_bounds(
    pos: int,
    window_size: int,
    global_start: int,
    global_end: int | None
) -> Tuple[int, int]:
    half_size = window_size // 2
    start = max(global_start, pos - half_size)
    end = start + window_size

    if global_end is not None and end > global_end:
        end = global_end
        start = max(global_start, end - window_size)

    return start, end


def _get_intersecting_annotations(
    w_start: int,
    w_end: int,
    annotations: List[Dict[str, Any]]
) -> Tuple[List[int], List[str]]:
    active_indices = []
    descriptions = []

    for idx, ann in enumerate(annotations):
        a_start = int(ann["core:sample_start"])
        a_end = a_start + int(ann["core:sample_count"])

        if a_start < w_end and a_end > w_start:
            active_indices.append(idx)
            descriptions.append(ann.get("core:description", f"ann_{idx}"))

    return active_indices, descriptions


def build_transition_windows(
    annotations: List[Dict[str, Any]],
    window_size: int,
    global_start: int = 0,
    global_end: int | None = None,
) -> List[TimeWindow]:
    if window_size <= 0:
        raise ValueError("window_size must be strictly positive.")

    transition_points: Set[int] = set()

    for ann in annotations:
        start = int(ann["core:sample_start"])
        end = start + int(ann["core:sample_count"])
        transition_points.update([start, end])

    valid_points = [
        p for p in transition_points
        if p >= global_start and (global_end is None or p <= global_end)
    ]

    unique_bounds: Set[Tuple[int, int]] = set()
    for pos in sorted(valid_points):
        bounds = _get_centered_bounds(pos, window_size, global_start, global_end)
        unique_bounds.add(bounds)

    windows: List[TimeWindow] = []
    for w_start, w_end in sorted(unique_bounds):
        active_idx, descriptions = _get_intersecting_annotations(
            w_start, w_end, annotations
        )
        
        if active_idx:
            windows.append(
                TimeWindow(
                    start=w_start,
                    end=w_end,
                    active_annotations=active_idx,
                    descriptions=descriptions,
                )
            )

    return windows


if __name__ == "__main__":
    meta_file = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
    meta = load_metadata(meta_file)
    annotations = meta.get("annotations", []) if meta else []

    PARAMS = {
        'window_size': 1_000_000,
        'offset': 0,
        'duration': 100_000_000,
    }
    
    windows = build_transition_windows(
        annotations=annotations,
        window_size=PARAMS['window_size'],
        global_start=PARAMS['offset'],
        global_end=PARAMS['offset'] + PARAMS['duration'],
    )
    
    print(f"{len(windows)} CWT windows to compute.")
    for i, w in enumerate(windows):
        print(
            f"[{i}] start={w.start}, end={w.end}, len={w.length}, "
            f"active={w.descriptions}"
        )