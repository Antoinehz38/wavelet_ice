from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.processing.tools.loaders import load_metadata


@dataclass
class TimeWindow:
    start: int
    end: int
    length: int
    active_annotations: List[int]
    descriptions: List[str]


def _clip_interval(start: int, end: int, global_start: int, global_end: int) -> Tuple[int, int] | None:
    """Clip [start, end) to the global interval [global_start, global_end).

    Returns None if the intersection is empty.
    """
    s = max(start, global_start)
    e = min(end, global_end)
    if e <= s:
        return None
    return s, e


def build_overlap_segments(
    annotations: List[Dict[str, Any]],
    global_start: int = 0,
    global_end: int | None = None,
) -> List[Dict[str, Any]]:
    """Build time segments where the set of active annotations is constant.

    Example:
      if A=[10,50), B=[30,80)
      => segments:
         [10,30) : {A}
         [30,50) : {A,B}
         [50,80) : {B}

    Parameters
    ----------
    annotations : list of dict
        List of SigMF annotations.
    global_start : int
        Start of the temporal domain to consider.
    global_end : int | None
        End of the temporal domain. If None, uses the max annotation end.

    Returns
    -------
    segments : list of dict
        Each segment contains:
        - start
        - end
        - active_annotations: indices of active annotations
        - descriptions: associated descriptions
    """
    events: List[Tuple[int, int, int]] = []
    # format event: (sample_index, kind, ann_idx)
    # kind = +1 => start, kind = -1 => end

    max_end_seen = global_start

    for ann_idx, ann in enumerate(annotations):
        start = int(ann["core:sample_start"])
        count = int(ann["core:sample_count"])
        end = start + count

        max_end_seen = max(max_end_seen, end)

        if global_end is None:
            clipped = _clip_interval(start, end, global_start, end)
        else:
            clipped = _clip_interval(start, end, global_start, global_end)

        if clipped is None:
            continue

        s, e = clipped
        events.append((s, +1, ann_idx))
        events.append((e, -1, ann_idx))

    if not events:
        return []

    if global_end is None:
        global_end = max_end_seen

    # Sort: at equal time, process ends before starts to respect [start, end)
    events.sort(key=lambda x: (x[0], x[1]))

    active = set()
    segments: List[Dict[str, Any]] = []

    i = 0
    current_pos = events[0][0]

    while i < len(events):
        pos = events[i][0]

        # Previous segment if annotations are active
        if pos > current_pos and active:
            active_sorted = sorted(active)
            segments.append({
                "start": current_pos,
                "end": pos,
                "active_annotations": active_sorted,
                "descriptions": [
                    annotations[idx].get("core:description", f"ann_{idx}")
                    for idx in active_sorted
                ],
            })

        # Consume all events at the same position
        same_pos_events = []
        while i < len(events) and events[i][0] == pos:
            same_pos_events.append(events[i])
            i += 1

        # Ends first (kind = -1), then starts (kind = +1)
        for _, kind, ann_idx in same_pos_events:
            if kind == -1:
                active.discard(ann_idx)
        for _, kind, ann_idx in same_pos_events:
            if kind == +1:
                active.add(ann_idx)

        current_pos = pos

    return segments


def split_segments_into_windows(
    segments: List[Dict[str, Any]],
    points_per_window: int = 100_000,
    padding: int = 150_000,
) -> List[TimeWindow]:
    """Split each segment into windows of size <= points_per_window."""
    if points_per_window <= 0:
        raise ValueError("points_per_window must be > 0")

    windows: List[TimeWindow] = []

    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        active_annotations = list(seg["active_annotations"])
        descriptions = list(seg["descriptions"])

        cursor = start
        while cursor < end:
            w_end = min(cursor + points_per_window, end)
            windows.append(TimeWindow(
                start=max(cursor- padding, 0),
                end=w_end+ padding,
                length=w_end - max(cursor- padding, 0) + padding,
                active_annotations=active_annotations,
                descriptions=descriptions,
            ))
            cursor = w_end

    return windows


def build_cwt_windows_from_annotations(
    annotations: List[Dict[str, Any]],
    points_per_window: int = 100_000,
    global_start: int = 0,
    global_end: int | None = None,
) -> List[TimeWindow]:
    """
    Full pipeline:
      annotations -> overlap segments -> CWT computation windows
    """
    segments = build_overlap_segments(
        annotations=annotations,
        global_start=global_start,
        global_end=global_end,
    )
    windows = split_segments_into_windows(
        segments=segments,
        points_per_window=points_per_window,
    )
    cleaned_windows = []
    for w in windows:
        for cw in cleaned_windows:
            if cw.descriptions == w.descriptions:
                break
        else:
            cleaned_windows.append(w)
    return cleaned_windows


if __name__ == "__main__":
    meta_file = "/home/antoine/Documents/ICE/projet/wavelet_ice/data/baseline/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-meta"
    meta = load_metadata(meta_file)
    annotations = meta.get("annotations", []) if meta else []


    PARAMS = {
        'points_per_window': 1_000_000,
        'offset': 0,
        'duration': 100_000_000,
    }
    windows = build_cwt_windows_from_annotations(
        annotations=annotations,
        points_per_window=PARAMS['points_per_window'],
        global_start=PARAMS['offset'],
        global_end=PARAMS['offset'] + PARAMS['duration'],
    )
    print(f"{len(windows)} CWT windows to compute.")
    for i, w in enumerate(windows):
        print(
            f"[{i}] start={w.start}, end={w.end}, len={w.length}, "
            f"active={w.descriptions}"
        )