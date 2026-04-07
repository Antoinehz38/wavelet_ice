import cv2
import numpy as np


def smooth_1d(x, k):
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def segments_from_profile(profile, rel_thr=0.2, smooth_k=21, min_len=10):
    p = smooth_1d(profile, smooth_k)
    if p.max() <= 0:
        return [], p

    thr = rel_thr * p.max()
    m = p > thr

    segs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(m) - start >= min_len:
        segs.append((start, len(m)))

    return segs, p


def merge_close_segments(segs, gap=5):
    if not segs:
        return []
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def robust_row_z(gray):
    x = gray.astype(np.float32)
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    mad = np.maximum(mad, 1.0)
    z = (x - med) / mad
    z = np.clip(z, 0, None)
    return z


def detect_signals_by_projections(gray):
    H, W = gray.shape
    z = robust_row_z(gray)

    # léger écrêtage pour éviter que quelques pixels dominent
    z = np.clip(z, 0, 8)

    # -------- 1) segmentation temporelle globale --------
    time_profile = z.sum(axis=0)
    time_segs, time_profile_s = segments_from_profile(
        time_profile,
        rel_thr=0.15,
        smooth_k=max(31, (W // 40) | 1),
        min_len=max(20, W // 100)
    )
    time_segs = merge_close_segments(time_segs, gap=max(8, W // 200))

    boxes = []

    # -------- 2) segmentation fréquentielle dans chaque plage de temps --------
    for x0, x1 in time_segs:
        roi = z[:, x0:x1]
        freq_profile = roi.sum(axis=1)

        freq_segs, freq_profile_s = segments_from_profile(
            freq_profile,
            rel_thr=0.20,
            smooth_k=max(9, (H // 40) | 1),
            min_len=max(6, H // 80)
        )
        freq_segs = merge_close_segments(freq_segs, gap=max(2, H // 150))

        for y0, y1 in freq_segs:
            sub = z[y0:y1, x0:x1]

            # resserrage fin avec énergie locale
            col_energy = sub.sum(axis=0)
            row_energy = sub.sum(axis=1)

            col_segs, _ = segments_from_profile(
                col_energy,
                rel_thr=0.25,
                smooth_k=max(11, ((x1 - x0) // 20) | 1),
                min_len=max(8, (x1 - x0) // 30)
            )
            row_segs, _ = segments_from_profile(
                row_energy,
                rel_thr=0.25,
                smooth_k=max(5, ((y1 - y0) // 4) | 1),
                min_len=max(4, (y1 - y0) // 4)
            )

            if not col_segs:
                col_segs = [(0, x1 - x0)]
            if not row_segs:
                row_segs = [(0, y1 - y0)]

            # en général on garde seulement le segment principal
            cx0, cx1 = max(col_segs, key=lambda t: t[1] - t[0])
            ry0, ry1 = max(row_segs, key=lambda t: t[1] - t[0])

            bx0 = max(0, x0 + cx0 - 4)
            bx1 = min(W, x0 + cx1 + 4)
            by0 = max(0, y0 + ry0 - 2)
            by1 = min(H, y0 + ry1 + 2)

            if bx1 - bx0 >= 10 and by1 - by0 >= 4:
                boxes.append((bx0, by0, bx1 - bx0, by1 - by0))

    return boxes, {
        "z": z,
        "time_profile": time_profile,
        "time_segments": time_segs,
    }


def draw_boxes(gray, boxes):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

def tighten_box_with_energy(z, box, qx=(0.01, 0.99), qy=(0.01, 0.99), pad_x=0, pad_y=0):
    x, y, w, h = box
    sub = z[y:y+h, x:x+w]

    col_energy = sub.sum(axis=0)
    row_energy = sub.sum(axis=1)

    cx0, cx1 = tight_interval_from_energy(col_energy, *qx)
    ry0, ry1 = tight_interval_from_energy(row_energy, *qy)

    nx0 = max(0, x + cx0 - pad_x)
    nx1 = min(z.shape[1], x + cx1 + pad_x)
    ny0 = max(0, y + ry0 - pad_y)
    ny1 = min(z.shape[0], y + ry1 + pad_y)

    return (nx0, ny0, nx1 - nx0, ny1 - ny0)


from scipy.ndimage import gaussian_filter, label

def tighten_box_2d(z, box,
                   sigma=2,
                   thr_rel=0.3,
                   min_area_ratio=0.01,
                   pad=0):
    x, y, w, h = box
    sub = z[y:y+h, x:x+w]

    # lissage 2D
    sm = gaussian_filter(sub, sigma=sigma)

    # seuil relatif au max local (beaucoup plus stable ici)
    thr = sm.max() * thr_rel
    mask = sm > thr

    # composantes connexes
    lbl, n = label(mask)

    if n == 0:
        return box

    # garder la plus grosse composante
    best = None
    best_area = 0

    for i in range(1, n+1):
        ys, xs = np.where(lbl == i)
        area = len(xs)

        if area > best_area:
            best_area = area
            best = (xs, ys)

    xs, ys = best

    # bbox serrée
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()

    # coords globales
    nx0 = max(0, x + left - pad)
    nx1 = min(z.shape[1], x + right + pad)
    ny0 = max(0, y + top - pad)
    ny1 = min(z.shape[0], y + bottom + pad)

    return (nx0, ny0, nx1 - nx0, ny1 - ny0)

import numpy as np

def tight_interval_from_energy(profile, q_low=0.01, q_high=0.99):
    p = np.asarray(profile, dtype=np.float64)
    p = np.maximum(p, 0)

    s = p.sum()
    if s <= 0:
        return 0, len(p)

    c = np.cumsum(p) / s

    i0 = np.searchsorted(c, q_low)
    i1 = np.searchsorted(c, q_high)

    i0 = max(0, min(i0, len(p)-1))
    i1 = max(i0 + 1, min(i1 + 1, len(p)))

    return i0, i1