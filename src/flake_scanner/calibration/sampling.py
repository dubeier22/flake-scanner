"""Core color-sampling and contrast math.

Ported from the original ``flake_finder.py`` (functions ``sample_patch``,
``auto_substrate``, ``contrast``) with behaviour preserved. Contrast is the
relative, substrate-referenced metric ``(substrate - flake) / substrate``
computed per channel, which is robust to lamp brightness / white-balance
drift between imaging sessions.

Arrays are BGR (OpenCV-native). Sampling returns float BGR means.
"""

from __future__ import annotations

import cv2
import numpy as np

SAMPLE_RADIUS = 8
SUBSTRATE_GRID = 25


def sample_patch(img: np.ndarray, x: int, y: int, r: int = SAMPLE_RADIUS) -> np.ndarray:
    """Mean BGR of a square patch of radius ``r`` centred at ``(x, y)``.

    Clips at image borders. Returns a float BGR triple.
    """
    h, w = img.shape[:2]
    patch = img[max(0, y - r) : min(h, y + r), max(0, x - r) : min(w, x + r)]
    return patch.mean(axis=(0, 1))


def global_substrate(img: np.ndarray, grid: int = SUBSTRATE_GRID) -> np.ndarray:
    """Estimate the global substrate colour as the per-channel median of a grid.

    Samples a ``grid``x``grid`` lattice of small patches across the interior of
    the image and takes the median, which is robust to flakes covering a
    minority of the chip area. Returns a float BGR triple.
    """
    h, w = img.shape[:2]
    pts = []
    for gy in np.linspace(0.05, 0.95, grid):
        for gx in np.linspace(0.05, 0.95, grid):
            y, x = int(gy * h), int(gx * w)
            pts.append(img[max(0, y - 4) : y + 4, max(0, x - 4) : x + 4].mean(axis=(0, 1)))
    return np.median(pts, axis=0)


def contrast(flake_bgr: np.ndarray, sub_bgr: np.ndarray) -> np.ndarray:
    """Per-channel relative contrast ``(sub - flake) / sub``.

    Guards against a near-zero substrate channel (returns 0 there). Inputs and
    output are BGR. Invariant to multiplicative scaling of the inputs (so bit
    depth / exposure scaling does not change the result).
    """
    flake_bgr = np.asarray(flake_bgr, dtype=np.float64)
    sub_bgr = np.asarray(sub_bgr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sub_bgr > 1, (sub_bgr - flake_bgr) / sub_bgr, 0.0)


def normalized_difference(flake_bgr: np.ndarray, sub_bgr: np.ndarray) -> np.ndarray:
    """Per-channel normalized difference ``(sub - flake) / (sub + flake)``.

    The preferred contrast metric (see docs/6-memo/contrast-metric-analysis.md):
    like plain relative contrast it is invariant to multiplicative exposure /
    white-balance changes, but it is additionally the most robust to additive
    black-level offsets, is bounded in [-1, 1] (numerically safe when the local
    substrate is dark), and gives the best thickness discrimination. Works on
    BGR triples or on whole ``(..., 3)`` image arrays.
    """
    f = np.asarray(flake_bgr, dtype=np.float64)
    s = np.asarray(sub_bgr, dtype=np.float64)
    return (s - f) / (s + f + 1e-6)


def local_substrate_ring(
    img: np.ndarray, x: int, y: int, r_in: int = 90, r_out: int = 140
) -> np.ndarray:
    """Median BGR of an annulus around ``(x, y)`` — a local substrate estimate.

    Sampling a ring at radius ``[r_in, r_out]`` around a flake captures nearby
    bare substrate, which tracks illumination falloff better than one global
    value. Returns a float BGR triple.
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[-r_out : r_out + 1, -r_out : r_out + 1]
    ring = (np.hypot(xx, yy) > r_in) & (np.hypot(xx, yy) < r_out)
    patch = img[max(0, y - r_out) : y + r_out + 1, max(0, x - r_out) : x + r_out + 1]
    ring = ring[: patch.shape[0], : patch.shape[1]]
    return np.median(patch[ring], axis=0)


def snap_sample(
    img: np.ndarray,
    x: int,
    y: int,
    window: int = 170,
    color_thresh: float = 40.0,
    min_area: int = 120,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(flake_bgr, substrate_bgr)`` for the flake at/near ``(x, y)``.

    Estimates a local substrate from the window median, segments flake pixels
    (BGR distance from substrate above ``color_thresh``), and picks a flake
    blob. To avoid grabbing a touching neighbour, it **prefers the blob the
    marked pixel actually sits on**; only if the mark is on bare substrate does
    it fall back to the nearest blob. ``exclude_mask`` (full-image, non-zero =
    ignore) drops pixels — e.g. the coloured annotation ink — from both the
    substrate estimate and the flake mean so labels can't tint the sample.
    Returns ``None`` if no flake is found.
    """
    h, w = img.shape[:2]
    x0, x1 = max(0, x - window), min(w, x + window)
    y0, y1 = max(0, y - window), min(h, y + window)
    win = img[y0:y1, x0:x1].astype(np.float32)
    keep = np.ones(win.shape[:2], bool)
    if exclude_mask is not None:
        keep = exclude_mask[y0:y1, x0:x1] == 0
    sub = np.median(win[keep].reshape(-1, 3), axis=0)
    mask = (np.linalg.norm(win - sub, axis=2) > color_thresh) & keep
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    cx, cy = x - x0, y - y0
    at = lab[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)]
    if at != 0 and stats[at][4] >= min_area:
        best = at  # the mark sits on a flake — use that flake
    else:
        best, best_d = None, np.inf
        for i in range(1, n):
            if stats[i][4] < min_area:
                continue
            d = np.hypot(cent[i][0] - cx, cent[i][1] - cy)
            if d < best_d:
                best, best_d = i, d
    if best is None:
        return None
    flake = win[lab == best].mean(axis=0)
    return flake, sub


def sample_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    exclude_mask: np.ndarray | None = None,
    color_thresh: float = 40.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample ``(flake_bgr, substrate_bgr)`` for a flake enclosed by a box.

    Substrate is the median of a ring just outside the box; the flake is the
    mean of the in-box pixels that differ from that substrate (i.e. the flake,
    not any bare substrate the box also encloses), excluding ``exclude_mask``
    ink pixels. Unambiguous — no blob picking or snap-to-neighbour needed.
    """
    ih, iw = img.shape[:2]
    m = max(w, h) // 3 + 5
    ox0, oy0 = max(0, x - m), max(0, y - m)
    ox1, oy1 = min(iw, x + w + m), min(ih, y + h + m)
    outer = img[oy0:oy1, ox0:ox1].astype(np.float32)
    inner_rel = (slice(y - oy0, y - oy0 + h), slice(x - ox0, x - ox0 + w))
    ring = np.ones(outer.shape[:2], bool)
    ring[inner_rel] = False
    if exclude_mask is not None:
        ring &= exclude_mask[oy0:oy1, ox0:ox1] == 0
    if ring.sum() < 10:
        return None
    sub = np.median(outer[ring].reshape(-1, 3), axis=0)

    box = img[y : y + h, x : x + w].astype(np.float32)
    keep = np.linalg.norm(box - sub, axis=2) > color_thresh
    if exclude_mask is not None:
        keep &= exclude_mask[y : y + h, x : x + w] == 0
    if keep.sum() < 10:
        return None
    flake = box[keep].mean(axis=0)
    return flake, sub
