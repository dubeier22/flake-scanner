"""Core color-sampling and contrast math.

Ported from the original ``flake_finder.py`` (functions ``sample_patch``,
``auto_substrate``, ``contrast``) with behaviour preserved. Contrast is the
relative, substrate-referenced metric ``(substrate - flake) / substrate``
computed per channel, which is robust to lamp brightness / white-balance
drift between imaging sessions.

Arrays are BGR (OpenCV-native). Sampling returns float BGR means.
"""

from __future__ import annotations

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
