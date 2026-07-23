"""Auto-sample flake and substrate colour from a high-magnification crop.

Each per-flake crop shows one flake roughly centred on a fairly uniform
substrate, often with burned-in measurement annotations (thin dark lines and
small red text). This module estimates the substrate colour from the crop
border, masks out the annotations, finds the central flake blob, and returns
mean flake/substrate BGR so a calibration contrast can be computed without
manual clicking.

NOTE: crops are high-mag with a different substrate tint than the low-mag
full-chip mosaics, so crop-derived *absolute* colour differs from the mosaic.
Relative contrast is more stable but not identical across the two domains --
see the plan's domain-mismatch discussion.
"""

from __future__ import annotations

import cv2
import numpy as np

from .markers import red_mask


def _border_substrate(img: np.ndarray, frac: float = 0.12) -> np.ndarray:
    """Median BGR of the crop's outer border ring (assumed bare substrate)."""
    h, w = img.shape[:2]
    b = int(min(h, w) * frac)
    ring = np.concatenate(
        [
            img[:b].reshape(-1, 3),
            img[-b:].reshape(-1, 3),
            img[:, :b].reshape(-1, 3),
            img[:, -b:].reshape(-1, 3),
        ]
    )
    return np.median(ring, axis=0)


def sample_crop(
    img: np.ndarray,
    color_thresh: float = 40.0,
    min_area_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return ``(flake_bgr, substrate_bgr, flake_area_px)`` for a crop.

    Substrate is estimated from the border ring; the flake is the largest
    connected blob of pixels whose BGR distance from the substrate exceeds
    ``color_thresh``, excluding burned-in red annotation ink. Returns ``None``
    if no flake blob of at least ``min_area_frac`` of the image is found.
    """
    h, w = img.shape[:2]
    sub = _border_substrate(img).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - sub, axis=2)
    red = red_mask(img)
    flake = ((dist > color_thresh) & (red == 0)).astype(np.uint8)
    flake = cv2.morphologyEx(flake, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    flake = cv2.morphologyEx(flake, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(flake)
    min_area = int(min_area_frac * h * w)
    best, best_score = None, -1.0
    cx0, cy0 = w / 2, h / 2
    for i in range(1, n):
        a = int(stats[i][4])
        if a < min_area:
            continue
        # prefer large blobs near the crop centre (the annotated flake)
        bx = stats[i][0] + stats[i][2] / 2
        by = stats[i][1] + stats[i][3] / 2
        centrality = 1.0 - (np.hypot(bx - cx0, by - cy0) / np.hypot(cx0, cy0))
        score = a * (0.5 + 0.5 * centrality)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    flake_bgr = img[lab == best].mean(axis=0)
    return flake_bgr, sub, int(stats[best][4])
