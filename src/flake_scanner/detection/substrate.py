"""Local substrate estimation for a full mosaic.

A single global substrate colour is wrong across a large chip because of
illumination falloff (the substrate is brighter in the centre than at the
edges). This builds a coarse local-substrate *field* by downsampling the image
and median-filtering, then upsampling back — cheap in memory (kept coarse) yet
per-region accurate. Flakes are a minority of pixels, so a block median is a
good substrate estimate even where flakes are present.
"""

from __future__ import annotations

import cv2
import numpy as np


def substrate_field(img: np.ndarray, block: int = 64, med: int = 5) -> np.ndarray:
    """Return a full-resolution BGR local-substrate estimate (float32).

    ``block`` sets the coarseness (downsample factor); ``med`` is the median
    kernel applied at the coarse scale to reject flake-tinted cells.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, med)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
