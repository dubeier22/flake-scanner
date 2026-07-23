"""Scan a full-chip mosaic for flakes matching a material/thickness signature.

Pipeline (see docs/6-memo/contrast-metric-analysis.md for the rationale):

1. Estimate a local substrate field (handles illumination falloff).
2. Classify every pixel by its normalized-difference contrast against the local
   substrate, via Mahalanobis distance to the target signature. This isolates
   the correct-thickness *regions* — so a terraced flake contributes only its
   on-target part, and off-colour flakes are excluded per-pixel.
3. Connect the matching pixels into regions, keep those large enough (a target
   lateral size in microns via the pixel scale), and rank by colour match.

Known limitation: the per-pixel mask is conservative at flake edges, so region
sizes for small, touching flakes are approximate (see the memo). Pixel scale is
currently supplied/estimated, not read from a scale bar.

Memory: the per-pixel Mahalanobis field is computed in horizontal strips so a
full-resolution float image is never held all at once.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..calibration.signature import MaterialSignature
from ..calibration.sampling import normalized_difference
from ..models import Candidate
from .substrate import substrate_field


def scan(
    img: np.ndarray,
    signature: MaterialSignature,
    um_per_px: float,
    min_size_um: float = 23.0,
    match_threshold: float = 2.0,
    n_strips: int = 20,
) -> list[Candidate]:
    """Return candidate flake regions matching ``signature``, ranked by match.

    ``match_threshold`` is the Mahalanobis cutoff (lower = stricter/cleaner
    list). ``min_size_um`` filters regions whose bounding box is smaller than
    the target lateral size in both dimensions.
    """
    h, w = img.shape[:2]
    min_px = int(min_size_um / um_per_px)
    subf = substrate_field(img)

    # per-pixel match mask, computed in strips to bound memory
    mask = np.zeros((h, w), np.uint8)
    strip = h // n_strips + 1
    for i in range(n_strips):
        y0 = i * strip
        y1 = min(h, y0 + strip)
        if y0 >= h:
            break
        ndi = normalized_difference(img[y0:y1], subf[y0:y1])
        mask[y0:y1] = (signature.mahalanobis_field(ndi) < match_threshold).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    cands: list[Candidate] = []
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        if bw < min_px or bh < min_px:
            continue
        region = img[y : y + bh, x : x + bw][lab[y : y + bh, x : x + bw] == i]
        flake = region.mean(axis=0)
        local_sub = subf[int(cent[i][1]), int(cent[i][0])]
        maha = signature.mahalanobis(normalized_difference(flake, local_sub))
        cands.append(
            Candidate(
                rank=0, maha=maha, x=int(x), y=int(y), w_px=int(bw), h_px=int(bh),
                cx=int(cent[i][0]), cy=int(cent[i][1]),
                w_um=bw * um_per_px, h_um=bh * um_per_px,
            )
        )
    cands.sort(key=lambda c: c.maha)
    for rk, c in enumerate(cands, 1):
        c.rank = rk
    return cands
