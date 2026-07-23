"""Write scan results: a ranked CSV and an annotated candidate map."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from ..models import Candidate

CSV_FIELDS = ["rank", "maha", "w_um", "h_um", "cx", "cy", "x", "y", "w_px", "h_px"]


def write_csv(candidates: list[Candidate], path: str | Path) -> None:
    """Write the ranked candidate list (positions + sizes + match score)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for c in candidates:
            w.writerow(
                {
                    "rank": c.rank, "maha": round(c.maha, 3),
                    "w_um": round(c.w_um, 1), "h_um": round(c.h_um, 1),
                    "cx": c.cx, "cy": c.cy, "x": c.x, "y": c.y,
                    "w_px": c.w_px, "h_px": c.h_px,
                }
            )


def write_map(
    img: np.ndarray, candidates: list[Candidate], path: str | Path, max_width: int = 2600
) -> None:
    """Write a whole-chip map with each candidate boxed in red and numbered by rank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    for c in candidates:
        pad = 25
        cv2.rectangle(vis, (c.x - pad, c.y - pad), (c.x + c.w_px + pad, c.y + c.h_px + pad),
                      (0, 0, 255), 10)
        cv2.putText(vis, str(c.rank), (c.x - pad, c.y - pad - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 8)
    h, w = img.shape[:2]
    out = cv2.resize(vis, (max_width, int(max_width * h / w)))
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
