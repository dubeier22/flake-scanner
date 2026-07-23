"""Detect hand-drawn red flake markers on a chip mosaic.

Some mosaics have the AFM'd flakes annotated with hand-drawn red numbers next
to each flake. This module locates those red marks (clustering the digit
strokes of one number into a single marker) and associates each marker with
the nearest flake-coloured blob, so a flake's pixel location and colour can be
recovered without manual clicking. Used both to bootstrap calibration (sample
colour at marked flakes) and to build validation ground truth (where the
known-thickness flakes are).

BGR arrays throughout (OpenCV-native).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Marker:
    """A detected red marker and the flake it annotates."""

    marker_x: int
    marker_y: int
    n_strokes: int
    flake_x: int | None = None
    flake_y: int | None = None
    flake_area: int | None = None


def red_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of saturated red ink (both hue wrap-around ranges)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
    m = cv2.bitwise_or(lo, hi)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def find_markers(
    img: np.ndarray,
    min_stroke_area: int = 150,
    cluster_dist: int = 160,
) -> list[Marker]:
    """Locate red markers by clustering nearby red strokes into one number.

    ``cluster_dist`` (px) groups the separate strokes of a single hand-drawn
    number; ``min_stroke_area`` drops speckle. Returns markers in reading order
    (top-to-bottom, left-to-right).
    """
    mask = red_mask(img)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    strokes = [
        (float(cent[i][0]), float(cent[i][1]), int(stats[i][4]))
        for i in range(1, n)
        if stats[i][4] >= min_stroke_area
    ]
    if not strokes:
        return []

    pts = np.array([[s[0], s[1]] for s in strokes])
    used = [False] * len(strokes)
    markers: list[Marker] = []
    for i in range(len(strokes)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        # transitive grouping: pull in any stroke close to any group member
        changed = True
        while changed:
            changed = False
            for j in range(len(strokes)):
                if used[j]:
                    continue
                if any(np.hypot(*(pts[j] - pts[k])) < cluster_dist for k in group):
                    group.append(j)
                    used[j] = True
                    changed = True
        cx = int(np.mean([pts[k][0] for k in group]))
        cy = int(np.mean([pts[k][1] for k in group]))
        markers.append(Marker(marker_x=cx, marker_y=cy, n_strokes=len(group)))

    markers.sort(key=lambda m: (m.marker_y, m.marker_x))
    return markers


def associate_flakes(
    img: np.ndarray,
    markers: list[Marker],
    substrate_bgr: np.ndarray,
    window: int = 260,
    color_thresh: float = 45.0,
    min_flake_area: int = 30,
) -> list[Marker]:
    """Fill each marker's flake location: the largest non-red, non-substrate blob
    near the marker.

    Searches a ``window``x``window`` box around each marker for pixels whose BGR
    distance from ``substrate_bgr`` exceeds ``color_thresh`` (a flake), excludes
    the red ink itself, and takes the largest connected blob as the annotated
    flake. Markers with no qualifying blob are returned with ``flake_x`` = None.
    """
    h, w = img.shape[:2]
    sub = np.asarray(substrate_bgr, dtype=np.float32)
    red = red_mask(img)
    out: list[Marker] = []
    for m in markers:
        x0, x1 = max(0, m.marker_x - window), min(w, m.marker_x + window)
        y0, y1 = max(0, m.marker_y - window), min(h, m.marker_y + window)
        roi = img[y0:y1, x0:x1].astype(np.float32)
        dist = np.linalg.norm(roi - sub, axis=2)
        flake = ((dist > color_thresh) & (red[y0:y1, x0:x1] == 0)).astype(np.uint8)
        flake = cv2.morphologyEx(flake, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(flake)
        best, best_area = None, 0
        for i in range(1, n):
            a = int(stats[i][4])
            if a > best_area and a >= min_flake_area:
                best, best_area = i, a
        if best is None:
            out.append(m)
        else:
            out.append(
                Marker(
                    marker_x=m.marker_x,
                    marker_y=m.marker_y,
                    n_strokes=m.n_strokes,
                    flake_x=int(cent[best][0]) + x0,
                    flake_y=int(cent[best][1]) + y0,
                    flake_area=best_area,
                )
            )
    return out


def annotate(img: np.ndarray, markers: list[Marker]) -> np.ndarray:
    """Draw detected markers (index + marker/flake points) for visual confirmation."""
    vis = img.copy()
    for idx, m in enumerate(markers, 1):
        cv2.circle(vis, (m.marker_x, m.marker_y), 70, (255, 0, 0), 6)
        cv2.putText(
            vis, str(idx), (m.marker_x + 75, m.marker_y),
            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 8,
        )
        if m.flake_x is not None:
            cv2.circle(vis, (m.flake_x, m.flake_y), 30, (0, 255, 0), 5)
            cv2.line(vis, (m.marker_x, m.marker_y), (m.flake_x, m.flake_y), (0, 255, 0), 3)
    return vis
