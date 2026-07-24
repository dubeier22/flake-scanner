"""Read magenta box + flake-ID annotations off a full-chip mosaic.

Workflow: in an image editor, draw a **magenta** (#FF00FF) box around each
AFM-measured flake and write a magenta flake-ID number (1, 2, 3, ...) beside it.
Magenta is used because it is essentially absent from the samples themselves
(graphite/hBN flakes are yellow/green/red/pink on a blue substrate), so the
annotation isolates cleanly by colour — unlike red, which clashes with reddish
flakes.

This module isolates the magenta ink, clusters each flake's box + number,
locates the flake, and OCRs its ID. Thicknesses are supplied separately (the
number on the map is the flake ID, not the thickness), so the ID reading only
needs the small integers 1..N — validated against that expected set.

OCR needs ``pytesseract`` + the ``tesseract`` binary (``brew install tesseract``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV-BGR of pure magenta; detection is by hue so exact value is not critical.
MAGENTA_BGR = (255, 0, 255)


@dataclass
class FlakeAnnotation:
    """One detected flake: its OCR'd ID (or None) and pixel location."""

    flake_id: int | None
    x: int
    y: int


def _import_pytesseract():
    try:
        import pytesseract

        return pytesseract
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "reading map annotations needs OCR: pip install pytesseract and "
            "install the tesseract binary (brew install tesseract)"
        ) from e


def ink_mask(img: np.ndarray, mark_bgr: tuple[int, int, int] = MAGENTA_BGR, tol: int = 18) -> np.ndarray:
    """Binary mask of the annotation ink, isolated by hue (robust to JPEG drift)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_h = int(cv2.cvtColor(np.uint8([[mark_bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    lo = np.array([max(0, target_h - tol), 90, 90], np.uint8)
    hi = np.array([min(179, target_h + tol), 255, 255], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    # drop tiny speckle
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    clean = np.zeros_like(m)
    for i in range(1, n):
        if stats[i][4] >= 60:
            clean[lab == i] = 255
    return clean


def _ocr_id(number_crop: np.ndarray, pytesseract) -> int | None:
    """OCR a small magenta number crop (upscaled, black-on-white)."""
    if number_crop.size == 0 or number_crop.max() == 0:
        return None
    sc = 140 / max(1, number_crop.shape[0])
    up = cv2.resize(number_crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    up = cv2.dilate(up, np.ones((3, 3), np.uint8))
    up = cv2.copyMakeBorder(up, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    ocr_img = cv2.cvtColor(255 - up, cv2.COLOR_GRAY2BGR)
    for psm in (8, 7, 10, 13, 6):
        txt = pytesseract.image_to_string(
            ocr_img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        )
        digits = "".join(c for c in txt if c.isdigit())
        if digits:
            return int(digits)
    return None


def read_flake_annotations(
    img: np.ndarray,
    mark_bgr: tuple[int, int, int] = MAGENTA_BGR,
    cluster_dilate: int = 80,
    color_thresh: float = 45.0,
) -> list[FlakeAnnotation]:
    """Locate each annotated flake and OCR its ID.

    Clusters the magenta ink per flake (box + number), finds the flake blob
    inside each cluster, and OCRs the ID. Returns one ``FlakeAnnotation`` per
    flake found; ``flake_id`` is ``None`` where the ID could not be read (the
    caller reconciles against the expected 1..N set / user verification).
    """
    pytesseract = _import_pytesseract()
    ink = ink_mask(img, mark_bgr)
    ink_n, ink_lab, ink_stats, _ = cv2.connectedComponentsWithStats(ink)
    gsub = np.median(img[::40, ::40].reshape(-1, 3), axis=0).astype(np.float32)

    # cluster box+number per flake; flakes are far apart so they stay separate
    clustered = cv2.dilate(ink, np.ones((cluster_dilate, cluster_dilate), np.uint8))
    n, clu_lab, stats, _cent = cv2.connectedComponentsWithStats(clustered)

    out: list[FlakeAnnotation] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4000:  # ignore stray specks that survived
            continue
        # flake location = largest non-substrate, non-ink blob inside the cluster
        roi = img[y : y + h, x : x + w].astype(np.float32)
        sub_ink = ink[y : y + h, x : x + w]
        flake = ((np.linalg.norm(roi - gsub, axis=2) > color_thresh) & (sub_ink == 0)).astype(np.uint8)
        flake = cv2.morphologyEx(flake, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fn, _fl, fstats, fcent = cv2.connectedComponentsWithStats(flake)
        best, best_a = None, 0
        for j in range(1, fn):
            if fstats[j][4] > best_a and fstats[j][4] >= 40:
                best, best_a = j, fstats[j][4]
        if best is None:
            continue  # no flake in this cluster -> stray annotation, skip
        fx, fy = int(fcent[best][0]) + x, int(fcent[best][1]) + y

        # Separate the ID number from the box: the box surrounds the flake (its
        # ink sits within ~half the box size of the flake centre); the number is
        # offset. OCR only the ink away from the flake (the number).
        near_r = max(60, int(0.7 * np.sqrt(best_a)))
        number_mask = sub_ink.copy()
        cv2.circle(number_mask, (fx - x, fy - y), near_r, 0, -1)  # erase the box
        fid = _ocr_id(number_mask, pytesseract)
        out.append(FlakeAnnotation(flake_id=fid, x=fx, y=fy))

    # dedupe flakes that resolved to nearly the same location (split clusters)
    deduped: list[FlakeAnnotation] = []
    for a in out:
        dup = next((b for b in deduped if np.hypot(a.x - b.x, a.y - b.y) < 300), None)
        if dup is None:
            deduped.append(a)
        elif dup.flake_id is None and a.flake_id is not None:
            dup.flake_id, dup.x, dup.y = a.flake_id, a.x, a.y
    return deduped


def annotate_preview(
    img: np.ndarray, annotations: list[FlakeAnnotation], max_width: int = 1600
) -> np.ndarray:
    """Draw detected flakes + their read IDs for the user to verify (RGB)."""
    vis = img.copy()
    for a in annotations:
        label = str(a.flake_id) if a.flake_id is not None else "?"
        cv2.circle(vis, (a.x, a.y), 60, (0, 0, 255), 6)
        cv2.putText(vis, label, (a.x + 65, a.y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 8)
    h, w = img.shape[:2]
    vis = cv2.resize(vis, (max_width, int(max_width * h / w)))
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
