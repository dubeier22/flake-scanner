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

# Detection is by hue, so exact BGR values aren't critical. Boxes and numbers
# use two different colours, both absent from the samples, so each is isolated
# independently (box detection never sees digit holes; OCR never sees box lines).
MAGENTA_BGR = (255, 0, 255)  # boxes
BLUE_BGR = (255, 0, 0)  # numbers


@dataclass
class FlakeAnnotation:
    """One detected flake: its OCR'd ID and the box drawn around it."""

    flake_id: int | None
    x: int  # box top-left
    y: int
    w: int  # box size
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


def _import_pytesseract():
    try:
        import pytesseract

        return pytesseract
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "reading map annotations needs OCR: pip install pytesseract and "
            "install the tesseract binary (brew install tesseract)"
        ) from e


def ink_mask(
    img: np.ndarray,
    mark_bgr: tuple[int, int, int] = MAGENTA_BGR,
    tol: int = 18,
    min_sat: int = 90,
) -> np.ndarray:
    """Binary mask of the annotation ink, isolated by hue (robust to JPEG drift).

    ``min_sat`` sets the saturation floor. Raise it for a colour that shares a
    hue with the (desaturated) substrate — e.g. blue numbers vs the blue-grey
    substrate — so only the fully-saturated ink survives.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_h = int(cv2.cvtColor(np.uint8([[mark_bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    lo = np.array([max(0, target_h - tol), min_sat, 90], np.uint8)
    hi = np.array([min(179, target_h + tol), 255, 255], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    # drop tiny speckle (vectorised label lookup — fast even with many components)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = stats[:, 4] >= 60
    keep[0] = False  # background
    return (keep[lab] * np.uint8(255)).astype(np.uint8)


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


def detect_boxes(
    img: np.ndarray,
    box_bgr: tuple[int, int, int] = MAGENTA_BGR,
    min_box: int = 30,
    max_box: int = 550,
) -> list[tuple[int, int, int, int]]:
    """Detect the box-colour bounding boxes as ``(x, y, w, h)`` rectangles.

    HSV-filters the box colour, reconnects fragmented lines, finds contours,
    and keeps the rectangular, hollow, box-sized ones (a box is hollow because
    the flake sits inside it). Nested/overlapping detections are deduped to the
    one enclosing the most flake. Since numbers are a different colour, digit
    holes are never mistaken for boxes.
    """
    ink = ink_mask(img, box_bgr)
    gsub = np.median(img[::40, ::40].reshape(-1, 3), axis=0).astype(np.float32)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_box <= min(w, h) and max(w, h) < max_box and 0.35 < w / h < 2.8):
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        interior_ink = ink[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]
        if not (4 <= len(approx) <= 8 and interior_ink.size and interior_ink.mean() / 255 < 0.25):
            continue
        # a real box has a flake inside it (a colour that differs from substrate,
        # and isn't the mark ink); this drops number-regions mis-read as boxes
        roi = img[y : y + h, x : x + w].astype(np.float32)
        flake_px = int(((np.linalg.norm(roi - gsub, axis=2) > 45) & (ink[y : y + h, x : x + w] == 0)).sum())
        if flake_px < 40:
            continue
        cands.append((flake_px, int(x), int(y), int(w), int(h)))
    # among overlapping candidates keep the one enclosing the most flake
    cands.sort(key=lambda b: -b[0])
    boxes: list[tuple[int, int, int, int]] = []
    for _fp, x, y, w, h in cands:
        bcx, bcy = x + w // 2, y + h // 2
        if not any(abs(bcx - (k[0] + k[2] // 2)) < 250 and abs(bcy - (k[1] + k[3] // 2)) < 250 for k in boxes):
            boxes.append((x, y, w, h))
    return boxes


def read_flake_annotations(
    img: np.ndarray,
    box_bgr: tuple[int, int, int] = MAGENTA_BGR,
    number_bgr: tuple[int, int, int] = BLUE_BGR,
) -> list[FlakeAnnotation]:
    """Detect each box (box colour) and OCR the ID number (number colour) near it.

    Boxes and numbers are different colours, so box detection and digit OCR are
    fully decoupled. The box defines the flake region (only flakes inside a box
    are used); the ID is the number-coloured text nearest the box. Returns one
    ``FlakeAnnotation`` per box; ``flake_id`` is ``None`` where the digit could
    not be read (the caller reconciles vs the expected 1..N set / verification).
    """
    pytesseract = _import_pytesseract()
    # high saturation floor: blue numbers vs the desaturated blue-grey substrate
    numbers = ink_mask(img, number_bgr, min_sat=180)
    out: list[FlakeAnnotation] = []
    for x, y, w, h in detect_boxes(img, box_bgr):
        # the ID is written next to the box (above by default); search a generous
        # region around the box for number-coloured text and OCR it.
        pad = max(w, h) + 60
        ry0, ry1 = max(0, y - pad), min(numbers.shape[0], y + h + pad // 2)
        rx0, rx1 = max(0, x - pad), min(numbers.shape[1], x + w + pad)
        crop = numbers[ry0:ry1, rx0:rx1].copy()
        # erase number-ink that falls inside the box (defensive; numbers sit outside)
        cv2.rectangle(crop, (x - rx0, y - ry0), (x - rx0 + w, y - ry0 + h), 0, -1)
        # keep only the number cluster nearest the box, then crop tight so the
        # digit fills the frame (OCR needs the number upscaled, not the region)
        cn, clab, cstats, ccent = cv2.connectedComponentsWithStats(crop)
        digits = [k for k in range(1, cn) if cstats[k][4] >= 40]
        if digits:
            bcx, bcy = (x + w // 2) - rx0, (y + h // 2) - ry0
            anchor = min(digits, key=lambda k: np.hypot(ccent[k][0] - bcx, ccent[k][1] - bcy))
            ax, ay = ccent[anchor]
            group = [k for k in digits if abs(ccent[k][1] - ay) < 1.0 * cstats[anchor][3]
                     and abs(ccent[k][0] - ax) < 4.0 * cstats[anchor][2]]
            gx0 = min(cstats[k][0] for k in group)
            gy0 = min(cstats[k][1] for k in group)
            gx1 = max(cstats[k][0] + cstats[k][2] for k in group)
            gy1 = max(cstats[k][1] + cstats[k][3] for k in group)
            tight = crop[gy0:gy1, gx0:gx1]
        else:
            tight = crop
        fid = _ocr_id(tight, pytesseract)
        out.append(FlakeAnnotation(flake_id=fid, x=x, y=y, w=w, h=h))
    return out


def annotate_preview(
    img: np.ndarray, annotations: list[FlakeAnnotation], max_width: int = 1600
) -> np.ndarray:
    """Draw detected boxes + their read IDs for the user to verify (RGB)."""
    vis = img.copy()
    for a in annotations:
        label = str(a.flake_id) if a.flake_id is not None else "?"
        cv2.rectangle(vis, (a.x, a.y), (a.x + a.w, a.y + a.h), (0, 0, 255), 6)
        cv2.putText(vis, label, (a.x, a.y - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 7)
    h, w = img.shape[:2]
    vis = cv2.resize(vis, (max_width, int(max_width * h / w)))
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
