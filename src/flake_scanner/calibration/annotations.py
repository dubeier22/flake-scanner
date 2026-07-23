"""Read digital labels a researcher types onto a mosaic to mark calibration flakes.

Workflow: after AFM, the researcher opens the full-chip mosaic in any image
editor and, next to each measured flake, types a number in a single designated
colour (default pure red). The number is either the flake's **thickness in nm**
(mode="thickness", no Notion needed) or its **flake-id number** (mode="id",
joined to Notion for thickness). Because the text is clean rendered digits (not
handwriting), it reads reliably — unlike the hand-drawn labels that varied per
chip.

This module finds the coloured marks, OCRs the digits, and snaps each label to
the nearest flake blob, giving ``(value, x, y)`` per calibration flake with no
coordinate hunting.

OCR needs the optional ``pytesseract`` package and the ``tesseract`` binary
(`brew install tesseract`); a clear error is raised if unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MARK_BGR = (0, 0, 255)  # pure red


@dataclass
class Label:
    value: int  # OCR'd number (thickness nm or flake-id number, per mode)
    x: int  # snapped flake centroid, pixels
    y: int


@dataclass
class BoxLabel:
    """A red box the researcher drew around one flake, plus its number label."""

    value: int  # OCR'd number (thickness nm or flake-id number, per mode)
    cx: int  # box centre, pixels
    cy: int
    x: int  # box top-left, pixels
    y: int
    w: int  # box size, pixels
    h: int


def _mark_mask(img: np.ndarray, mark_bgr: tuple[int, int, int], tol: int = 60) -> np.ndarray:
    """Robust mask of the annotation colour, tolerant of JPEG/anti-alias drift.

    Detects by hue (generous saturation/value) so a hand-drawn colour still
    registers, then closes small gaps so thin box lines form a connected
    outline. ``tol`` is the hue half-window in degrees-ish (OpenCV hue units).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_h = int(cv2.cvtColor(np.uint8([[mark_bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    ht = max(8, tol // 4)  # hue tolerance
    ranges = [(max(0, target_h - ht), min(179, target_h + ht))]
    if target_h - ht < 0:  # red wraps around 0/180
        ranges.append((180 + (target_h - ht), 179))
    if target_h + ht > 179:
        ranges.append((0, (target_h + ht) - 180))
    m = np.zeros(img.shape[:2], np.uint8)
    for lo_h, hi_h in ranges:
        m |= cv2.inRange(hsv, (lo_h, 70, 70), (hi_h, 255, 255))
    # bridge breaks in thin/JPEG-fragmented lines without filling box interiors
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _ocr_digits(glyph_img: np.ndarray) -> int | None:
    try:
        import pytesseract
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "digital-label calibration needs OCR: pip install pytesseract and "
            "install the tesseract binary (brew install tesseract)"
        ) from e
    txt = pytesseract.image_to_string(
        glyph_img,
        config="--psm 8 -c tessedit_char_whitelist=0123456789",
    ).strip()
    digits = "".join(ch for ch in txt if ch.isdigit())
    return int(digits) if digits else None


def _snap_to_flake(
    img: np.ndarray, mark: np.ndarray, cx: int, cy: int, window: int, min_area: int
) -> tuple[int, int] | None:
    """Nearest flake blob to (cx, cy), excluding the coloured mark pixels."""
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - window), min(w, cx + window)
    y0, y1 = max(0, cy - window), min(h, cy + window)
    win = img[y0:y1, x0:x1].astype(np.float32)
    sub = np.median(win.reshape(-1, 3), axis=0)
    flake = (np.linalg.norm(win - sub, axis=2) > 40) & (mark[y0:y1, x0:x1] == 0)
    flake = cv2.morphologyEx(flake.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(flake)
    lx, ly = cx - x0, cy - y0
    best, best_d = None, np.inf
    for i in range(1, n):
        if stats[i][4] < min_area:
            continue
        d = np.hypot(cent[i][0] - lx, cent[i][1] - ly)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    return int(cent[best][0]) + x0, int(cent[best][1]) + y0


def read_labels(
    img: np.ndarray,
    mark_bgr: tuple[int, int, int] = DEFAULT_MARK_BGR,
    tol: int = 60,
    cluster_dist: int = 90,
    snap_window: int = 200,
    min_flake_area: int = 120,
) -> list[Label]:
    """Detect coloured number labels, OCR them, and snap each to its flake.

    ``cluster_dist`` groups the strokes/digits of one number; ``snap_window`` is
    how far from a label to look for its flake. Returns one ``Label`` per number
    successfully read and snapped.
    """
    mask = _mark_mask(img, mark_bgr, tol)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    strokes = [(cent[i][0], cent[i][1], stats[i]) for i in range(1, n) if stats[i][4] >= 20]
    if not strokes:
        return []

    # cluster nearby stroke components into per-number groups
    pts = np.array([[s[0], s[1]] for s in strokes])
    used = [False] * len(strokes)
    labels: list[Label] = []
    for i in range(len(strokes)):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(strokes)):
                if used[j]:
                    continue
                if any(np.hypot(*(pts[j] - pts[k])) < cluster_dist for k in grp):
                    grp.append(j)
                    used[j] = True
                    changed = True
        xs = [int(strokes[k][2][0]) for k in grp]
        ys = [int(strokes[k][2][1]) for k in grp]
        xe = [int(strokes[k][2][0] + strokes[k][2][2]) for k in grp]
        ye = [int(strokes[k][2][1] + strokes[k][2][3]) for k in grp]
        gx0, gy0, gx1, gy1 = min(xs), min(ys), max(xe), max(ye)
        pad = 6
        glyph = mask[max(0, gy0 - pad) : gy1 + pad, max(0, gx0 - pad) : gx1 + pad]
        value = _ocr_digits(cv2.bitwise_not(glyph))  # black digits on white for OCR
        if value is None:
            continue
        cx, cy = int(np.mean([pts[k][0] for k in grp])), int(np.mean([pts[k][1] for k in grp]))
        snapped = _snap_to_flake(img, mask, cx, cy, snap_window, min_flake_area)
        if snapped is None:
            continue
        labels.append(Label(value=value, x=snapped[0], y=snapped[1]))
    return labels


def read_boxes(
    img: np.ndarray,
    mark_bgr: tuple[int, int, int] = DEFAULT_MARK_BGR,
    tol: int = 60,
    min_box_px: int = 60,
) -> list[BoxLabel]:
    """Detect red boxes drawn around flakes, OCR each box's number label.

    A box is a large, hollow rectangular red component; number labels are the
    remaining small red components, clustered and OCR'd, then paired to the
    nearest box. Unambiguous vs typed numbers alone: the box says exactly which
    flake and its extent, so there is no snap-to-nearest guessing. Returns one
    ``BoxLabel`` per box that has a readable number nearby.
    """
    mask = _mark_mask(img, mark_bgr, tol)
    n, comp_lab, stats, cent = cv2.connectedComponentsWithStats(mask)

    boxes: list[tuple[int, int, int, int]] = []  # x, y, w, h
    strokes: list[tuple[int, float, float, tuple]] = []  # comp_index, cx, cy, stats
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) >= min_box_px:
            # a box is a hollow rectangle: its central region holds little mark ink
            interior = mask[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]
            if interior.size and interior.mean() / 255 < 0.12:
                boxes.append((int(x), int(y), int(w), int(h)))
                continue
        if area >= 20:
            strokes.append((i, cent[i][0], cent[i][1], stats[i]))

    out: list[BoxLabel] = []
    for bx, by, bw, bh in boxes:
        bcx, bcy = bx + bw // 2, by + bh // 2
        # the number is written above the box (roughly x-aligned); search a band
        # there, generous vertically to cover the large fixed-size font.
        rx0, rx1 = bx - bw * 0.6, bx + bw * 1.6
        ry0, ry1 = by - max(bh * 2.5, 380), by + bh * 0.25
        near = [s for s in strokes if rx0 <= s[1] <= rx1 and ry0 <= s[2] <= ry1]
        if not near:  # fall back to any strokes close to the box
            near = [
                s for s in strokes if np.hypot(s[1] - bcx, s[2] - bcy) < 1.5 * max(bw, bh)
            ]
        if not near:
            continue
        gx0 = min(int(s[3][0]) for s in near)
        gy0 = min(int(s[3][1]) for s in near)
        gx1 = max(int(s[3][0] + s[3][2]) for s in near)
        gy1 = max(int(s[3][1] + s[3][3]) for s in near)
        pad = 8
        # isolate just these strokes (drop any box-outline ink caught in the crop)
        sub = np.zeros_like(mask[max(0, gy0 - pad) : gy1 + pad, max(0, gx0 - pad) : gx1 + pad])
        y_off, x_off = max(0, gy0 - pad), max(0, gx0 - pad)
        for s in near:
            sub[comp_lab[y_off : gy1 + pad, x_off : gx1 + pad] == s[0]] = 255
        value = _ocr_digits(cv2.bitwise_not(sub))
        if value is None:
            continue
        out.append(BoxLabel(value=value, cx=bcx, cy=bcy, x=bx, y=by, w=bw, h=bh))
    return out

