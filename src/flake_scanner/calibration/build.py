"""Build calibration records from a digitally-labelled mosaic.

Reads the coloured number labels off an annotated mosaic (see
``annotations.read_labels``), samples each flake's colour in-domain, and writes
calibration records. The label number is interpreted as the thickness directly
(``mode="thickness"``) or as a flake-id number joined to Notion for thickness
(``mode="id"``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .annotations import DEFAULT_MARK_BGR, _mark_mask, read_boxes, read_labels
from .notion import NotionThickness
from .sampling import sample_box, snap_sample
from .store import CalibrationRecord, CalibrationStore


def build_from_mosaic(
    image_path: str | Path,
    store: CalibrationStore,
    material: str,
    substrate: str,
    mode: str = "thickness",
    style: str = "box",
    mark_bgr: tuple[int, int, int] = DEFAULT_MARK_BGR,
    chip: str = "",
    notion: NotionThickness | None = None,
    date: str = "",
) -> list[float]:
    """Add calibration records from an annotated mosaic; return the thicknesses added.

    ``style="box"``: each flake is enclosed in a red box with a number label
    (unambiguous — recommended). ``style="number"``: just a number typed next to
    each flake (snapped to the nearest flake).
    ``mode="thickness"``: the label number IS the thickness (nm).
    ``mode="id"``: the label number is the flake-id number; thickness is looked
    up in ``notion`` using ``date`` and ``chip`` (id = ``f"{chip}{value}"``).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    # dilate the annotation ink so its anti-aliased edges are excluded from sampling
    ink = cv2.dilate(_mark_mask(img, mark_bgr, tol=60), np.ones((7, 7), np.uint8))
    marks = read_boxes(img, mark_bgr=mark_bgr) if style == "box" else read_labels(img, mark_bgr=mark_bgr)

    added: list[float] = []
    for mk in marks:
        if mode == "thickness":
            flake_id = f"{chip}?" if chip else "?"
            thickness = float(mk.value)
        elif mode == "id":
            if notion is None:
                raise ValueError("mode='id' requires a Notion lookup")
            flake_id = f"{chip}{mk.value}"
            t = notion.get(date, flake_id)
            if t is None:
                continue  # no AFM thickness on record for this flake
            thickness = t
        else:
            raise ValueError(f"unknown mode: {mode}")

        if style == "box":
            res = sample_box(img, mk.x, mk.y, mk.w, mk.h, exclude_mask=ink)
        else:
            res = snap_sample(img, mk.x, mk.y, exclude_mask=ink)
        if res is None:
            continue
        flake, sub = res
        store.add(
            CalibrationRecord(
                material=material, substrate=substrate, chip=chip, flake_id=flake_id,
                thickness_nm=thickness,
                flake_b=float(flake[0]), flake_g=float(flake[1]), flake_r=float(flake[2]),
                sub_b=float(sub[0]), sub_g=float(sub[1]), sub_r=float(sub[2]),
                source_image=str(image_path),
            )
        )
        added.append(thickness)
    return added
