"""Build calibration records from a magenta box+ID annotated mosaic.

Reads the flake annotations (see ``annotations.read_flake_annotations``),
joins each flake's ID to a thickness the user supplies separately (the number
on the map is the flake ID, not the thickness), samples the flake colour, and
writes records with uniform flake IDs (e.g. ``HBN_2026_07_07_A01``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .annotations import BLUE_BGR, MAGENTA_BGR, FlakeAnnotation, ink_mask, read_flake_annotations
from .sampling import sample_box
from .store import CalibrationRecord, CalibrationStore, flake_id


@dataclass
class BuiltFlake:
    """Result for one annotated flake (for the verification report)."""

    id_number: int | None
    x: int
    y: int
    thickness: float | None
    added: bool


def read_annotations(image_path: str | Path) -> list[FlakeAnnotation]:
    """Read flake annotations from a mosaic (for the verification preview)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    return read_flake_annotations(img)


def build_from_id_mosaic(
    image_path: str | Path,
    store: CalibrationStore,
    material: str,
    date: str,
    chip: str,
    substrate: str,
    thicknesses: dict[int, float],
    annotations: list[FlakeAnnotation] | None = None,
) -> list[BuiltFlake]:
    """Add calibration records from an annotated mosaic; return per-flake results.

    ``thicknesses`` maps flake-ID number -> thickness (nm). ``annotations`` may
    be supplied (e.g. after the user corrected IDs in the UI); otherwise they
    are read from the image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    if annotations is None:
        annotations = read_flake_annotations(img)

    # exclude annotation ink (magenta box + blue number) from colour sampling, so
    # the box outline can't tint the flake colour. Dilate to catch anti-aliasing.
    ink = ink_mask(img, MAGENTA_BGR) | ink_mask(img, BLUE_BGR, min_sat=140)
    ink = cv2.dilate(ink, np.ones((7, 7), np.uint8))

    results: list[BuiltFlake] = []
    for ann in annotations:
        t = thicknesses.get(ann.flake_id) if ann.flake_id is not None else None
        added = False
        if t is not None:
            res = sample_box(img, ann.x, ann.y, ann.w, ann.h, exclude_mask=ink)
            if res is not None:
                flake, sub = res
                fid = flake_id(material, date, chip, ann.flake_id)
                store.add(
                    CalibrationRecord(
                        material=material.upper(), substrate=substrate,
                        chip=chip.upper(), flake_id=fid, thickness_nm=t,
                        flake_b=float(flake[0]), flake_g=float(flake[1]), flake_r=float(flake[2]),
                        sub_b=float(sub[0]), sub_g=float(sub[1]), sub_r=float(sub[2]),
                        source_image=str(image_path),
                    )
                )
                added = True
        results.append(BuiltFlake(ann.flake_id, ann.x, ann.y, t, added))
    return results
