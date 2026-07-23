"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """A detected candidate flake region on a scanned mosaic."""

    rank: int
    maha: float  # Mahalanobis distance to the target signature (lower = better)
    x: int  # bbox top-left, pixels
    y: int
    w_px: int
    h_px: int
    cx: int  # centroid, pixels
    cy: int
    w_um: float  # size in microns (via pixel scale)
    h_um: float
