"""The learned colour signature of a material at a target thickness band.

A signature is the mean and covariance of the normalized-difference (ndi)
contrast vectors of the calibration flakes in a chosen material / substrate /
thickness band. Detection scores an unknown flake by its Mahalanobis distance
to this signature, which weights the BGR channels by their reliability and
correlations (see docs/6-memo/contrast-metric-analysis.md — the green channel
signals "is this the material", the red channel encodes thickness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaterialSignature:
    """Mean + covariance of ndi contrast for a target band; scores by Mahalanobis."""

    mean: np.ndarray  # (3,) BGR
    cov: np.ndarray  # (3, 3)
    inv_cov: np.ndarray  # (3, 3)
    n_flakes: int
    material: str
    substrate: str
    tmin: float
    tmax: float

    @classmethod
    def from_ndi(
        cls,
        ndi_samples: np.ndarray,
        material: str,
        substrate: str,
        tmin: float,
        tmax: float,
        reg: float = 1e-4,
    ) -> "MaterialSignature":
        """Build a signature from an ``(N, 3)`` array of ndi BGR vectors.

        ``reg`` regularises the covariance so it stays invertible with few or
        collinear samples.
        """
        ndi_samples = np.asarray(ndi_samples, dtype=np.float64)
        if ndi_samples.ndim != 2 or ndi_samples.shape[1] != 3:
            raise ValueError("ndi_samples must be (N, 3)")
        if len(ndi_samples) < 3:
            raise ValueError(
                f"need >=3 calibration flakes for {material}/{substrate} "
                f"in {tmin}-{tmax}nm, got {len(ndi_samples)}"
            )
        mean = ndi_samples.mean(axis=0)
        cov = np.cov(ndi_samples.T) + np.eye(3) * reg
        return cls(
            mean=mean,
            cov=cov,
            inv_cov=np.linalg.inv(cov),
            n_flakes=len(ndi_samples),
            material=material,
            substrate=substrate,
            tmin=tmin,
            tmax=tmax,
        )

    def mahalanobis(self, ndi: np.ndarray) -> float:
        """Mahalanobis distance of a single ndi BGR vector to the signature."""
        d = np.asarray(ndi, dtype=np.float64) - self.mean
        return float(np.sqrt(d @ self.inv_cov @ d))

    def mahalanobis_field(self, ndi_img: np.ndarray) -> np.ndarray:
        """Per-pixel Mahalanobis distance for an ``(H, W, 3)`` ndi array."""
        d = np.asarray(ndi_img, dtype=np.float64) - self.mean
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.inv_cov, d))
