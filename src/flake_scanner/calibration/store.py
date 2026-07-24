"""Read/write the calibration database and build material signatures.

The calibration DB is a CSV that grows over time. Each row is one AFM-confirmed
flake: its material / substrate / thickness plus the sampled flake and local
substrate colours (raw BGR, so any contrast metric can be recomputed later).
Signatures are derived on demand by filtering to a material + substrate +
thickness band and summarising the normalized-difference contrast.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sampling import normalized_difference
from .signature import MaterialSignature

FIELDS = [
    "material",
    "substrate",
    "chip",
    "flake_id",
    "thickness_nm",
    "flake_b",
    "flake_g",
    "flake_r",
    "sub_b",
    "sub_g",
    "sub_r",
    "source_image",
]


def chip_id(material: str, date: str, chip: str) -> str:
    """Uniform chip identifier, e.g. HBN_2026_07_07_A."""
    return f"{material.upper()}_{date}_{chip.upper()}"


def flake_id(material: str, date: str, chip: str, n: int) -> str:
    """Uniform flake identifier, e.g. HBN_2026_07_07_A01."""
    return f"{chip_id(material, date, chip)}{n:02d}"


@dataclass
class CalibrationRecord:
    material: str
    substrate: str
    chip: str
    flake_id: str
    thickness_nm: float
    flake_b: float
    flake_g: float
    flake_r: float
    sub_b: float
    sub_g: float
    sub_r: float
    source_image: str

    @property
    def ndi(self) -> np.ndarray:
        return normalized_difference(
            [self.flake_b, self.flake_g, self.flake_r],
            [self.sub_b, self.sub_g, self.sub_r],
        )


class CalibrationStore:
    """A CSV-backed calibration database keyed by material/substrate/thickness."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[CalibrationRecord] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.records.append(
                    CalibrationRecord(
                        material=row["material"],
                        substrate=row["substrate"],
                        chip=row["chip"],
                        flake_id=row["flake_id"],
                        thickness_nm=float(row["thickness_nm"]),
                        flake_b=float(row["flake_b"]),
                        flake_g=float(row["flake_g"]),
                        flake_r=float(row["flake_r"]),
                        sub_b=float(row["sub_b"]),
                        sub_g=float(row["sub_g"]),
                        sub_r=float(row["sub_r"]),
                        source_image=row.get("source_image", ""),
                    )
                )

    def add(self, rec: CalibrationRecord) -> None:
        """Add or update a record; flake_id is the unique key (no duplicates)."""
        self.records = [r for r in self.records if r.flake_id != rec.flake_id]
        self.records.append(rec)

    def save(self) -> None:
        """Atomically write the DB (temp file + replace, to avoid corruption)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        tmp.replace(self.path)

    def signature(
        self, material: str, substrate: str, tmin: float, tmax: float
    ) -> MaterialSignature:
        """Build a signature from rows matching material + substrate + band."""
        ndi = [
            r.ndi
            for r in self.records
            if r.material == material
            and r.substrate == substrate
            and tmin <= r.thickness_nm <= tmax
        ]
        if not ndi:
            raise ValueError(
                f"no calibration flakes for {material}/{substrate} in {tmin}-{tmax}nm"
            )
        return MaterialSignature.from_ndi(
            np.array(ndi), material, substrate, tmin, tmax
        )
