"""Parse a Notion CSV export of AFM measurements into a thickness lookup.

The lab logs each AFM-confirmed flake in Notion with a Flake ID (chip letter +
number, e.g. ``B1``), an exfoliation date, and a measured thickness. This maps
``(date_key, flake_id) -> thickness`` so the calibration builder can join a
flake's identity (read off an annotated mosaic) to its AFM thickness without
manual entry. Flake IDs are NOT globally unique (they recur across
exfoliations), so lookups are always scoped by date.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Notion writes dates like "July 7, 2026"; folders/CLI use "2026_07_07".
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


def date_key(notion_date: str) -> str | None:
    """Convert a Notion date string to a ``YYYY_MM_DD`` key, or None."""
    m = re.match(r"([A-Za-z]+)\s+(\d+),\s*(\d{4})", notion_date.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if not month:
        return None
    return f"{int(m.group(3))}_{month:02d}_{int(m.group(2)):02d}"


def chip_of(flake_id: str) -> str:
    """The chip letter(s) prefix of a flake id (``B1`` -> ``B``)."""
    m = re.match(r"([A-Za-z]+)", flake_id.strip())
    return m.group(1) if m else ""


class NotionThickness:
    """Thickness lookup by ``(date_key, flake_id)`` from a Notion CSV export."""

    def __init__(self, csv_path: str | Path):
        self.by_key: dict[tuple[str, str], float] = {}
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fid = row.get("Flake ID", "").strip()
                dk = date_key(row.get("Date of Exfoliation", ""))
                try:
                    t = float(row.get("Thickness (nm)", ""))
                except ValueError:
                    continue
                if fid and dk:
                    self.by_key[(dk, fid)] = t

    def get(self, date: str, flake_id: str) -> float | None:
        """Thickness (nm) for a flake on a given date key, or None if unknown."""
        return self.by_key.get((date, flake_id))
