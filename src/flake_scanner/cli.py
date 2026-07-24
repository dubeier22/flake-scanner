"""flake-scanner command-line interface.

    flake-scanner calibrate --image chip.jpg --db calib.csv --material hBN
    flake-scanner scan      --image chip.jpg --db calib.csv --material hBN \
                            --min-thickness 35 --max-thickness 45 --pixel-scale 0.30
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import typer

from .calibration.build import build_from_id_mosaic
from .calibration.store import CalibrationStore
from .detection.scan import scan as scan_mosaic
from .io.report import write_csv, write_map

app = typer.Typer(add_completion=False, help="Detect 2D-material flakes in microscope mosaics.")


def _parse_thicknesses(s: str) -> dict[int, float]:
    """'25,26,34' -> {1: 25.0, 2: 26.0, 3: 34.0} (position = flake ID)."""
    return {i: float(v) for i, v in enumerate(s.split(","), start=1) if v.strip()}


@app.command()
def calibrate(
    image: Path = typer.Option(..., help="Mosaic annotated with magenta box + ID per flake."),
    db: Path = typer.Option(..., help="Calibration database CSV (created/appended)."),
    material: str = typer.Option(..., help="Material, e.g. HBN."),
    date: str = typer.Option(..., help="Exfoliation date, YYYY_MM_DD, e.g. 2026_07_07."),
    chip: str = typer.Option(..., help="Chip letter, e.g. A."),
    thicknesses: str = typer.Option(
        ..., help="Comma-separated thicknesses (nm) in flake-ID order: t1,t2,...,tN."
    ),
    substrate: str = typer.Option("SiO2-285nm", help="Substrate name."),
) -> None:
    """Add calibration flakes from a magenta-annotated mosaic (ID box + thickness list)."""
    store = CalibrationStore(db)
    tmap = _parse_thicknesses(thicknesses)
    results = build_from_id_mosaic(image, store, material, date, chip, substrate, tmap)
    store.save()
    added = [r for r in results if r.added]
    typer.echo(f"Added {len(added)} flakes -> {db} ({len(store.records)} total).")
    read_ids = sorted(r.id_number for r in results if r.id_number is not None)
    typer.echo(f"Read flake IDs: {read_ids}  (expected 1..{len(tmap)})")
    missing = sorted(set(tmap) - set(read_ids))
    if missing:
        typer.echo(f"NOT read (re-annotate or check): IDs {missing}")


@app.command()
def scan(
    image: Path = typer.Option(..., help="Mosaic to scan."),
    db: Path = typer.Option(..., help="Calibration database CSV."),
    material: str = typer.Option(..., help="Material to detect."),
    substrate: str = typer.Option("SiO2-285nm", help="Substrate name."),
    min_thickness: float = typer.Option(..., help="Target thickness band low (nm)."),
    max_thickness: float = typer.Option(..., help="Target thickness band high (nm)."),
    pixel_scale: float = typer.Option(..., help="Microns per pixel of the mosaic."),
    min_size: float = typer.Option(23.0, help="Minimum flake lateral size (microns)."),
    threshold: float = typer.Option(2.0, help="Mahalanobis match cutoff (lower = stricter)."),
    out: Path = typer.Option(Path("scan_results"), help="Output directory."),
) -> None:
    """Scan a mosaic and write a ranked candidate list + annotated map."""
    store = CalibrationStore(db)
    sig = store.signature(material, substrate, min_thickness, max_thickness)
    typer.echo(f"Signature: {sig.n_flakes} flakes, mean ndi {sig.mean.round(3)}")
    img = cv2.imread(str(image))
    if img is None:
        raise typer.BadParameter(f"cannot read image: {image}")
    cands = scan_mosaic(img, sig, pixel_scale, min_size_um=min_size, match_threshold=threshold)
    stem = Path(image).stem
    write_csv(cands, out / f"{stem}_candidates.csv")
    write_map(img, cands, out / f"{stem}_candidate_map.jpg")
    typer.echo(f"{len(cands)} candidates -> {out}/{stem}_candidates.csv + _candidate_map.jpg")


@app.command()
def ui() -> None:
    """Launch the local web interface (Scan / Calibrate) in your browser."""
    app_path = Path(__file__).with_name("app.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except FileNotFoundError:
        typer.echo("Streamlit is not installed. Run: pip install streamlit")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
