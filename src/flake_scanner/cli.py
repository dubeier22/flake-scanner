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

from .calibration.build import build_from_mosaic
from .calibration.notion import NotionThickness
from .calibration.store import CalibrationStore
from .detection.scan import scan as scan_mosaic
from .io.report import write_csv, write_map

app = typer.Typer(add_completion=False, help="Detect 2D-material flakes in microscope mosaics.")


def _parse_bgr(s: str) -> tuple[int, int, int]:
    b, g, r = (int(v) for v in s.split(","))
    return (b, g, r)


@app.command()
def calibrate(
    image: Path = typer.Option(..., help="Annotated mosaic (numbers typed in the mark colour)."),
    db: Path = typer.Option(..., help="Calibration database CSV (created/appended)."),
    material: str = typer.Option(..., help="Material name, e.g. hBN."),
    substrate: str = typer.Option("SiO2-285nm", help="Substrate name."),
    style: str = typer.Option("box", help="'box' (red box + number, recommended) or 'number'."),
    mode: str = typer.Option("thickness", help="'thickness' (label=nm) or 'id' (label=flake-id)."),
    chip: str = typer.Option("", help="Chip letter (for mode=id, and provenance)."),
    notion: Path = typer.Option(None, help="Notion CSV export (required for mode=id)."),
    date: str = typer.Option("", help="Date key YYYY_MM_DD (for mode=id Notion lookup)."),
    mark_color: str = typer.Option("0,0,255", help="Mark colour as B,G,R (default pure red)."),
) -> None:
    """Add calibration flakes by reading typed number labels off an annotated mosaic."""
    store = CalibrationStore(db)
    ntx = NotionThickness(notion) if notion else None
    added = build_from_mosaic(
        image, store, material, substrate, mode=mode, style=style,
        mark_bgr=_parse_bgr(mark_color), chip=chip, notion=ntx, date=date,
    )
    store.save()
    typer.echo(f"Added {len(added)} calibration flakes -> {db} ({len(store.records)} total).")
    if added:
        typer.echo(f"Thicknesses read: {sorted(int(t) for t in added)}")
    typer.echo("Verify this matches your annotations; re-annotate any missed flakes "
               "(thicker box lines read more reliably) and run again.")


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
