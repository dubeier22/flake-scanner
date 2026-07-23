# flake-scanner

Material-agnostic detection of 2D-material flakes (graphite, hBN, TMDs) in
reflected-light microscope mosaics. Given a full-chip mosaic, it flags flakes
matching a target material and thickness so you inspect a short ranked list
instead of scanning the whole chip by eye.

Detection uses **relative colour contrast** against the local substrate
(`normalized_difference = (substrate − flake) / (substrate + flake)`, per
channel), which is invariant to exposure and white-balance drift between
sessions. A material's target-thickness signature is the mean + covariance of
its calibration flakes' contrast; unknown flakes are scored by **Mahalanobis
distance** to that signature. See `docs/ARCHI.md` and
`docs/6-memo/contrast-metric-analysis.md`.

## Install

Runs in the lab conda env; install the package into it (editable):

```bash
conda activate flakes           # numpy pandas opencv typer scikit-image
pip install -e .
# optional, for the digital-label calibration input:
brew install tesseract && pip install pytesseract
```

## Usage

**1. Calibrate** — teach it what your target flakes look like. In any image
editor, type each AFM-measured flake's **thickness (nm)** next to it on the
mosaic in a single designated colour (default pure red). Then:

```bash
flake-scanner calibrate --image chipB_annotated.jpg --db calib.csv \
    --material hBN --mode thickness --chip B
```

It reads the numbers (OCR), samples each flake's colour in-domain, and appends
to the calibration database. Repeat per chip — the database grows over time.
(Alternatively `--mode id` reads flake-id numbers and joins thickness from a
Notion CSV export via `--notion` / `--date`.)

**2. Scan** — find matching flakes on a new mosaic:

```bash
flake-scanner scan --image chipC.jpg --db calib.csv \
    --material hBN --min-thickness 35 --max-thickness 45 \
    --pixel-scale 0.30 --min-size 23
```

Outputs to `scan_results/`: a ranked `*_candidates.csv` (rank, size µm, match
score, pixel coordinates) and a `*_candidate_map.jpg` with each candidate boxed
and numbered — a navigation map for the microscope.

## Status & limitations

- Validated on hBN: the colour signature reliably separates target-thickness
  flakes from thinner/thicker ones; the ranked shortlist surfaces clean,
  correctly-sized candidates.
- Colour cannot detect cracked/folded flakes (they reflect abnormally), and
  exact sizing of small *touching* flakes is approximate (needs instance
  segmentation — future work).
- Pixel scale is currently supplied via `--pixel-scale`; reading it from a
  burned-in scale bar is planned.

## Development

```bash
ruff check .
pytest
```
