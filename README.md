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

## Web interface (easiest)

```bash
pip install streamlit
flake-scanner ui
```

Opens a local app in your browser with two tabs: **Scan** (pick a material +
thickness range, choose a mosaic, get the ranked candidate map + table) and
**Calibrate** (add flakes from a boxed/annotated mosaic). Everything runs
locally, so full-resolution mosaics never leave your machine.

## Command-line usage

**1. Calibrate** — teach it what your target flakes look like. In any image
editor, **draw a magenta (#FF00FF) box around each AFM-measured flake** and
write a magenta **flake-ID number** (1, 2, 3, …) next to it — no connecting
leader lines. Magenta is used because it is absent from the samples themselves
(flakes are yellow/green/red/pink), so it isolates cleanly; red clashes with
reddish flakes. Supply the thicknesses separately, in flake-ID order:

```bash
flake-scanner calibrate --image chipA_annotated.jpg --db calib.csv \
    --material HBN --date 2026_07_07 --chip A \
    --thicknesses 120,97,250,105,115,25,34,113,26,34,109,108
```

It locates each flake, reads its ID, joins the matching thickness, samples the
colour, and writes records with uniform IDs like `HBN_2026_07_07_A01`
(re-importing a chip updates rather than duplicates). The **web UI** shows a
verification preview so you can correct any misread ID before saving.

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
