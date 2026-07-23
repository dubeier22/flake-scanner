# Testing Guidelines

## Test Framework

pytest (target direction — no tests exist yet as of TRIP init). Coverage
via `pytest-cov`.

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_detection.py::test_contrast_positive_when_flake_darker

# With coverage
pytest --cov=flake_scanner --cov-report=term-missing
```

## Test Organization

Tests live under `tests/`, mirroring the source layout once
`flake_finder.py` is restructured into a package per `docs/ARCHI.md`
Section 4 (e.g. `tests/test_calibration.py`, `tests/test_detection.py`,
`tests/test_viewer.py`). Fixture images (small, synthetic or cropped —
never full multi-hundred-MB mosaics) live in `tests/fixtures/`. Test
functions are named `test_<function>_<scenario>`.

## Writing Tests

- Test the pure/quantitative functions directly: `contrast()`,
  `score_region()`, `auto_substrate()`, `sample_patch()`,
  `load_calibration()` — these have clear expected outputs given known
  inputs and don't require the viewer or real files.
- For `build_mask()` (strip-based processing), use small synthetic images
  and assert no seam artifacts appear at strip boundaries.
- Prefer a small seeded `calibration_data.csv` fixture over relying on the
  real lab dataset in tests.
- The `ZoomViewer` (OpenCV window + keyboard/trackpad loop) is hard to
  unit test directly — test its coordinate-transform helpers
  (`img_to_screen`, `centre_img_coords`) in isolation rather than the
  interactive loop.

## Coverage Requirements

Not yet defined. Priority is correctness of the quantitative core
(contrast math, scoring, masking) over raw coverage percentage — see
`docs/4-unit-tests/COVERAGE-DEBT.md` (create when first needed) for any
deliberately deferred hard-to-cover paths, per the hard-to-cover policy in
`TRIP-test`.
