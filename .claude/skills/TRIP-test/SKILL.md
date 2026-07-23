---
name: TRIP-test
description: Write/run tests following project standards (deep test authoring)
disable-model-invocation: true
argument-hint: "component or feature to test"
---

# Testing Mode

You are now in **testing mode** for **flake-scanner**.

This skill is the **deep test-authoring reference**: the `TRIP-2-implement` testing gate points here for heavy authoring work and full guidance. Invoke it standalone for test backfill or coverage work outside an implementation session.

## Prerequisites - Read First

Before testing, you MUST read:

1. @docs/ARCHI.md - Understand system architecture
2. @docs/4-unit-tests/TESTING.md - Testing guidelines

## Your Task

Test: $ARGUMENTS

---

## Testing Guidelines

### Scope

- Only run tests for relevant files that changed (not the whole project)
- Focus on the new feature/fix/refactor

### Commands

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_detection.py::test_contrast_positive_when_flake_darker

# With coverage
pytest --cov=flake_scanner --cov-report=term-missing
```

### Test Structure

Tests live under `tests/`, mirroring the source layout (e.g.
`tests/test_calibration.py`, `tests/test_detection.py`,
`tests/test_viewer.py` once the package is restructured per ARCHI.md
Section 4). Small synthetic/cropped fixture images live in
`tests/fixtures/` — never require multi-hundred-MB real mosaics in the
test suite. Test functions: `test_<function>_<scenario>`.

### Testing Priorities

**Unit Tests**:

- `contrast()` — known flake/substrate BGR pairs → expected per-channel contrast, including the near-zero-substrate guard (`sub_bgr > 1` check)
- `sample_patch()` — patch averaging near image edges (radius clipping)
- `auto_substrate()` — median-of-grid behavior on a synthetic image with a known substrate color and partial flake coverage
- `score_region()` — known contrast distance → expected 0–100 score, including the `std_c` weighting
- `load_calibration()` — thickness-range filtering, and the "no flakes in range, using all" fallback

**Integration Tests**:

- `build_mask()` on a small synthetic multi-strip image — verify no seam artifacts at strip boundaries and correct masking vs. a known target signature
- End-to-end `calibrate` → `scan` on fixture images with a pre-seeded small calibration CSV
- CLI mode dispatch (`calibrate`/`scan`/`both`) once restructured behind Typer

**What to Test**:

- Happy path: valid image, calibration data present and in range
- Missing/corrupt image file, missing `calibration_data.csv`
- No calibration flakes in the requested thickness range
- Thickness range boundary values (exact min/max)
- Very small candidate regions near `MIN_AREA_PX`
- Substrate sampling near image edges (illumination falloff scenario)

---

## Hard-to-Test Code

Seam ladder, cheapest first: **exported pure helper → injectable client/adapter → module mock → integration/emulator test**. Take the first rung that works; refactor for a seam only if the refactor is smaller than the feature you're shipping — otherwise it's coverage debt. Before refactoring legacy code, pin it with characterization tests (assert current behavior as-is, then refactor safely).

Uncovered risky paths: one line each in `docs/4-unit-tests/COVERAGE-DEBT.md` (`path | why hard | escape plan`). Delete a ledger line in the same change that gives its path meaningful coverage.

---

## Post-Testing Summary

After completing tests, create a summary file:

**File**: `docs/4-unit-tests/wa_vx.y.z_test.md`
(a = project week, x.y.z = version)

**Content**:

```markdown
# Test Summary - Week a, V. x.y.z

## What Was Tested

[List of tested components/functions]

## Test Results

- Total tests: X
- Passed: X
- Failed: X
- Coverage: X%

## Key Findings

[Any issues discovered, edge cases found, etc.]

## Notes

[Additional context or recommendations]
```
