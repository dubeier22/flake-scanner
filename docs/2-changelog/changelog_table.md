# Changelog Table

| Version   | Week | Commit Message                  |
| --------- | ---- | -------------------------------- |
| `0.1.0`   | 1    | chore: initialize TRIP workflow |

# Changelog Summary

- **v0.1.0 (TRIP Initialization - Week 1, 22-07-2026)**:
  - **Setup**: Initialized TRIP workflow with docs structure
  - **Documentation**: Generated `docs/ARCHI.md` documenting the existing `flake_finder.py` script (calibration/scan/viewer pipeline, calibration data schema, known channel-reliability findings, and the multi-material/batch-import roadmap) as a CLI Tool with an image-analysis-pipeline aspect
  - **Tooling direction**: conda stays for environment management; ruff/mypy/pytest and Typer set as the target restructuring direction for the currently single-file script
  - **Repo hygiene**: added `.gitignore` for large untracked binary data (raw microscope images, generated scan outputs, `.DS_Store`) while keeping `calibration_data.csv` and `flake_finder.py` tracked
  - **Files Added**: `docs/ARCHI.md`, `docs/ARCHI-rules.md`, `docs/2-changelog/changelog_table.md`, `docs/4-unit-tests/TESTING.md`, `.gitignore`
