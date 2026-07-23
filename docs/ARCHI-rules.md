# Architecture Documentation Rules

[ARCHI.md](ARCHI.md) documents the flake-scanner architecture. After each
task (new feature, refactor, bug fix), determine if ARCHI.md needs updating.

## When to Update

Update after ANY change that alters:

- Project structure (e.g. the flat single-script layout moving toward the
  target `src/flake_scanner/` package structure)
- Technology stack (new dependencies, e.g. adopting Typer, or dropping a
  planned tool)
- The calibration data schema (`calibration_data.csv` columns)
- The detection pipeline (preprocessing, substrate sampling, mask-building,
  scoring) or its known findings (e.g. channel reliability for a material)
- The `ZoomViewer` control scheme
- Configuration/constants that move from hardcoded to configurable
- Roadmap items (mark as done, revise scope, or add newly discovered ones)

## How to Update by Change Type

### Major Feature / Refactor

Review: Project Structure, Core Architecture Principles, Detection
Pipeline, Calibration Data Model, Roadmap

### Minor Feature / Enhancement

Update: the specific section it touches (e.g. Configuration, Command
Structure, Image Format Handling)

### Bug Fix

Usually no update needed, unless it reveals/fixes an architectural flaw
(e.g. a wrong assumption in the contrast formula, or a memory-handling bug
in strip processing)

### Dependency Changes

Update: Technology Stack, and any affected architectural sections

### Calibration Data Changes

Update: Calibration Data Model, and Key Findings From Existing Calibration
Data if new findings emerge (e.g. channel reliability for a new material)

## Guidelines

- Be precise and factual — reflect the actual codebase, not the aspirational
  target unless clearly labeled "Target direction"
- Be concise — enough detail to understand, not implementation specifics
- Update the Mermaid data-flow diagram when the pipeline stages change
- Reference actual file paths
- Keep Roadmap items honest — move completed items out of Roadmap and into
  the relevant architecture section rather than leaving them listed as
  both "done" and "planned"
